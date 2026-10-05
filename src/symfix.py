"""
symfix.py -- make generation use the SAME cell basis the model was trained in.

THE BUG THIS FIXES
------------------
DATA.py stores every MP-20 structure "as_is" (CELL_SETTING = "as_is"), which for
centred groups (F, I, C, A, R) is usually a PRIMITIVE cell, and it stores the
Wyckoff projectors (P, o), the in-cell multiplicities and the symmetry
operations in THAT cell's fractional basis.  DATA.py even warns about this:

    "Expand orbits with the SAVED per-structure sym_rot/sym_trans (valid in the
     stored basis), NOT with SpaceGroup.from_int_number(n).symmetry_ops
     (conventional setting) -- mixing the two produces wrong structures."

SITETOKENS.py nevertheless builds `sym_rot_lut` / `_SG_OPS_CACHE` from
`SpaceGroup.from_int_number(sg).symmetry_ops` (CONVENTIONAL basis, including the
centring translations), and `project_to_crystal_family` imposes CONVENTIONAL
metrics (e.g. 90 deg for cubic) on a lattice head that was trained on the stored
(e.g. 60 deg rhombohedral-primitive) cells.  At sampling time the decoder
therefore projects coordinates with stored-basis projectors and expands them with
conventional-basis operations.  Consequences seen in the evaluation log:

  * atoms/cell far above MP-20 (mean 35.9, max 256; Fm-3m up to 144):
    centring translations multiply every orbit by 2-4, and a point that is
    special in the stored basis is generally NOT special in the conventional
    one, so it gets the general-position multiplicity (up to 192);
  * too few atoms on special positions (55% vs 77%) -- same reason;
  * wrong stoichiometry -> lower SMACT validity, worst for F-centred groups
    (Fm-3m 53%);
  * near-coincident images -> conflicts -> cell inflation (VPA 1.24x median);
  * Pnma 0% match (stored orthorhombic cells are often in a non-standard
    setting, so stored projectors and standard operations disagree).

THE FIX (no re-mining, no change to the checkpoint)
---------------------------------------------------
For every space group, take the operation set that the training data actually
used most often (the "modal stored basis"), build the Wyckoff codebook ONLY from
structures stored in that same basis, and use those operations everywhere the
model expands, counts multiplicities or checks conflicts.  The lattice is
symmetrised by averaging its metric tensor over the point group,
    G_sym = (1/|G|) sum_R  R^T G R ,      G = L L^T  (rows of L = lattice vectors),
which is the exact orthogonal projection onto metrics invariant under the group
and works in any basis (it reduces to the usual crystal-family constraints in
the standard setting).  Everything the decoder and lattice head were trained on
(Stage 1) is in the stored basis, so this makes sampling consistent with
training.  Generated cells then have MP-20-like atom counts by construction.

USAGE
-----
    import symfix
    report = symfix.install_stored_basis_symmetry(model, [train_ds],
                                                  module=<SITETOKENS module>)
    symfix.roundtrip_check([test_ds], report)     # no network, no GPU needed

Call it AFTER any load_state_dict / load_checkpoint (the codebook buffers are
persistent and a later checkpoint load would overwrite them).
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict

import numpy as np
import torch

MAX_OPS = 192


# ---------------------------------------------------------------------------
# operation-set signatures
# ---------------------------------------------------------------------------
def _canon_ops(rot, trans, decimals=3):
    """Integer rotations + translations reduced to [0,1), as a sorted tuple."""
    rot = np.rint(np.asarray(rot, dtype=np.float64).reshape(-1, 3, 3)).astype(np.int64)
    t = np.asarray(trans, dtype=np.float64).reshape(-1, 3)
    t = np.round(t - np.floor(t), decimals)
    t[t >= 1.0 - 0.5 * 10 ** (-decimals)] = 0.0
    keys = sorted(tuple(r.ravel().tolist()) + tuple(tt.tolist())
                  for r, tt in zip(rot, t))
    return tuple(keys)


def _is_closed_group(rot, trans, tol=1e-3):
    """True if the op set is closed under composition (mod lattice translations)."""
    rot = np.rint(np.asarray(rot, dtype=np.float64)).astype(np.int64)
    trans = np.asarray(trans, dtype=np.float64)
    keys = {tuple(r.ravel()) + tuple(np.round(t % 1.0, 3) % 1.0) for r, t in zip(rot, trans)}
    n = len(rot)
    if n > 64:                       # large groups: check a random subset of products
        idx = np.random.default_rng(0).choice(n, size=(512, 2))
    else:
        idx = np.array([(i, j) for i in range(n) for j in range(n)])
    for i, j in idx:
        R = rot[i] @ rot[j]
        t = rot[i] @ trans[j] + trans[i]
        k = tuple(R.ravel()) + tuple(np.round(t % 1.0, 3) % 1.0)
        if k not in keys:
            # tolerate rounding at the 1.0/0.0 boundary
            k2 = tuple(R.ravel()) + tuple(np.round((t + tol) % 1.0 - tol, 3) % 1.0)
            if k2 not in keys:
                return False
    return True


# ---------------------------------------------------------------------------
# mining
# ---------------------------------------------------------------------------
def mine_stored_basis_ops(datasets, verbose=True):
    """Per space group: the modal operation set among training structures.

    Returns (table, family) where
      table[sg]  = dict(rot, trans, sig, n_family, n_total, n_variants, closed)
      family     = set of (dataset_index, graph_index) stored in the modal basis
    Only structures whose mined space group agrees with the dataset's space-group
    label and whose symmetry analysis succeeded are used.
    """
    per_sg = defaultdict(Counter)
    reps = {}
    members = defaultdict(list)
    for di, ds in enumerate(datasets):
        sgs = getattr(ds, "space_groups", None)
        for gi, g in enumerate(ds.graph_data):
            sg_m = int(getattr(g, "space_group", 1) or 1)
            sg_l = int(sgs[gi]) if sgs is not None else sg_m
            if sg_m != sg_l or getattr(g, "symmetry_ok", True) is False:
                continue
            rot = np.asarray(getattr(g, "sym_rot", np.eye(3)[None])).reshape(-1, 3, 3)
            tr = np.asarray(getattr(g, "sym_trans", np.zeros((1, 3)))).reshape(-1, 3)
            if rot.shape[0] == 0:
                continue
            sig = _canon_ops(rot, tr)
            per_sg[sg_m][sig] += 1
            reps.setdefault((sg_m, sig), (rot.astype(np.float64), tr.astype(np.float64)))
            members[sg_m].append((di, gi, sig))

    table, family = {}, set()
    for sg, ctr in per_sg.items():
        sig, cnt = ctr.most_common(1)[0]
        rot, tr = reps[(sg, sig)]
        table[sg] = dict(rot=rot, trans=tr % 1.0, sig=sig, n_family=int(cnt),
                         n_total=int(sum(ctr.values())), n_variants=len(ctr),
                         closed=_is_closed_group(rot, tr))
        for di, gi, s in members[sg]:
            if s == sig:
                family.add((di, gi))

    if verbose:
        n_tot = sum(v["n_total"] for v in table.values())
        n_fam = sum(v["n_family"] for v in table.values())
        n_open = sum(1 for v in table.values() if not v["closed"])
        print(f"  [symfix] stored-basis operation sets mined for {len(table)} groups; "
              f"{n_fam}/{n_tot} structures ({100 * n_fam / max(n_tot, 1):.1f}%) are in "
              f"their group's modal basis"
              + (f"; {n_open} op sets failed the closure check" if n_open else ""))
    return table, family


class _FilteredDataset:
    """Minimal stand-in exposing only what build_wyckoff_codebook reads."""

    def __init__(self, graph_data, space_groups):
        self.graph_data = graph_data
        self.space_groups = np.asarray(space_groups, dtype=np.int32)


def _family_datasets(datasets, family):
    gd, sgs = [], []
    for di, ds in enumerate(datasets):
        s = getattr(ds, "space_groups", None)
        for gi, g in enumerate(ds.graph_data):
            if (di, gi) in family:
                gd.append(g)
                sgs.append(int(s[gi]) if s is not None else int(getattr(g, "space_group", 1)))
    return [_FilteredDataset(gd, sgs)]


# ---------------------------------------------------------------------------
# operation look-up tables
# ---------------------------------------------------------------------------
def _identity_first(ops):
    return sorted(ops, key=lambda rt: (np.abs(np.asarray(rt[0]) - np.eye(3)).sum()
                                       + np.abs(np.asarray(rt[1])).sum()))


def build_op_tables(table, module, max_ops=MAX_OPS):
    """(R, T, ok) tensors of shape (230, max_ops, ...) plus per-group op lists.

    Groups never seen in training fall back to the conventional pymatgen ops
    (they have zero prior mass and are only reachable by an explicit request).
    """
    R = torch.eye(3, dtype=torch.float32).view(1, 1, 3, 3).repeat(230, max_ops, 1, 1)
    T = torch.zeros(230, max_ops, 3, dtype=torch.float32)
    ok = torch.zeros(230, max_ops, dtype=torch.bool)
    ops_by_sg, source = {}, {}
    for sg in range(1, 231):
        if sg in table:
            ops = [(np.asarray(r, dtype=np.float64), np.asarray(t, dtype=np.float64) % 1.0)
                   for r, t in zip(table[sg]["rot"], table[sg]["trans"])]
            source[sg] = "stored"
        else:
            ops = list(module._get_sg_ops(int(sg))) or [(np.eye(3), np.zeros(3))]
            source[sg] = "conventional-fallback"
        ops = _identity_first(ops)[:max_ops]
        ops_by_sg[sg] = ops
        for i, (r, t) in enumerate(ops):
            R[sg - 1, i] = torch.as_tensor(r, dtype=torch.float32)
            T[sg - 1, i] = torch.as_tensor(t, dtype=torch.float32)
            ok[sg - 1, i] = True
    ok[:, 0] = True
    return (R, T, ok), ops_by_sg, source


# ---------------------------------------------------------------------------
# lattice: metric symmetrisation (replaces project_to_crystal_family)
# ---------------------------------------------------------------------------
def make_metric_projector(module):
    """project_to_crystal_family(p, sg) -> p, basis-agnostic.

    p: (B, >=6) lattice parameters (a, b, c, alpha, beta, gamma in degrees).
    The metric G = L L^T is averaged over the rotations of the group in the
    SAME basis the operations are expressed in.  Differentiable.
    """
    def project_to_crystal_family(p, sg):
        R_all, _, ok_all = module._SYM_TENSORS
        sgi = sg.view(-1).long().clamp(1, 230) - 1
        R = R_all.to(device=p.device, dtype=p.dtype)[sgi]            # (B,M,3,3)
        w = ok_all.to(p.device)[sgi].to(p.dtype)                      # (B,M)
        L = module.lattice_params_to_matrix(p[:, :6])                 # rows = vectors
        G = L @ L.transpose(-1, -2)
        GR = torch.einsum("bmji,bjk,bmkl->bmil", R, G, R)             # R^T G R
        Gs = (GR * w[..., None, None]).sum(1) / w.sum(1).clamp(min=1.0)[:, None, None]
        a = Gs[:, 0, 0].clamp(min=1e-8).sqrt()
        b = Gs[:, 1, 1].clamp(min=1e-8).sqrt()
        c = Gs[:, 2, 2].clamp(min=1e-8).sqrt()
        lim = 1.0 - 1e-6
        al = torch.acos((Gs[:, 1, 2] / (b * c)).clamp(-lim, lim)) * 180.0 / math.pi
        be = torch.acos((Gs[:, 0, 2] / (a * c)).clamp(-lim, lim)) * 180.0 / math.pi
        ga = torch.acos((Gs[:, 0, 1] / (a * b)).clamp(-lim, lim)) * 180.0 / math.pi
        # Build the result out-of-place.  lattice_params_to_matrix above saved
        # views of p (p[:, 0], p[:, 1], ...) for backward; writing into p here
        # would bump their version and break loss.backward() in Stage 3.
        out = torch.stack([a, b, c, al, be, ga], dim=-1)
        if p.shape[-1] > 6:
            out = torch.cat([out, p[:, 6:]], dim=-1)
        return out

    project_to_crystal_family.__doc__ = make_metric_projector.__doc__
    return project_to_crystal_family


# ---------------------------------------------------------------------------
# install
# ---------------------------------------------------------------------------
def install_stored_basis_symmetry(model, train_datasets, module, verbose=True,
                                  rebuild_codebook=True):
    """Switch a loaded DirectCrystalFlow (and its module) to stored-basis symmetry.

    Must be called after any checkpoint load.  Idempotent.
    """
    table, family = mine_stored_basis_ops(train_datasets, verbose=verbose)
    # build fallback ops BEFORE the module caches are replaced
    (R, T, ok), ops_by_sg, source = build_op_tables(table, module)

    # 1) module-level tables used by expand_generated, _get_sg_ops, _sg_ops_arrays,
    #    orbit_multiplicity (via buffers), wyckoff_multiplicity_table, resolvers
    module._SYM_TENSORS = (R, T, ok)
    module._SG_OPS_CACHE.clear()
    module._SG_OPS_ARR_CACHE.clear()
    for sg, ops in ops_by_sg.items():
        module._SG_OPS_CACHE[sg] = ops
        module._SG_OPS_ARR_CACHE[sg] = (np.stack([np.asarray(r, float) for r, _ in ops], 0),
                                        np.stack([np.asarray(t, float) for _, t in ops], 0))

    # 2) model buffers
    n_buf = 0
    for m in model.modules():
        for name, src in (("sym_rot_lut", R), ("sym_trans_lut", T), ("sym_valid_lut", ok)):
            buf = m._buffers.get(name, None)
            if buf is not None:
                buf.copy_(src.to(device=buf.device, dtype=buf.dtype))
                n_buf += 1

    # 3) lattice projection in the same basis
    module.project_to_crystal_family = make_metric_projector(module)

    # 4) Wyckoff codebook from structures stored in the modal basis only, so the
    #    projectors (P, o) agree with the operations; multiplicity table rebuilt
    #    against the new operations inside install_wyckoff_codebook.
    n_valid_before = n_valid_after = None
    if rebuild_codebook and hasattr(model, "decoder") and \
            hasattr(model.decoder, "install_wyckoff_codebook"):
        n_valid_before = int(model.decoder.wyckoff_valid.sum())
        model.decoder.install_wyckoff_codebook(_family_datasets(train_datasets, family))
        n_valid_after = int(model.decoder.wyckoff_valid.sum())

    report = dict(table=table, family=family, source=source, n_buffers=n_buf,
                  codebook_classes_before=n_valid_before,
                  codebook_classes_after=n_valid_after)
    if verbose:
        n_fb = sum(1 for s in source.values() if s != "stored")
        print(f"  [symfix] installed: {n_buf} LUT buffers updated, "
              f"{230 - n_fb} groups on stored-basis ops, {n_fb} on conventional "
              f"fallback (unseen in training); lattice projection = metric "
              f"symmetrisation")
        if n_valid_before is not None:
            print(f"  [symfix] Wyckoff codebook: {n_valid_before} -> {n_valid_after} "
                  f"(group, letter) classes (modal-basis structures only)")
        _big = sorted(((v["n_family"] / max(v["n_total"], 1), sg, v)
                       for sg, v in table.items() if v["n_total"] >= 200))[:5]
        if _big:
            print("  [symfix] lowest modal-basis coverage among common groups: "
                  + ", ".join(f"SG{sg} {100 * f:.0f}% ({v['n_variants']} bases)"
                              for f, sg, v in _big))
    return report


# ---------------------------------------------------------------------------
# network-free checks (run these first; they need no checkpoint quality)
# ---------------------------------------------------------------------------
def _orbit(f, rot, trans, L, tol_ang=0.05):
    img = (np.einsum("mij,j->mi", rot, f) + trans) % 1.0
    keep = []
    for x in img:
        if keep:
            d = np.asarray(keep) - x
            d -= np.round(d)
            if (np.linalg.norm(d @ L, axis=1) < tol_ang).any():
                continue
        keep.append(x)
    return np.asarray(keep)


def roundtrip_check(datasets, report, module=None, n_max=3000, seed=0, verbose=True):
    """Expand each structure's TRUE asymmetric unit and compare with the stored cell.

    For structures in their group's modal basis, expansion with the installed
    (stored-basis) operations must reproduce the stored atom count and
    composition exactly.  If `module` is given, the same is done with the
    conventional pymatgen operations the old code used, to quantify the bug.
    Also checks that metric symmetrisation leaves the true lattice unchanged.
    """
    table = report["table"]
    rng = np.random.default_rng(seed)
    rows = []
    for di, ds in enumerate(datasets):
        for gi, g in enumerate(ds.graph_data):
            rows.append((di, gi, g))
    if len(rows) > n_max:
        rows = [rows[i] for i in rng.choice(len(rows), n_max, replace=False)]

    stats = Counter()
    ratio_new, ratio_old, metric_err = [], [], []
    conv_cache = {}
    for di, gi, g in rows:
        sg = int(getattr(g, "space_group", 1) or 1)
        if sg not in table:
            continue
        v = table[sg]
        in_fam = _canon_ops(g.sym_rot, g.sym_trans) == v["sig"]
        L = np.asarray(g.lattice_matrix, dtype=np.float64)
        frac = np.asarray(g.frac_coords, dtype=np.float64) % 1.0
        Z = np.asarray(g.nodes)
        am = np.asarray(g.asym_unit_mask, dtype=bool)
        n_true = len(Z)
        comp_true = Counter(Z.tolist())

        def _expand(rot, trans):
            n, comp = 0, Counter()
            for f, z in zip(frac[am], Z[am]):
                k = len(_orbit(f, rot, trans, L))
                n += k
                comp[int(z)] += k
            return n, comp

        if in_fam:
            n_new, comp_new = _expand(v["rot"], v["trans"])
            stats["in_family"] += 1
            stats["new_exact"] += int(n_new == n_true and comp_new == comp_true)
            ratio_new.append(n_new / max(n_true, 1))
            G = L @ L.T
            Gs = np.mean([r.T @ G @ r for r in np.rint(v["rot"])], axis=0)
            metric_err.append(np.abs(Gs - G).max() / max(np.abs(G).max(), 1e-8))
        if module is not None:
            if sg not in conv_cache:
                try:
                    from pymatgen.symmetry.groups import SpaceGroup
                    ops = SpaceGroup.from_int_number(sg).symmetry_ops
                    conv_cache[sg] = (np.stack([o.rotation_matrix for o in ops]),
                                      np.stack([o.translation_vector for o in ops]) % 1.0)
                except Exception:
                    conv_cache[sg] = None
            if conv_cache[sg] is not None:
                n_old, comp_old = _expand(*conv_cache[sg])
                stats["old_checked"] += 1
                stats["old_exact"] += int(n_old == n_true and comp_old == comp_true)
                ratio_old.append(n_old / max(n_true, 1))

    out = dict(n_checked=len(rows), **stats,
               new_exact_rate=stats["new_exact"] / max(stats["in_family"], 1),
               old_exact_rate=(stats["old_exact"] / max(stats["old_checked"], 1)
                               if stats["old_checked"] else None),
               new_atoms_ratio_mean=float(np.mean(ratio_new)) if ratio_new else None,
               old_atoms_ratio_mean=float(np.mean(ratio_old)) if ratio_old else None,
               metric_rel_err_p99=float(np.quantile(metric_err, 0.99)) if metric_err else None)
    if verbose:
        print(f"  [symfix] round-trip on {out['n_checked']} structures "
              f"({stats['in_family']} in modal basis):")
        print(f"    stored-basis ops : exact atom count+composition "
              f"{100 * out['new_exact_rate']:.1f}%  (mean n_expanded/n_stored "
              f"{out['new_atoms_ratio_mean']})")
        if out["old_exact_rate"] is not None:
            print(f"    conventional ops : exact {100 * out['old_exact_rate']:.1f}%  "
                  f"(mean n_expanded/n_stored {out['old_atoms_ratio_mean']:.2f})  "
                  f"<- the pre-fix behaviour")
        print(f"    metric symmetrisation, true lattices: p99 relative change "
              f"{out['metric_rel_err_p99']}")
    return out
