#!/usr/bin/env python3
"""
dft_stability.py -- DFT check of the MLFF stability numbers (Table 5).

Pipeline (Materials-Project-compatible, same protocol as MatterGen / atomate2):
    MatterSim-relaxed structure  ->  MPRelaxSet relax  ->  MPRelaxSet relax (2nd)
                                 ->  MPStaticSet energy
    ->  MaterialsProject2020Compatibility corrections
    ->  E_hull against the FULL Materials Project GGA/GGA+U hull
    ->  stable / metastable / S.U.N. rates + a LaTeX row for Table 5.

Uses only pymatgen + custodian (+ mp-api for the reference hull); no workflow
database needed. Requires a licensed VASP and pymatgen's POTCAR setup.

Sub-commands (run in this order):
    python dft_stability.py select    # choose structures from emf_eval_results.json
    python dft_stability.py run --dry-run --pos 0   # check inputs for one structure
    python dft_stability.py slurm     # write submit_dft.sbatch   (or: run --all --workers N)
    python dft_stability.py status    # progress
    python dft_stability.py analyze   # E_hull, rates, Table 5 row

--------------------------------------------------------------------------------
ONE-TIME SETUP
    pip install pymatgen custodian mp-api
    # POTCARs (pymatgen needs the PBE POTCAR directory):
    pmg config -p /path/to/POTCAR_dir  /path/to/pymatgen_potcars     # see pymatgen docs
    pmg config --add PMG_VASP_PSP_DIR /path/to/pymatgen_potcars
    export MP_API_KEY="..."            # or edit MP_API_KEY below
    export VASP_CMD="srun -n 32 vasp_std"   # or "mpirun -np 32 vasp_std"

REQUIRED CHANGE TO YOUR EVALUATION SCRIPT (it currently discards the structures)
    In compute_stability_v2, replace the last lines
        for r in ok:
            r.pop("structure", None)
        out["records"] = ok
    by
        out["structures"] = [dict(idx=r["idx"], structure=r["structure"].as_dict())
                             for r in ok]
        for r in ok:
            r.pop("structure", None)
        out["records"] = ok
    and re-run the evaluation (the relaxed structures must be in the JSON).

OPTIONAL, for the novelty part of S.U.N. (novelty vs the MP-20 training set):
    # in your evaluation session, once:
    json.dump([graph_to_structure(g).as_dict() for g in tr_ds.graph_data],
              open("mp20_train_structures.json", "w"))
    # then:  python dft_stability.py analyze --novelty-ref mp20_train_structures.json
--------------------------------------------------------------------------------
"""
import argparse
import csv
import gzip
import json
import logging
import os
import pickle
import random
import shlex
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from pymatgen.core import Composition, Structure

# =============================================================================
# CONFIGURATION
# =============================================================================
MP_API_KEY = "YOUR_MP_API_KEY_HERE"          # or: export MP_API_KEY=...
VASP_CMD = os.environ.get("VASP_CMD", "mpirun -np 16 vasp_std")

RESULTS_JSON = "emf_eval_results.json"       # output of your evaluation script
SELECTION_FILE = "dft_selection.json"
WORKDIR = "dft_runs"
RESULTS_PREFIX = "dft_results"               # -> dft_results.json / .csv
MP_ENTRY_CACHE = "mp_entries_cache"          # cached reference-hull entries

SCREEN_E_HULL = 0.2      # eV/atom: run DFT on every structure with MLFF E_hull <= this
N_RANDOM_CONTROL = 30    # extra random structures ABOVE the cut, to test the screen
SEED = 0

STABLE_THR = 0.0         # eV/atom, E_hull <= this and >= 2 elements  -> "stable"
META_THR = 0.1           # eV/atom                                     -> "metastable"

log = logging.getLogger("dft_stability")

PATCH_MSG = """
The results JSON contains no structures (stability.structures is missing).
Your evaluation script deletes them before saving. In compute_stability_v2 replace

    for r in ok:
        r.pop("structure", None)
    out["records"] = ok

with

    out["structures"] = [dict(idx=r["idx"], structure=r["structure"].as_dict())
                         for r in ok]
    for r in ok:
        r.pop("structure", None)
    out["records"] = ok

and re-run the evaluation (or pass --structures-file with a JSON list of
{"idx": ..., "structure": <Structure.as_dict()>} entries).
"""


# =============================================================================
# helpers
# =============================================================================
def _api_key():
    key = os.environ.get("MP_API_KEY") or MP_API_KEY
    if not key or key.startswith("YOUR_"):
        raise SystemExit("Set your Materials Project API key: edit MP_API_KEY at the "
                         "top of this file or `export MP_API_KEY=...`.")
    return key


def _load_json(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as f:
        return json.load(f)


def _save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)


def _run_dir(workdir, entry):
    return Path(workdir) / f"idx_{int(entry['idx']):05d}"


def _load_selection(path):
    if not Path(path).exists():
        raise SystemExit(f"{path} not found -- run `select` first.")
    return _load_json(path)


def _pick_entries(sel, args):
    entries = sel["entries"]
    if getattr(args, "pos", None) is not None:
        return [entries[args.pos]]
    if getattr(args, "idx", None) is not None:
        hit = [e for e in entries if int(e["idx"]) == int(args.idx)]
        if not hit:
            raise SystemExit(f"idx {args.idx} is not in the selection")
        return hit
    if getattr(args, "all", False):
        lo = getattr(args, "start", 0) or 0
        hi = getattr(args, "stop", None)
        return entries[lo:hi]
    raise SystemExit("Choose what to run: --pos N, --idx N or --all")


def rmsd_periodic(s_before, s_after):
    """Mean atomic displacement (A) between two structures with the same atom
    order, minimum-image, measured in the relaxed cell (same definition as the
    MLFF RMSD in the evaluation script)."""
    a = s_before.get_sorted_structure()          # VASP input sets sort the same way
    if [str(x.specie) for x in a] != [str(x.specie) for x in s_after]:
        return None
    f0 = np.asarray(a.frac_coords)
    f1 = np.asarray(s_after.frac_coords)
    L = np.asarray(s_after.lattice.matrix)
    shifts = np.array([[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1)
                       for k in (-1, 0, 1)], dtype=float)
    base = (f1 - f0) @ L
    cand = base[:, None, :] + (shifts @ L)[None, :, :]
    d2 = np.einsum("nkc,nkc->nk", cand, cand).min(axis=1)
    return float(np.sqrt(d2.mean()))


# =============================================================================
# select
# =============================================================================
def cmd_select(args):
    R = _load_json(args.results)
    st = R.get("stability", {})
    recs = st.get("records", [])
    if args.structures_file:
        raw = _load_json(args.structures_file)
    else:
        raw = st.get("structures", [])
    if not recs:
        raise SystemExit("No stability records in the results JSON.")
    if not raw:
        raise SystemExit(PATCH_MSG)
    structs = {int(d["idx"]): d["structure"] for d in raw}

    rng = random.Random(args.seed)
    screened, unscreened, above, unary = [], [], [], []
    for r in recs:
        if int(r["idx"]) not in structs:
            continue
        if r.get("n_el", 0) < 2:
            unary.append(r)
        elif r.get("e_hull") is None:
            unscreened.append(r)
        elif r["e_hull"] <= args.screen:
            screened.append(r)
        else:
            above.append(r)

    n_ctrl = min(args.n_random, len(above))
    control = rng.sample(above, n_ctrl) if n_ctrl else []

    def mk(r, group):
        return dict(idx=int(r["idx"]), formula=r.get("formula"), group=group,
                    mlff_e_hull=r.get("e_hull"), mlff_e_form=r.get("e_form"),
                    comp_valid=bool(r.get("comp_valid")), n_el=int(r.get("n_el", 0)),
                    structure=structs[int(r["idx"])])

    entries = ([mk(r, "screened") for r in sorted(screened, key=lambda x: x["e_hull"])]
               + [mk(r, "unscreened") for r in unscreened]
               + [mk(r, "control") for r in control])
    for pos, e in enumerate(entries):
        e["pos"] = pos

    meta = dict(
        source=str(args.results), created=time.strftime("%Y-%m-%d %H:%M:%S"),
        mlff=st.get("mlff"), seed=args.seed, screen_e_hull=args.screen,
        n_relaxed_total=len(recs),
        n_comp_valid_total=int(sum(bool(r.get("comp_valid")) for r in recs)),
        n_unary=len(unary), n_pool_screened=len(screened),
        n_pool_unscreened=len(unscreened), n_pool_above=len(above),
        n_control=len(control))
    _save_json(dict(meta=meta, entries=entries), args.out)

    print(f"Wrote {args.out}: {len(entries)} structures for DFT")
    print(f"  screened   (MLFF E_hull <= {args.screen}): {len(screened)}")
    print(f"  unscreened (no MLFF E_hull available)   : {len(unscreened)}")
    print(f"  control    (random, above the cut)      : {len(control)} of {len(above)}")
    print(f"  excluded unary structures               : {len(unary)}")
    print(f"  denominator for rates (n_relaxed_total) : {len(recs)}")


# =============================================================================
# run  (relax -> relax -> static, with custodian error handling)
# =============================================================================
def _vasp_sets():
    from pymatgen.io.vasp.sets import MPRelaxSet, MPStaticSet
    return MPRelaxSet, MPStaticSet


def _run_vasp_in(stage_dir, vasp_cmd):
    """Run VASP in stage_dir under custodian (auto-fixes common VASP failures)."""
    from custodian.custodian import Custodian
    from custodian.vasp.handlers import (
        FrozenJobErrorHandler, MeshSymmetryErrorHandler, NonConvergingErrorHandler,
        PositiveEnergyErrorHandler, PotimErrorHandler, StdErrHandler,
        UnconvergedErrorHandler, VaspErrorHandler)
    from custodian.vasp.jobs import VaspJob

    handlers = [VaspErrorHandler(), MeshSymmetryErrorHandler(),
                UnconvergedErrorHandler(), NonConvergingErrorHandler(),
                PotimErrorHandler(), PositiveEnergyErrorHandler(),
                FrozenJobErrorHandler(), StdErrHandler()]
    job = VaspJob(shlex.split(vasp_cmd), final=True, suffix="", auto_npar=False,
                  auto_gamma=False, backup=True)
    cwd = os.getcwd()
    os.chdir(stage_dir)
    try:
        Custodian(handlers, [job], max_errors=8).run()
    finally:
        os.chdir(cwd)


def _read_vasprun(stage_dir):
    from pymatgen.io.vasp.outputs import Vasprun
    p = Path(stage_dir) / "vasprun.xml"
    if not p.exists():
        return None
    try:
        return Vasprun(str(p), parse_dos=False, parse_eigen=False,
                       parse_projected_eigen=False, parse_potcar_file=False)
    except Exception as exc:
        log.warning("cannot parse %s (%s)", p, exc)
        return None


def _stage(kind, stage_dir, structure, vasp_cmd, dry_run):
    MPRelaxSet, MPStaticSet = _vasp_sets()
    stage_dir = Path(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)
    vr = _read_vasprun(stage_dir)
    if vr is not None and not dry_run:              # finished earlier: resume
        ok = vr.converged_electronic and (vr.converged_ionic or kind == "static")
        if ok or (kind != "static" and (stage_dir / "CONTCAR").exists()
                  and vr.converged_electronic):
            return vr
    vis = MPRelaxSet(structure) if kind != "static" else MPStaticSet(structure)
    vis.write_input(str(stage_dir), potcar_spec=dry_run)   # dry run: POTCAR.spec only
    if dry_run:
        inc = vis.incar
        print(f"    [{kind}] {stage_dir}: ENCUT={inc.get('ENCUT')} "
              f"ISPIN={inc.get('ISPIN')} LDAU={inc.get('LDAU', False)} "
              f"EDIFFG={inc.get('EDIFFG')} NSW={inc.get('NSW')} "
              f"POTCAR functional={vis.potcar_functional}")
        return None
    _run_vasp_in(stage_dir, vasp_cmd)
    return _read_vasprun(stage_dir)


def run_one(entry, vasp_cmd, workdir, dry_run=False):
    d = _run_dir(workdir, entry)
    d.mkdir(parents=True, exist_ok=True)
    if (d / "DONE").exists() and not dry_run:
        print(f"[{entry['pos']:>4}] idx {entry['idx']} {entry['formula']}: already done")
        return
    (d / "FAILED").unlink(missing_ok=True)
    s0 = Structure.from_dict(entry["structure"])
    print(f"[{entry['pos']:>4}] idx {entry['idx']} {entry['formula']} "
          f"({len(s0)} atoms, group={entry['group']})")
    t0 = time.time()
    res = dict(idx=entry["idx"], pos=entry["pos"], formula=entry["formula"])
    try:
        vr1 = _stage("relax1", d / "relax1", s0, vasp_cmd, dry_run)
        if dry_run:
            print("    (dry run: later stages need relax1 output, not written)")
            return
        if vr1 is None or not (d / "relax1" / "CONTCAR").exists():
            raise RuntimeError("relax1 produced no usable output")
        s1 = Structure.from_file(d / "relax1" / "CONTCAR")

        vr2 = _stage("relax2", d / "relax2", s1, vasp_cmd, dry_run)
        if vr2 is None or not (d / "relax2" / "CONTCAR").exists():
            raise RuntimeError("relax2 produced no usable output")
        s2 = Structure.from_file(d / "relax2" / "CONTCAR")

        vr3 = _stage("static", d / "static", s2, vasp_cmd, dry_run)
        if vr3 is None or not vr3.converged_electronic:
            raise RuntimeError("static calculation did not converge electronically")

        res.update(relax_converged=bool(vr2.converged_ionic),
                   static_converged=bool(vr3.converged_electronic),
                   n_ionic_steps_relax2=len(vr2.ionic_steps),
                   final_energy=float(vr3.final_energy),
                   wall_seconds=time.time() - t0)
        _save_json(res, d / "result.json")
        (d / "DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
        print(f"    done in {(time.time() - t0) / 60:.1f} min, "
              f"relax2 converged={res['relax_converged']}")
    except Exception as exc:
        (d / "FAILED").write_text(f"{type(exc).__name__}: {exc}")
        print(f"    FAILED: {type(exc).__name__}: {exc}")


def cmd_run(args):
    sel = _load_selection(args.selection)
    todo = _pick_entries(sel, args)
    if args.workers > 1 and len(todo) > 1 and not args.dry_run:
        def _sub(e):
            cmd = [sys.executable, os.path.abspath(__file__), "run", "--pos", str(e["pos"]),
                   "--selection", args.selection, "--workdir", args.workdir,
                   "--vasp-cmd", args.vasp_cmd]
            return subprocess.run(cmd).returncode
        print(f"Running {len(todo)} structures, {args.workers} at a time "
              f"(each with VASP_CMD='{args.vasp_cmd}')")
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            list(ex.map(_sub, todo))
    else:
        for e in todo:
            run_one(e, args.vasp_cmd, args.workdir, dry_run=args.dry_run)


# =============================================================================
# slurm / status
# =============================================================================
def cmd_slurm(args):
    sel = _load_selection(args.selection)
    n = len(sel["entries"])
    lines = ["#!/bin/bash",
             "#SBATCH --job-name=dcf-dft",
             f"#SBATCH --array=0-{n - 1}%{args.max_concurrent}",
             "#SBATCH --nodes=1",
             f"#SBATCH --ntasks={args.ntasks}",
             f"#SBATCH --time={args.time}",
             "#SBATCH --output=slurm_logs/%x_%A_%a.out"]
    if args.partition:
        lines.append(f"#SBATCH --partition={args.partition}")
    if args.account:
        lines.append(f"#SBATCH --account={args.account}")
    lines += ["",
              "mkdir -p slurm_logs",
              "# --- EDIT: load your environment (VASP, python with pymatgen/custodian) ---",
              "# module load vasp/6.4.2",
              "# source ~/venvs/dft/bin/activate",
              f'export VASP_CMD="{args.vasp_cmd_slurm.format(ntasks=args.ntasks)}"',
              "# export PMG_VASP_PSP_DIR=/path/to/pymatgen_potcars   # if not in ~/.config/.pmgrc.yaml",
              "",
              f"python {os.path.abspath(__file__)} run --pos $SLURM_ARRAY_TASK_ID "
              f"--selection {args.selection} --workdir {args.workdir} "
              '--vasp-cmd "$VASP_CMD"']
    Path(args.out).write_text("\n".join(lines) + "\n")
    print(f"Wrote {args.out} ({n} array tasks). Submit with:  sbatch {args.out}")
    print("Edit the environment lines first. Runs are resumable: re-submit to "
          "finish failed or timed-out tasks.")


def cmd_status(args):
    sel = _load_selection(args.selection)
    c = Counter()
    for e in sel["entries"]:
        d = _run_dir(args.workdir, e)
        c["done" if (d / "DONE").exists() else "failed" if (d / "FAILED").exists()
          else "started" if d.exists() else "pending"] += 1
    print(f"{len(sel['entries'])} structures: " + ", ".join(f"{k} {v}" for k, v in c.items()))
    for e in sel["entries"]:
        f = _run_dir(args.workdir, e) / "FAILED"
        if f.exists():
            print(f"  FAILED idx {e['idx']} {e['formula']}: {f.read_text()[:120]}")


# =============================================================================
# analyze
# =============================================================================
def reference_entries(elements, cache_dir):
    """Full MP GGA/GGA+U entries (already MP2020-corrected) for a chemical system.
    Cached on disk so repeated analyses do not re-query the API."""
    from monty.serialization import dumpfn, loadfn
    key = "-".join(sorted(elements))
    cache = Path(cache_dir) / f"{key}.json.gz"
    if cache.exists():
        return loadfn(str(cache))
    from mp_api.client import MPRester
    with MPRester(_api_key()) as mpr:
        # NB: the default thermo type is now the mixed GGA/GGA+U/r2SCAN hull; PBE DFT
        # energies must be compared with the GGA/GGA+U hull.
        ents = mpr.get_entries_in_chemsys(
            sorted(elements), additional_criteria={"thermo_types": ["GGA_GGA+U"]})
    cache.parent.mkdir(exist_ok=True)
    dumpfn(ents, str(cache))
    return ents


def e_above_hull(entry, ref_entries):
    """E_hull of a corrected entry against reference entries (negative allowed)."""
    from pymatgen.analysis.phase_diagram import PhaseDiagram
    try:
        return float(PhaseDiagram(ref_entries).get_e_above_hull(entry, allow_negative=True))
    except Exception as exc:
        log.warning("hull failed for %s: %s", entry.composition.reduced_formula, exc)
        return None


def _dft_record(entry, workdir, compat, ppd=None, cache_dir=MP_ENTRY_CACHE):
    d = _run_dir(workdir, entry)
    rec = dict(idx=entry["idx"], pos=entry["pos"], formula=entry["formula"],
               group=entry["group"], mlff_e_hull=entry.get("mlff_e_hull"),
               comp_valid=entry.get("comp_valid"), n_el=entry.get("n_el"),
               status="missing")
    if not (d / "DONE").exists():
        rec["status"] = "failed" if (d / "FAILED").exists() else "missing"
        return rec, None
    res = _load_json(d / "result.json")
    vr = _read_vasprun(d / "static")
    if vr is None:
        rec["status"] = "no_vasprun"
        return rec, None
    ce = vr.get_computed_entry(inc_structure=True, entry_id=f"dcf-{entry['idx']}")
    processed = compat.process_entries([ce], clean=True, on_error="ignore")
    if not processed:
        rec["status"] = "incompatible_with_MP2020"    # e.g. unsupported POTCAR set
        return rec, None
    corrected = processed[0]
    final_struct = ce.structure
    rec.update(status="ok", relax_converged=res["relax_converged"],
               energy_per_atom_corrected=float(corrected.energy_per_atom),
               formula_dft=final_struct.composition.reduced_formula,
               rmsd_mlff_to_dft=rmsd_periodic(Structure.from_dict(entry["structure"]),
                                              final_struct),
               vpa_mlff=Structure.from_dict(entry["structure"]).volume / len(final_struct),
               vpa_dft=final_struct.volume / len(final_struct))
    if ppd is not None:
        try:
            rec["dft_e_hull"] = float(ppd.get_e_above_hull(corrected, allow_negative=True))
        except Exception as exc:
            log.warning("patched PD failed for %s: %s", rec["formula"], exc)
            rec["dft_e_hull"] = None
    else:
        elems = [str(e) for e in final_struct.composition.elements]
        rec["dft_e_hull"] = e_above_hull(corrected, reference_entries(elems, cache_dir))
    return rec, final_struct


def _load_novelty_ref(path):
    if not path:
        return None
    if Path(path).is_dir():
        return [Structure.from_file(str(p)) for p in sorted(Path(path).glob("*.cif"))]
    return [Structure.from_dict(d) for d in _load_json(path)]


def _is_novel(structure, ref_by_elems, matcher):
    key = frozenset(int(sp.Z) for sp in structure.composition.elements)
    for r in ref_by_elems.get(key, []):
        try:
            if matcher.fit(structure, r):
                return False
        except Exception:
            continue
    return True


def summarize(records, structures, meta, novelty_ref=None):
    """records: per-structure dicts (with dft_e_hull); structures: idx -> DFT structure.
    Pure function of its inputs so it can be tested without VASP."""
    from pymatgen.analysis.structure_matcher import StructureMatcher
    ok = [r for r in records if r["status"] == "ok"]
    have = [r for r in ok if r.get("dft_e_hull") is not None]
    n_total = meta["n_relaxed_total"]

    def is_stable(r, thr):
        return r["dft_e_hull"] <= thr and r["n_el"] >= 2

    stable = [r for r in have if is_stable(r, STABLE_THR)]
    metast = [r for r in have if is_stable(r, META_THR)]

    # S.U.N.: unique among the stable set, then novel vs the training set
    sun = None
    if stable:
        matcher = StructureMatcher()
        groups = matcher.group_structures([structures[r["idx"]] for r in stable])
        n_unique = len(groups)
        if novelty_ref:
            by = {}
            for s in novelty_ref:
                by.setdefault(frozenset(int(sp.Z) for sp in s.composition.elements), []).append(s)
            reps = [g[0] for g in groups]
            n_sun = sum(_is_novel(s, by, matcher) for s in reps)
        else:
            n_sun = None
        sun = dict(n_unique=n_unique, n_sun=n_sun)
    else:
        sun = dict(n_unique=0, n_sun=0 if novelty_ref else None)

    scr = [r for r in have if r["group"] in ("screened", "unscreened")]
    ctl = [r for r in have if r["group"] == "control"]
    ctl_stable = [r for r in ctl if is_stable(r, STABLE_THR)]
    ctl_meta = [r for r in ctl if is_stable(r, META_THR)]
    n_above = meta["n_pool_above"]

    def extrapolate(hits_screened, hits_ctl):
        """screened hits + control-group rate scaled to the unrun pool above the cut"""
        if not ctl:
            return None
        return len(hits_screened) + len(hits_ctl) / len(ctl) * n_above

    est_stable = extrapolate([r for r in scr if is_stable(r, STABLE_THR)], ctl_stable)
    est_meta = extrapolate([r for r in scr if is_stable(r, META_THR)], ctl_meta)

    cv_total = meta["n_comp_valid_total"]
    stable_cv = [r for r in stable if r.get("comp_valid")]

    both = [r for r in have if r.get("mlff_e_hull") is not None]
    diff = np.array([r["dft_e_hull"] - r["mlff_e_hull"] for r in both]) if both else np.array([])

    out = dict(
        n_relaxed_total=n_total, n_dft_run=len(ok), n_dft_with_hull=len(have),
        n_failed=sum(r["status"] in ("failed", "missing", "no_vasprun") for r in records),
        frac_relax_converged=(float(np.mean([r["relax_converged"] for r in ok])) if ok else None),
        rmsd_mean=(float(np.nanmean([r["rmsd_mlff_to_dft"] for r in ok
                                     if r.get("rmsd_mlff_to_dft") is not None]))
                   if any(r.get("rmsd_mlff_to_dft") is not None for r in ok) else None),
        n_stable=len(stable), n_metastable=len(metast),
        stable_rate=len(stable) / n_total, metastable_rate=len(metast) / n_total,
        stable_rate_comp_valid=(len(stable_cv) / cv_total if cv_total else None),
        n_unique_stable=sun["n_unique"], n_sun=sun["n_sun"],
        sun_rate=(sun["n_sun"] / n_total if sun["n_sun"] is not None else None),
        n_control_run=len(ctl), n_control_stable=len(ctl_stable),
        n_control_metastable=len(ctl_meta),
        stable_rate_extrapolated=(est_stable / n_total if est_stable is not None else None),
        metastable_rate_extrapolated=(est_meta / n_total if est_meta is not None else None),
        mlff_vs_dft_ehull=(dict(mean_dft_minus_mlff=float(diff.mean()),
                                mae=float(np.abs(diff).mean()), n=int(diff.size))
                           if diff.size else None),
        stable_list=sorted([dict(idx=r["idx"], formula=r["formula"], dft_e_hull=r["dft_e_hull"],
                                 mlff_e_hull=r.get("mlff_e_hull"), comp_valid=r.get("comp_valid"))
                            for r in stable], key=lambda x: x["dft_e_hull"]))
    return out


def print_summary(s, meta):
    f = lambda v, p=1: "n/a" if v is None else f"{100 * v:.{p}f}"
    print("\n" + "=" * 72)
    print("  DFT stability summary (PBE/PBE+U, MP-compatible; vs full MP GGA/GGA+U hull)")
    print("=" * 72)
    print(f"  Structures in the MLFF analysis (denominator): {s['n_relaxed_total']}")
    print(f"  DFT completed: {s['n_dft_run']}   with E_hull: {s['n_dft_with_hull']}   "
          f"failed/missing: {s['n_failed']}")
    print(f"  Relaxations converged: {f(s['frac_relax_converged'])}%   "
          f"mean displacement MLFF->DFT: "
          f"{'n/a' if s['rmsd_mean'] is None else format(s['rmsd_mean'], '.3f')} A")
    if s["mlff_vs_dft_ehull"]:
        m = s["mlff_vs_dft_ehull"]
        print(f"  E_hull  DFT - MLFF (n={m['n']}): mean {m['mean_dft_minus_mlff']:+.3f}, "
              f"MAE {m['mae']:.3f} eV/atom")
    print(f"  Stable (E_hull <= {STABLE_THR}, >=2 el.): {s['n_stable']}  "
          f"({f(s['stable_rate'], 2)}% of {s['n_relaxed_total']})")
    print(f"  Metastable (<= {META_THR}):              {s['n_metastable']}  "
          f"({f(s['metastable_rate'], 2)}%)")
    print(f"  Unique among stable: {s['n_unique_stable']}   "
          f"S.U.N.: {'n/a (no --novelty-ref)' if s['n_sun'] is None else s['n_sun']}  "
          f"({f(s['sun_rate'], 2)}%)")
    print(f"  Stable among compositionally valid: {f(s['stable_rate_comp_valid'], 2)}%")
    print(f"  Screening check: {s['n_control_stable']}/{s['n_control_run']} random control "
          f"structures (MLFF E_hull above {meta['screen_e_hull']}) are DFT-stable, "
          f"{s['n_control_metastable']} metastable")
    if s["n_control_stable"] or s["n_control_metastable"]:
        print("  *** the MLFF screen missed some stable/metastable structures; "
              "use the extrapolated rates below, or lower/raise --screen and rerun ***")
    print(f"  Extrapolated rates (screened hits + control rate x unrun pool): "
          f"stable {f(s['stable_rate_extrapolated'], 2)}%  "
          f"metastable {f(s['metastable_rate_extrapolated'], 2)}%")
    if s["stable_list"]:
        print("  DFT-stable structures:")
        for r in s["stable_list"]:
            print(f"    idx {r['idx']:>5}  {r['formula']:<14} E_hull(DFT) {r['dft_e_hull']:+.3f}  "
                  f"MLFF {r['mlff_e_hull']:+.3f}  comp_valid={r['comp_valid']}")
    print("\n  LaTeX row for Table 5 (n = number of DFT calculations that completed):")
    row = (f"  \\dcf{{}}, DFT subset ($n={s['n_dft_run']}$)$^{{\\ddagger}}$ & PBE(+U) & "
           f"{f(s['frac_relax_converged'])} & "
           f"{'--' if s['rmsd_mean'] is None else format(s['rmsd_mean'], '.2f')} & "
           f"{f(s['stable_rate'])} & {f(s['metastable_rate'])} & "
           f"{'--' if s['sun_rate'] is None else f(s['sun_rate'])} \\\\")
    print(row)
    print("  Footnote: stable rates are DFT-confirmed counts over all "
          f"{s['n_relaxed_total']} sampled structures; the remainder were screened out by "
          f"MatterSim (E_hull > {meta['screen_e_hull']} eV/atom).")
    print("=" * 72)


def cmd_analyze(args):
    from pymatgen.entries.compatibility import MaterialsProject2020Compatibility
    sel = _load_selection(args.selection)
    meta, entries = sel["meta"], sel["entries"]
    compat = MaterialsProject2020Compatibility()

    ppd = None
    if args.ppd:
        opener = gzip.open if args.ppd.endswith(".gz") else open
        with opener(args.ppd, "rb") as fh:
            ppd = pickle.load(fh)
        print(f"Using patched phase diagram from {args.ppd} (no MP API needed).")

    records, structures = [], {}
    for e in entries:
        rec, fs = _dft_record(e, args.workdir, compat, ppd=ppd, cache_dir=args.cache)
        records.append(rec)
        if fs is not None:
            structures[rec["idx"]] = fs

    novelty_ref = _load_novelty_ref(args.novelty_ref)
    s = summarize(records, structures, meta, novelty_ref)
    print_summary(s, meta)

    _save_json(dict(meta=meta, summary=s, records=records), f"{args.out}.json")
    cols = ["idx", "pos", "formula", "group", "status", "mlff_e_hull", "dft_e_hull",
            "relax_converged", "rmsd_mlff_to_dft", "vpa_mlff", "vpa_dft", "comp_valid", "n_el"]
    with open(f"{args.out}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(records)
    print(f"\nWrote {args.out}.json and {args.out}.csv")


# =============================================================================
# CLI
# =============================================================================
def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(q):
        q.add_argument("--selection", default=SELECTION_FILE)
        q.add_argument("--workdir", default=WORKDIR)

    q = sub.add_parser("select", help="choose structures from the evaluation JSON")
    q.add_argument("--results", default=RESULTS_JSON)
    q.add_argument("--structures-file", default=None,
                   help="JSON list of {idx, structure} if not inside the results JSON")
    q.add_argument("--out", default=SELECTION_FILE)
    q.add_argument("--screen", type=float, default=SCREEN_E_HULL,
                   help="MLFF E_hull cut in eV/atom (default %(default)s)")
    q.add_argument("--n-random", type=int, default=N_RANDOM_CONTROL)
    q.add_argument("--seed", type=int, default=SEED)
    q.set_defaults(fn=cmd_select)

    q = sub.add_parser("run", help="run VASP (relax, relax, static)")
    common(q)
    q.add_argument("--pos", type=int, help="position in the selection (SLURM array index)")
    q.add_argument("--idx", type=int, help="structure idx from the evaluation")
    q.add_argument("--all", action="store_true")
    q.add_argument("--start", type=int, default=0)
    q.add_argument("--stop", type=int, default=None)
    q.add_argument("--workers", type=int, default=1,
                   help="structures in parallel (each launches its own VASP_CMD)")
    q.add_argument("--vasp-cmd", default=VASP_CMD)
    q.add_argument("--dry-run", action="store_true",
                   help="only write inputs (POTCAR.spec) and print key settings")
    q.set_defaults(fn=cmd_run)

    q = sub.add_parser("slurm", help="write a SLURM array script")
    common(q)
    q.add_argument("--out", default="submit_dft.sbatch")
    q.add_argument("--ntasks", type=int, default=32)
    q.add_argument("--time", default="24:00:00")
    q.add_argument("--partition", default="")
    q.add_argument("--account", default="")
    q.add_argument("--max-concurrent", type=int, default=20)
    q.add_argument("--vasp-cmd-slurm", default="srun -n {ntasks} vasp_std")
    q.set_defaults(fn=cmd_slurm)

    q = sub.add_parser("status", help="show progress")
    common(q)
    q.set_defaults(fn=cmd_status)

    q = sub.add_parser("analyze", help="E_hull, rates and Table 5 row")
    common(q)
    q.add_argument("--out", default=RESULTS_PREFIX)
    q.add_argument("--cache", default=MP_ENTRY_CACHE)
    q.add_argument("--novelty-ref", default=None,
                   help="JSON list of Structure dicts (or folder of CIFs): MP-20 train set")
    q.add_argument("--ppd", default=None,
                   help="optional pickled PatchedPhaseDiagram (e.g. Matbench Discovery "
                        "2023-02-07 MP hull) to use instead of the live MP API")
    q.set_defaults(fn=cmd_analyze)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
