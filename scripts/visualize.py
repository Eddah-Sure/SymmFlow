
"""Render one generation trajectory

Targets the GNN SITE-TOKEN variant of DirectCrystalFlow: the latent is one token
per crystallographic site of the asymmetric unit, and there are no capsules, no
routing and no per-capsule gate. Panel B is therefore a per-SITE decomposition
rather than a per-capsule one.

This drives model.sample() rather than re-implementing the decode, so the panels
always show what the model actually does. It reports:

  * per-site occupancy against the count head's own cut, so a site count stuck
    at a constant is visible (the count is regressed against the true
    asymmetric-unit size, so a constant is a training failure, not a
    thresholding artefact);
  * per-site TYPE CONFIDENCE and the spread of the latent tokens, which are the
    two numbers that separate the causes of a unary cell: identical tokens mean
    the flow collapsed the latent set, while diverse tokens with a low
    confidence mean the decoder never committed to an element;
  * distinct elements per cell, flagged when the cell is unary, since a unary
    cell is passed UNCONDITIONALLY by the SMACT gate and so inflates
    composition validity without saying anything about the model;
  * structural validity under BOTH the reference 0.5 A gate and the model's
    own stricter training criterion, kept apart because only the first is
    comparable with published numbers;
  * optionally, the shuffled-conditioning control, which is the only thing
    that separates learned space-group conditioning from the guarantee
    supplied by orbit expansion.

CIF export is audited before anything is written; see the CIF EXPORT section.

SPACE-GROUP PANEL MODE  (--panel)
  Generates crystals under REQUESTED space groups (default P-1, P2_1/c, R-3,
  P6_3/mmc, Fm-3m; three each) and draws them in one figure, with the asymmetric
  unit outlined in red, Wyckoff labels, and the formula and space group beneath
  each. A sample is shown only if it passes the reference structural gate and the
  SMACT composition gate, is not unary, and spglib recovers exactly the requested
  group. Counts of what was tried and rejected go to the JSON next to the figure.

      python visualizer.py --panel --checkpoint stage3_finetune.pt
      python visualizer.py --panel_redraw fig_panel_c.json      # redraw, no model

  Before generating, the panel mode installs the stored-basis symmetry correction
  (symfix.py), exactly as the evaluation script does. Without it the orbit
  expansion is the pre-fix one and the cells are about 3x too large. Needs
  symfix.py on the path; --no_symfix turns it off (and says so loudly).
"""

import os
import sys
import json
import argparse
import re
import warnings
from pathlib import Path
from collections import Counter
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from mpl_toolkits import mplot3d as _mplot3d   # registers the '3d' projection
_ = _mplot3d
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Path3DCollection

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  USER SETTINGS — edit these in Colab, then run the cell.                  ║
# ║  (When run as a CLI script, command-line flags override these defaults.)  ║
# ╚══════════════════════════════════════════════════════════════════════════╝
CHECKPOINT         = os.environ.get("EMF_CHECKPOINT",
                                    os.path.join("checkpoints", "stage3_finetune.pt"))

DATASET_PATH       = os.environ.get("MP20_ROOT", os.path.join("data", "MP20"))
TARGET_ENERGY      = -1.0         # target formation energy (eV/atom); None = unconditional
NUM_CRYSTALS       = 3             # independent crystals to generate
NUM_STEPS          = 100           # ODE integration steps
GUIDANCE_SCALE     = None          # None -> use the checkpoint's cfg.guidance.
TEMPERATURE        = 0.0           # decoder sampling temperature. 0.0 = argmax
MAX_UNIQUE_TYPES   = None          # POST-HOC unique-element cap applied by the
                                   # VISUALISER on top of whatever the model did.
                                   # None/0 = off, which is the default now that
                                   # the model ships with cfg.unique_type_cap=0.
                                   # Setting it rewrites elements the decoder
                                   # actually emitted, so the rendered crystal is
                                   # no longer the model's output; any edit is
                                   # listed in the figure caption.
COND_SG            = None          # space group fed to the FLOW; None = same as
                                   # the group the cell is built under
SG_CONTROL         = False         # also run the shuffled-conditioning control
SEED               = 42            # base RNG seed (sample i uses SEED + i)
OUTPUT             = "crystal_trajectory_direct.png"
SAVE_CIF           = True          # write the final crystal to CIF (needs pymatgen)
DPI                = 200
DEVICE             = "auto"        # "auto" | "cuda" | "cpu"
EXPAND_ORBITS      = None          # None = follow cfg.decode_asymmetric_unit.
CIF_SYMMETRIZE     = False         # a symmetry-reduced copy is written ONLY if
                                   # it survives an expansion round-trip
PROJECT_LATTICE    = True          # kept for the CLI; sample() always projects
                                   # the cell onto the space-group family now, so
                                   # this only affects the printed banner
SNAPSHOT_TS        = [0.0, 0.25, 0.50, 0.75, 1.0]
# MP-20 test-split median per-atom volume. Used only to annotate the density of
# a rendered cell as a RATIO, since "59.4 A^3/atom" means nothing to a reader
# who does not already know the reference is ~18.
MP20_REF_VPA       = 18.2
SHOW_INLINE        = True          # display saved figures inline when in a notebook
# ── space-group panel mode ────────────────────────────────────────────────────
PANEL_MODE         = False         # True = run the panel without passing --panel
PANEL_GROUPS       = [2, 14, 148, 194, 225]   # P-1, P2_1/c, R-3, P6_3/mmc, Fm-3m
PANEL_PER_GROUP    = 3             # crystals shown per group
PANEL_CANDIDATES   = 24            # max samples tried per group to find PER_GROUP eligible ones
PANEL_MAX_ATOMS    = 32            # skip cells larger than this (legibility only)
PANEL_OUTPUT       = "fig_panel_c.png"
TARGET_NAME        = "formation_energy_per_atom"   # for building the dataset symfix needs
AUTO_INSTALL_DEPS  = True
if AUTO_INSTALL_DEPS:
    import importlib.util as _ilu, subprocess as _sp
    if _ilu.find_spec("pymatgen") is None:
        print("Installing pymatgen (for CIF export; one-time)...")
        try:
            _sp.check_call([sys.executable, "-m", "pip", "install", "-q", "pymatgen"])
            print("  pymatgen installed.")
        except Exception as _e:
            print(f"  [warn] pymatgen install failed ({_e}); CIF export skipped.")

try:
    from pymatgen.core import Structure as _PmgStructure, Lattice as _PmgLattice
    from pymatgen.io.cif import CifWriter as _CifWriter
    from pymatgen.analysis.structure_matcher import StructureMatcher as _SM
    _HAVE_PMG = True
except ImportError:
    _HAVE_PMG = False
    _SM = None

import datetime as _dt
import inspect as _inspect

try:
    from sklearn.decomposition import PCA
    _HAVE_PCA = True
except ImportError:
    _HAVE_PCA = False

warnings.filterwarnings("ignore")
plt.rcParams.update({
    "font.family":      "sans-serif",
    "font.sans-serif":  ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size":        9,
    "axes.labelsize":   9,
    "axes.titlesize":   10,
    "axes.titleweight": "bold",
    "axes.linewidth":   0.7,
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
    "legend.fontsize":  7.5,
    "legend.framealpha": 0.85,
    "pdf.fonttype":     42,
})

# ─────────────────────────────────────────────────────────────────────────────
# Element colours (Jmol palette subset) and symbols
# ─────────────────────────────────────────────────────────────────────────────
_JMOL = {
    1:"#FFFFFF", 2:"#D9FFFF", 3:"#CC80FF", 4:"#C2FF00", 5:"#FFB5B5",
    6:"#909090", 7:"#3050F8", 8:"#FF0D0D", 9:"#90E050", 10:"#B3E3F5",
    11:"#AB5CF2", 12:"#8AFF00", 13:"#BFA6A6", 14:"#F0C8A0", 15:"#FF8000",
    16:"#FFFF30", 17:"#1FF01F", 18:"#80D1E3", 19:"#8F40D4", 20:"#3DFF00",
    22:"#BFC2C7", 23:"#A6A6AB", 24:"#8A99C7", 25:"#9C7AC7", 26:"#E06633",
    27:"#F090A0", 28:"#50D050", 29:"#C88033", 30:"#7D80B0",
    38:"#00FF00", 40:"#94BFFF", 41:"#73C2C9", 42:"#54B5B5", 47:"#C0C0C0",
    50:"#668080", 56:"#00C900", 57:"#70D4FF", 74:"#2194D6", 78:"#D0D0E0",
    79:"#FFD123", 82:"#575961", 83:"#9E4FB5",
}


def _elem_color(z: int) -> str:
    z = int(z)
    if z in _JMOL:
        return _JMOL[z]
    import colorsys                       # golden-angle hue spacing for the tail
    h = ((z * 137) % 360) / 360.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.45, 0.88)
    return "#%02X%02X%02X" % (int(r * 255), int(g * 255), int(b * 255))


_PT_SYMBOLS = [
    "X",
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
    "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
    "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
]


def _z_sym(z: int) -> str:
    z = int(z)
    return _PT_SYMBOLS[z] if 1 <= z < len(_PT_SYMBOLS) else f"Z{z}"


_CAP_PALETTE = [
    "#E63946", "#457B9D", "#2A9D8F", "#E9C46A", "#264653", "#F4A261",
    "#8338EC", "#3A86FF", "#06D6A0", "#EF476F", "#118AB2", "#FFD166",
]


def _site_color(j: int) -> str:
    return _CAP_PALETTE[j % len(_CAP_PALETTE)]


# ─────────────────────────────────────────────────────────────────────────────
# 3-D structure rendering + CIF export
# ─────────────────────────────────────────────────────────────────────────────
_COVALENT_R = {
    1:0.31, 3:1.28, 4:0.96, 5:0.84, 6:0.76, 7:0.71, 8:0.66, 9:0.57, 11:1.66,
    12:1.41, 13:1.21, 14:1.11, 15:1.07, 16:1.05, 17:1.02, 19:2.03, 20:1.76,
    22:1.60, 23:1.53, 24:1.39, 25:1.39, 26:1.32, 27:1.26, 28:1.24, 29:1.32,
    30:1.22, 32:1.20, 33:1.19, 34:1.20, 35:1.20, 38:1.95, 39:1.90, 40:1.75,
    41:1.64, 42:1.54, 46:1.39, 47:1.45, 48:1.44, 50:1.39, 51:1.39, 52:1.38,
    56:2.15, 57:2.07, 72:1.75, 73:1.70, 74:1.62, 78:1.36, 79:1.36, 82:1.46,
    83:1.48,
}


def _cov_r(z: int) -> float:
    return _COVALENT_R.get(int(z), 1.3)


def _unit_cell_edges_cart(L: np.ndarray):
    """The 12 edges of the unit-cell parallelepiped as cartesian segments."""
    corners = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)],
                       dtype=float) @ L
    edges = []
    for a in range(8):
        for b in range(a + 1, 8):
            fa = np.array([(a >> 2) & 1, (a >> 1) & 1, a & 1])
            fb = np.array([(b >> 2) & 1, (b >> 1) & 1, b & 1])
            if int(np.abs(fa - fb).sum()) == 1:
                edges.append([corners[a], corners[b]])
    return edges


def _draw_crystal_3d(ax, frac, z_list, L, draw_bonds=True, bond_scale=1.25,
                     show_labels=True, title=None, view=(22, -60),
                     highlight=None):
    """Render a crystal in 3-D: cartesian atoms coloured by element, the unit-cell
    wireframe, and (optionally) heuristic bonds within a covalent-radius cutoff.

    frac   : [N,3] fractional coords          z_list : [N] atomic numbers
    L      : [3,3] lattice matrix, row-vector convention (cart = frac @ L)
    highlight : indices drawn with a heavy outline
    """
    frac = np.asarray(frac, dtype=float) % 1.0
    z_list = np.asarray(z_list, dtype=int)
    L = np.asarray(L, dtype=float)
    cart = frac @ L

    ax.add_collection3d(Line3DCollection(_unit_cell_edges_cart(L),
                                         colors="#555555", linewidths=0.6, alpha=0.5))

    if draw_bonds and len(cart) > 1:
        shifts = np.array([[i, j, k] for i in (-1, 0, 1)
                           for j in (-1, 0, 1) for k in (-1, 0, 1)], dtype=float)
        img_cart = shifts @ L
        segs = []
        for i in range(len(cart)):
            for j in range(i + 1, len(cart)):
                cand = (cart[j] + img_cart) - cart[i]
                kb = int(np.argmin(np.einsum("kc,kc->k", cand, cand)))
                d = float(np.linalg.norm(cand[kb]))
                if 0.4 < d <= bond_scale * (_cov_r(z_list[i]) + _cov_r(z_list[j])):
                    segs.append([cart[i], cart[i] + cand[kb]])
        if segs:
            ax.add_collection3d(Line3DCollection(segs, colors="#999999",
                                                 linewidths=1.4, alpha=0.65))

    seen = set()
    hi = set(int(i) for i in (highlight if highlight is not None else []))
    for ai, ((x, y, zc), zz) in enumerate(zip(cart, z_list)):
        sym = _z_sym(int(zz))
        lbl = sym if sym not in seen else None
        seen.add(sym)
        _hl = ai in hi
        ax.scatter([x], [y], [zc], s=90 * _cov_r(int(zz)), c=_elem_color(int(zz)),
                   edgecolors=("#d62728" if _hl else "#222"),
                   linewidths=(1.8 if _hl else 0.5), depthshade=True,
                   zorder=5, label=lbl)
        if show_labels:
            ax.text(x, y, zc, sym, fontsize=6, ha="center", va="center",
                    fontweight="bold", zorder=6)

    base = np.vstack([np.zeros((1, 3)), np.array([[1, 1, 1]]) @ L])
    pts = np.vstack([base, cart]) if len(cart) else base
    ctr = pts.mean(0)
    span = max(pts.max(0) - pts.min(0)) * 0.55 + 1e-6
    ax.set_xlim(ctr[0] - span, ctr[0] + span)
    ax.set_ylim(ctr[1] - span, ctr[1] + span)
    ax.set_zlim(ctr[2] - span, ctr[2] + span)
    ax.set_xlabel("x (Å)", fontsize=7, labelpad=-4)
    ax.set_ylabel("y (Å)", fontsize=7, labelpad=-4)
    ax.set_zlabel("z (Å)", fontsize=7, labelpad=-4)
    ax.tick_params(labelsize=5.5, pad=-2)
    ax.view_init(elev=view[0], azim=view[1])
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass
    if title:
        ax.set_title(title, fontsize=8.5, fontweight="bold")
    if show_labels:
        h, l = ax.get_legend_handles_labels()
        if h:
            ax.legend(h[:6], l[:6], loc="upper left", fontsize=6, framealpha=0.7,
                      edgecolor="none", handlelength=0.7, borderpad=0.3,
                      bbox_to_anchor=(0.0, 1.0))


# ═══════════════════════════════════════════════════════════════════════════
# CIF EXPORT
#
# A generated structure is not an experimental one and not a relaxed one, and a
# CIF that does not say so will eventually be mistaken for both. Everything
# below exists to make the exported file honest about three things: what the
# geometry actually is, what symmetry it actually has, and where it came from.
#
# The previous writer handed the raw decoder output straight to pymatgen. That
# silently accepted four things a CIF cannot legitimately represent:
#   1. a left-handed lattice matrix. CIF stores only (a, b, c, alpha, beta,
#      gamma), from which any reader rebuilds a RIGHT-handed cell, so a
#      left-handed generated cell is re-read as its own mirror image. Distances
#      survive (they depend only on the metric tensor) but chirality does not.
#   2. two atoms on the same position. Expansion can place different elements on
#      one orbit; the result is not an ordered structure and no refinement
#      program will accept it.
#   3. a symmetry-reduced copy written at symprec=0.1 that, when expanded again,
#      is a DIFFERENT structure from the one generated. That file looks tidier
#      and is wrong.
#   4. fractional coordinates of exactly 1.0, which some readers treat as a
#      duplicate of 0.0 and others do not.
# ═══════════════════════════════════════════════════════════════════════════

CIF_MIN_DIST      = 0.5      # CDVAE structural-validity threshold (Angstrom)
CIF_MIN_VOLUME    = 0.1      # non-degenerate cell (Angstrom^3)
CIF_COINCIDENT    = 1.0e-3   # closer than this and it is the SAME site
CIF_SNAP          = 1.0e-6   # fractional-coordinate snapping tolerance
CIF_MIN_ANGLE     = 1.0      # degrees; below this the cell is unusable
CIF_MAX_ANGLE     = 179.0


def cell_params_from_matrix(L):
    """(a, b, c, alpha, beta, gamma) in Angstrom and degrees."""
    L = np.asarray(L, dtype=float).reshape(3, 3)
    lens = np.linalg.norm(L, axis=1)
    ang = np.zeros(3)
    for i, (j, k) in enumerate(((1, 2), (0, 2), (0, 1))):
        denom = lens[j] * lens[k]
        c = 0.0 if denom < 1e-12 else float(np.dot(L[j], L[k]) / denom)
        ang[i] = np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))
    return (float(lens[0]), float(lens[1]), float(lens[2]),
            float(ang[0]), float(ang[1]), float(ang[2]))


def fix_handedness(L, frac):
    """Make the cell right-handed WITHOUT moving any atom.

    CIF cannot express handedness, so a left-handed matrix would be re-read as
    the enantiomorph. Negating the third lattice vector and the third
    fractional component together leaves every Cartesian position x = f L
    exactly where it was, because (f D)(D L) = f L for D = diag(1, 1, -1),
    while flipping the sign of the determinant.
    """
    L = np.asarray(L, dtype=float).reshape(3, 3).copy()
    frac = np.asarray(frac, dtype=float).reshape(-1, 3).copy()
    if np.linalg.det(L) >= 0:
        return L, frac, False
    L[2] = -L[2]
    if frac.size:
        frac[:, 2] = -frac[:, 2]
    return L, frac, True


def wrap_frac(frac):
    """Wrap into [0, 1) and snap values within tolerance of a cell edge to 0."""
    f = np.asarray(frac, dtype=float).reshape(-1, 3) % 1.0
    f[np.abs(f - 1.0) < CIF_SNAP] = 0.0
    f[np.abs(f) < CIF_SNAP] = 0.0
    return f


def min_image_distance_matrix(frac, L):
    """Pairwise minimum-image distances over the 27 nearest cell images."""
    frac = np.asarray(frac, dtype=float).reshape(-1, 3)
    L = np.asarray(L, dtype=float).reshape(3, 3)
    n = frac.shape[0]
    if n == 0:
        return np.zeros((0, 0))
    d = frac[:, None, :] - frac[None, :, :]
    shifts = np.array([[i, j, k] for i in (-1, 0, 1)
                       for j in (-1, 0, 1) for k in (-1, 0, 1)], dtype=float)
    best = np.full((n, n), np.inf)
    for s in shifts:
        cart = (d + s) @ L
        best = np.minimum(best, np.linalg.norm(cart, axis=-1))
    return best


def audit_structure(Z, frac, L):
    """Validate and canonicalise before anything is serialised.

    Returns (ok, payload, report). `payload` carries the cleaned arrays and the
    cell parameters; `report` is a dict of everything a caller should print or
    embed in the file. Nothing here silently repairs chemistry -- coincident
    sites of DIFFERENT elements are a refusal, not a merge, because there is no
    ordered structure to write.
    """
    rep = {"errors": [], "warnings": [], "notes": []}
    Z = np.asarray(Z, dtype=int).reshape(-1)
    frac = np.asarray(frac, dtype=float).reshape(-1, 3)
    L = np.asarray(L, dtype=float).reshape(3, 3)

    if Z.size == 0:
        rep["errors"].append(
            "cell contains no atoms. The site-count head is bounded below at 1, "
            "so the decoder cannot emit an empty asymmetric unit; an empty cell "
            "here means the atoms were lost downstream (orbit expansion or a "
            "post-hoc type edit), not that the count head produced nothing.")
        return False, None, rep
    if Z.size != frac.shape[0]:
        rep["errors"].append(f"{Z.size} species but {frac.shape[0]} positions.")
        return False, None, rep
    if not np.isfinite(L).all() or not np.isfinite(frac).all():
        rep["errors"].append("non-finite lattice or coordinates.")
        return False, None, rep
    bad_z = Z[(Z < 1) | (Z > 118)]
    if bad_z.size:
        rep["errors"].append(f"atomic numbers outside 1-118: {sorted(set(bad_z.tolist()))}")
        return False, None, rep

    vol = abs(float(np.linalg.det(L)))
    if vol < CIF_MIN_VOLUME:
        rep["errors"].append(f"degenerate cell, |det L| = {vol:.3e} < {CIF_MIN_VOLUME}")
        return False, None, rep

    L, frac, flipped = fix_handedness(L, frac)
    if flipped:
        rep["notes"].append(
            "lattice matrix was left-handed; the c axis and the third "
            "fractional component were negated together, which preserves every "
            "Cartesian position exactly and prevents the file being re-read as "
            "the enantiomorph.")

    a, b, c, al, be, ga = cell_params_from_matrix(L)
    for nm, v in (("alpha", al), ("beta", be), ("gamma", ga)):
        if not (CIF_MIN_ANGLE < v < CIF_MAX_ANGLE):
            rep["errors"].append(f"cell angle {nm} = {v:.2f} deg is degenerate.")
            return False, None, rep
    if min(a, b, c) < 1e-2:
        rep["errors"].append(f"cell edge below 0.01 A (a,b,c = {a:.3f},{b:.3f},{c:.3f}).")
        return False, None, rep

    frac = wrap_frac(frac)

    # Coincident sites. Same element -> one site counted twice, which is a
    # writable structure once deduplicated. Different elements -> not an
    # ordered structure at all.
    dm = min_image_distance_matrix(frac, L)
    n = len(Z)
    iu = np.triu_indices(n, k=1)
    close = np.where(dm[iu] < CIF_COINCIDENT)[0]
    drop = set()
    for idx in close:
        i, j = int(iu[0][idx]), int(iu[1][idx])
        if Z[i] != Z[j]:
            rep["errors"].append(
                f"sites {i} ({Z[i]}) and {j} ({Z[j]}) are {dm[i, j]:.2e} A apart "
                f"but hold DIFFERENT elements. This is a cross-element site "
                f"conflict from orbit expansion, not an ordered structure; "
                f"refusing to write a CIF that claims two elements occupy one "
                f"position.")
            return False, None, rep
        drop.add(max(i, j))
    if drop:
        keep = np.array([i for i in range(n) if i not in drop], dtype=int)
        rep["notes"].append(
            f"{len(drop)} duplicate site(s) of the same element within "
            f"{CIF_COINCIDENT} A were merged; symmetry expansion had emitted the "
            f"same orbit position more than once.")
        Z, frac = Z[keep], frac[keep]
        dm = min_image_distance_matrix(frac, L)

    # Physical plausibility. Below the CDVAE gate we still write the file,
    # because it is what the model produced, but the file says so.
    n = len(Z)
    min_d = None
    if n > 1:
        off = dm + np.eye(n) * (CIF_MIN_DIST + 1e3)
        min_d = float(off.min())
        if min_d < CIF_MIN_DIST:
            rep["warnings"].append(
                f"minimum interatomic distance {min_d:.3f} A is below the "
                f"{CIF_MIN_DIST} A structural-validity threshold; this structure "
                f"fails the standard gate and is exported for inspection only.")

    vol = abs(float(np.linalg.det(L)))
    payload = dict(Z=Z, frac=frac, L=L,
                   params=(a, b, c, al, be, ga), volume=vol,
                   min_dist=min_d, n_sites=n, vpa=vol / max(n, 1))
    rep.update(volume=vol, min_dist=min_d, n_sites=n, vpa=vol / max(n, 1),
               params=(a, b, c, al, be, ga), handedness_flipped=flipped)
    return True, payload, rep


def provenance_lines(meta):
    """CIF audit block. Standard tags only, so no parser trips over it."""
    today = _dt.date.today().isoformat()
    body = [
        "_audit_creation_date              " + today,
        "_audit_creation_method",
        ";",
        "  COMPUTER-GENERATED STRUCTURE. This is neither an experimental",
        "  determination nor a DFT-relaxed structure. Atomic positions are the",
        "  raw output of a generative model and have NOT been relaxed with any",
        "  electronic-structure or interatomic-potential method. Any energy",
        "  quoted below is a model prediction, not a calculated formation",
        "  energy. Treat this file as a candidate for relaxation, not as a",
        "  structure determination.",
        "",
        "  Model                : " + str(meta.get("model", "EquiCap / DirectCrystalFlow")),
        "  Checkpoint           : " + str(meta.get("checkpoint", "unknown")),
        "  Sampling             : " + str(meta.get("sampling", "unknown")),
        "  Property target      : " + str(meta.get("target", "unconditional")),
        "  Property (latent)    : " + str(meta.get("y_latent", "n/a")),
        "  Property (re-encoded): " + str(meta.get("y_reenc", "n/a")),
        "  Space group requested: " + str(meta.get("sg_requested", "n/a")),
        "  Space group realised : " + str(meta.get("sg_realized", "n/a")),
        "  Seed                 : " + str(meta.get("seed", "n/a")),
    ]
    for extra in meta.get("notes", []):
        body.append("  Note                 : " + str(extra))
    for w in meta.get("warnings", []):
        body.append("  WARNING              : " + str(w))
    body.append(";")
    # Inside a semicolon-delimited text field a line starting with ';' closes
    # the field, so any caller-supplied text that begins that way would break
    # the file. Indent it instead; only the opening and closing delimiters are
    # allowed at column zero.
    safe = [body[0], body[1], ";"]
    for ln in body[3:-1]:
        safe.append((" " + ln) if ln.startswith(";") else ln)
    safe.append(";")
    return safe


def _inject_provenance(cif_text, meta):
    """Put the audit block immediately after the data_ header."""
    out, done = [], False
    for ln in cif_text.splitlines():
        out.append(ln)
        if not done and ln.strip().startswith("data_"):
            out.append("")
            out.extend(provenance_lines(meta))
            out.append("")
            done = True
    if not done:
        # No data_ header to attach to. Tags outside a data block are not valid
        # CIF, so open one rather than emitting orphaned items.
        out = ["data_generated_structure", ""] + provenance_lines(meta) + [""] + out
    txt = "\n".join(out) + "\n"
    if "_symmetry_space_group_name_H-M" not in txt and "_space_group_name_H-M" not in txt:
        txt = txt.replace("_audit_creation_date",
                          "_symmetry_space_group_name_H-M   'P 1'\n"
                          "_symmetry_Int_Tables_number      1\n"
                          "_audit_creation_date", 1)
    return txt


def _roundtrip_ok(path, ref_struct, primitive=False, stol=0.05):
    """Read the file back and confirm it is the structure we meant to write.

    A CIF is only useful if reading it reproduces the structure. This matters
    most for the symmetry-reduced copy, which stores an asymmetric unit plus a
    set of operations; if the generated structure only APPROXIMATELY has the
    claimed symmetry, expanding the file gives something else.
    """
    if not _HAVE_PMG or _SM is None:
        return None, "pymatgen/StructureMatcher unavailable"
    try:
        back = _PmgStructure.from_file(str(path))
    except Exception as e:
        return False, f"could not be parsed back ({e})"
    try:
        m = _SM(ltol=0.05, stol=stol, angle_tol=2.0,
                primitive_cell=primitive, scale=False, attempt_supercell=False)
        if m.fit(ref_struct, back):
            return True, f"round-trip verified ({len(back)} sites)"
        return False, (f"round-trip MISMATCH: wrote {len(ref_struct)} sites, "
                       f"reading the file back gives {len(back)}")
    except Exception as e:
        return None, f"round-trip check failed ({e})"


def write_cif(types_Z, frac, L, path, meta=None, symmetrize=False, symprec=0.1):
    """Audit, canonicalise and write a P1 CIF, plus an optional verified
    symmetry-reduced copy.

    Returns (ok, report). The report is also embedded in the file, so a CIF that
    left this function carries its own caveats.
    """
    meta = dict(meta or {})
    ok, pay, rep = audit_structure(types_Z, frac, L)
    for nt in rep["notes"]:
        print(f"  [cif] note: {nt}")
    for w in rep["warnings"]:
        print(f"  [cif] WARNING: {w}")
    if not ok:
        for e in rep["errors"]:
            print(f"  [cif] REFUSED: {e}")
        return False, rep
    if not _HAVE_PMG:
        print("  [cif] pymatgen unavailable; the structure passed the audit but "
              "cannot be serialised.")
        return False, rep

    # Rebuild the cell from (a, b, c, alpha, beta, gamma) in the standard
    # right-handed setting, which is exactly what a reader reconstructs from the
    # file, then confirm the internal geometry survived the change of frame.
    a, b, c, al, be, ga = pay["params"]
    try:
        latt = _PmgLattice.from_parameters(a, b, c, al, be, ga)
        L_std = np.asarray(latt.matrix, dtype=float)
    except Exception as e:
        rep["errors"].append(f"could not rebuild the cell from its parameters ({e})")
        print(f"  [cif] REFUSED: {rep['errors'][-1]}")
        return False, rep
    d0 = min_image_distance_matrix(pay["frac"], pay["L"])
    d1 = min_image_distance_matrix(pay["frac"], L_std)
    drift = float(np.abs(d0 - d1).max()) if d0.size else 0.0
    if drift > 1e-4:
        rep["errors"].append(
            f"rebuilding the cell from its parameters moved interatomic "
            f"distances by up to {drift:.2e} A. The written file would not be "
            f"the generated structure.")
        print(f"  [cif] REFUSED: {rep['errors'][-1]}")
        return False, rep

    struct = _PmgStructure(latt, [int(z) for z in pay["Z"]],
                           pay["frac"].tolist(), coords_are_cartesian=False)
    meta.setdefault("notes", []).extend(rep["notes"])
    meta.setdefault("warnings", []).extend(rep["warnings"])
    meta["cell"] = (f"a={a:.4f} b={b:.4f} c={c:.4f} "
                    f"alpha={al:.3f} beta={be:.3f} gamma={ga:.3f}")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        text = str(_CifWriter(struct, symprec=None))     # explicit P1
    except Exception as e:
        rep["errors"].append(f"CifWriter failed ({e})")
        print(f"  [cif] REFUSED: {rep['errors'][-1]}")
        return False, rep
    path.write_text(_inject_provenance(text, meta), encoding="utf-8")

    rt, msg = _roundtrip_ok(path, struct, primitive=False)
    rep["roundtrip_P1"] = msg
    print(f"  [cif] wrote {pay['n_sites']} site(s) in P1 "
          f"(V = {pay['volume']:.2f} A^3, {pay['vpa']:.2f} A^3/atom"
          + (f", min d = {pay['min_dist']:.3f} A" if pay["min_dist"] is not None else "")
          + f") -> {path}")
    if rt is False:
        print(f"  [cif] WARNING: {msg}")
    elif rt is True:
        print(f"  [cif] {msg}")

    # The symmetry-reduced copy is written ONLY if expanding it reproduces the
    # structure. At symprec=0.1 an approximately symmetric generated cell is
    # readily assigned a group it does not have, and the resulting file is a
    # tidier-looking DIFFERENT crystal.
    if symmetrize:
        sym_path = path.with_name(path.stem + f"_sym{symprec:g}" + path.suffix)
        try:
            sym_text = str(_CifWriter(struct, symprec=float(symprec)))
            sym_path.write_text(_inject_provenance(sym_text, meta), encoding="utf-8")
            ok_rt, msg_rt = _roundtrip_ok(sym_path, struct, primitive=True, stol=0.1)
            if ok_rt is False:
                sym_path.unlink(missing_ok=True)
                print(f"  [cif] symmetry-reduced copy DISCARDED at symprec="
                      f"{symprec:g}: {msg_rt}. The generated cell is only "
                      f"approximately symmetric, so the reduced file would "
                      f"expand to a different structure.")
                rep["roundtrip_sym"] = "discarded: " + msg_rt
            else:
                print(f"  [cif] symmetry-reduced copy -> {sym_path}  ({msg_rt})")
                rep["roundtrip_sym"] = msg_rt
        except Exception as e:
            print(f"  [cif] symmetry-reduced copy failed: {e}")
            rep["roundtrip_sym"] = f"failed: {e}"
    return True, rep


def write_crystal_cif(types_Z, frac, L, path, symmetrize=False, meta=None):
    """Backwards-compatible shim for the old call signature."""
    ok, _ = write_cif(types_Z, frac, L, path, meta=meta, symmetrize=symmetrize)
    return ok


# ═══════════════════════════════════════════════════════════════════════════
# MODEL CLASSES — reuse the live session if available, else import the module
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("RESOLVING MODEL CLASSES")
print("=" * 80)

import importlib
import dataclasses as _dc

# Declared up front so static analysers see them
DirectCrystalFlow = DirectFlowConfig = None
matrix_to_lattice_params = lattice_params_to_matrix = None
project_to_crystal_family = smact_balance_types = None
expand_generated = structural_validity = pairwise_pbc = None
enforce_unique_types = build_edges_from_geometry = None
apply_formability_mask = wyckoff_sites = composition_validity = None
min_image_disp = orbit_multiplicity = None

# Required symbols for the direct flow visualizer
_REQUIRED = ("DirectCrystalFlow", "DirectFlowConfig", "matrix_to_lattice_params",
             "lattice_params_to_matrix", "project_to_crystal_family",
             "smact_balance_types", "expand_generated", "structural_validity",
             "pairwise_pbc")
# build_edges_from_frac never existed in any version of the model file; the real
# symbol is build_edges_from_geometry (added with the structural property head).
_OPTIONAL = ("apply_formability_mask", "build_edges_from_geometry",
             "wyckoff_sites", "enforce_unique_types", "composition_validity",
             "min_image_disp", "orbit_multiplicity")


def _valid_source(ns):
    """True if `ns` has every required symbol AND DirectFlowConfig is a dataclass."""
    if any(ns.get(_n) is None for _n in _REQUIRED):
        return False
    return _dc.is_dataclass(ns["DirectFlowConfig"])


_mod = None
_src_ns = None
_import_errors = {}

if _valid_source(globals()):
    _src_ns = globals()
    _LOAD_SOURCE = "live-session"
    print("  Found valid model classes already defined in this session — reusing them.")
elif globals().get("DirectFlowConfig") is not None and not _dc.is_dataclass(globals()["DirectFlowConfig"]):
    print(f"  [warn] a global 'DirectFlowConfig' exists but is not the dataclass (got "
          f"{type(globals()['DirectFlowConfig']).__name__}) -- ignoring it and trying imports.")

if _src_ns is None:
    _HERE = (os.path.dirname(os.path.abspath(__file__))
             if "__file__" in globals() else os.getcwd())
    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)
    _candidates = list(dict.fromkeys(c for c in [
        os.environ.get("EMF_MODEL_MODULE"),
        "sitetokens",
    ] if c))
    for _modname in _candidates:
        try:
            _cand = importlib.import_module(_modname)
        except Exception as _e:
            _import_errors[_modname] = repr(_e)
            continue
        if not _valid_source(vars(_cand)):
            _missing = [_n for _n in _REQUIRED if not hasattr(_cand, _n)]
            _import_errors[_modname] = (
                f"missing {_missing}" if _missing else
                f"DirectFlowConfig is not a dataclass (got {type(_cand.DirectFlowConfig).__name__})")
            continue
        _mod = _cand
        _src_ns = vars(_cand)
        _LOAD_SOURCE = _mod.__name__
        break

if _src_ns is None:
    raise ImportError(
        "Could not find valid model classes. Not present (as a dataclass "
        f"DirectFlowConfig) in this session's globals, and none of {_candidates} "
        f"imported successfully. Import errors: {_import_errors}.\n"
        "In Colab: run the cell that defines sitetokens.py's contents first "
        "(so DirectCrystalFlow/DirectFlowConfig live in globals), or place this file "
        "next to sitetokens.py, or set EMF_MODEL_MODULE to its module name.")

for _n in _REQUIRED:
    globals()[_n] = _src_ns[_n]
for _opt in _OPTIONAL:
    globals()[_opt] = _src_ns.get(_opt, None)
print(f"  Model classes resolved from '{_LOAD_SOURCE}'"
      + (f" ({getattr(_mod, '__file__', '?')})" if _mod is not None else "") + ".")
assert _dc.is_dataclass(DirectFlowConfig), "sanity check failed: DirectFlowConfig is not a dataclass"


# ═══════════════════════════════════════════════════════════════════════════
# Config / checkpoint helpers
# ═══════════════════════════════════════════════════════════════════════════

def _resolve_device(name):
    if name in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _config_from_dict(cfg_dict, dataset_path):
    """Rebuild a DirectFlowConfig from a saved cfg.__dict__."""
    import dataclasses
    cfg_dict = dict(cfg_dict or {})
    if not cfg_dict.get("atomic_numbers"):
        zs = set()
        for split in ("train", "val", "test"):
            p = Path(dataset_path) / split / f"{split}_config.json"
            if p.exists():
                with open(p) as f:
                    zs.update(int(z) for z in json.load(f).get("atomic_numbers", []))
        cfg_dict["atomic_numbers"] = sorted(zs) or list(range(1, 90))
    if not cfg_dict.get("num_types"):
        cfg_dict["num_types"] = len(cfg_dict["atomic_numbers"])
    if not dataclasses.is_dataclass(DirectFlowConfig):
        raise TypeError(
            f"DirectFlowConfig resolved to {DirectFlowConfig!r} (a {type(DirectFlowConfig).__name__}), not "
            "the @dataclass DirectFlowConfig from sitetokens.py.")
    fields = {f.name for f in dataclasses.fields(DirectFlowConfig)}
    # Filter to valid, non-None values (mirrors sitetokens.config_from_dict)
    kw = {}
    for k, v in cfg_dict.items():
        if k in fields and v is not None:
            kw[k] = v
    dropped = sorted(set(cfg_dict) - fields)
    if dropped:
        print(f"  [cfg] ignoring {len(dropped)} unknown key(s) from the "
              f"checkpoint config: {dropped[:6]}")
    # Merge weights with defaults if present (mirrors sitetokens.config_from_dict)
    weights = kw.pop('weights', None)
    cfg = DirectFlowConfig(**kw)
    if isinstance(weights, dict):
        merged = dict(DirectFlowConfig().weights)
        merged.update(weights)
        cfg.weights = merged
    return cfg


def _load_model(checkpoint_path, dataset_path, device):
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = _config_from_dict(ckpt.get("config", {}), dataset_path)
    model = DirectCrystalFlow(cfg).to(device)
    state = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
    res = model.load_state_dict(state, strict=False)
    miss = list(getattr(res, "missing_keys", []))
    extra = list(getattr(res, "unexpected_keys", []))
    # Deterministic buffers rebuilt from module tables / cfg -- missing ones are
    # harmless. `elem_feat` is rebuilt from cfg.elem_feat_table, `sg_prior` and
    # `_stage3_step`/`initialized` are bookkeeping. The sg-conditioning
    # parameters (conditioner.sg_emb / sg_null / mix) are deliberately NOT here:
    # they are learned, and a checkpoint without them predates sg conditioning.
    _BENIGN = ("wyckoff_proj_lut", "wyckoff_off_lut", "wyckoff_n_free",
               "wyckoff_valid", "wyckoff_general", "sym_rot_lut", "sym_trans_lut",
               "sym_valid_lut", "radii_lut", "ox_lut", "ox_min_lut", "ox_max_lut",
               "ox_states_padded", "ox_states_mask", "formability_mask",
               "elem_feat", "sg_prior", "_stage3_step", "initialized",
               "_encoder_pretrained", "_wyck_step",
               # chemistry tables added with the SMACT-faithful loss; they are
               # non-persistent buffers rebuilt from the module tables, so they
               # are absent from every checkpoint by design
               "eneg_lut", "metal_lut", "_wyckoff_from_data")
    benign = [k for k in miss
              if k.endswith(_BENIGN) or k.startswith("property_frozen.")]
    fatal = [k for k in miss if k not in set(benign)]
    if any(k.startswith("property_frozen.") for k in benign) and hasattr(
            model, "refresh_property_surrogate"):
        model.refresh_property_surrogate()
    if benign:
        print(f"  [load] {len(benign)} deterministic buffer(s) rebuilt from the "
              f"module tables.")
    if fatal:
        # A picture of a partly random model is worse than no picture: it looks
        # exactly like a picture of a trained one.
        print(f"  [load] !! {len(fatal)} LEARNED param(s) left at random init: "
              f"e.g. {fatal[:4]}")
        if os.environ.get("EMF_ALLOW_PARTIAL_LOAD", "0") != "1":
            raise RuntimeError(
                f"{len(fatal)} learned parameter(s) missing from "
                f"{checkpoint_path}. Set EMF_ALLOW_PARTIAL_LOAD=1 to render "
                f"anyway (the figure would not describe a trained model).")
    if extra:
        print(f"  [load] {len(extra)} checkpoint key(s) unused: e.g. {extra[:3]}")
    model.eval()
    _nops = int(model.sym_valid_lut.sum()) if hasattr(model, "sym_valid_lut") else None
    if _nops is not None and _nops <= 230:
        print("  [NO SYMMETRY] the space-group operation table holds only the "
              "identity (built without pymatgen). Orbit expansion and every SG "
              "number in the figure are meaningless. Install pymatgen and delete "
              "~/.cache/emergent_motif_flow/.")
    print(f"  num_types={cfg.num_types}  n_sites={cfg.n_sites}  "
          f"use_wyckoff={getattr(cfg, 'use_wyckoff', True)}  "
          f"refiner={'yes' if getattr(model, 'refiner', None) is not None else 'no'}  "
          f"params={sum(p.numel() for p in model.parameters()):,}")
    print("  latent: one token per site of the asymmetric unit "
          "(GNN encoder; no capsules, routing or gate)")
    _sgc = hasattr(getattr(model, "conditioner", None), "sg_emb")
    _sg_msg = ("YES — space group is an input, and --target_sg is real conditioning"
               if _sgc else
               "NO (legacy model) — --target_sg is applied post-hoc to an "
               "already-committed latent")
    print(f"  sg conditioning : {_sg_msg}")
    if _sgc and not hasattr(model, "sg_prior"):
        print("  [warn] sg conditioning present but no sg_prior buffer; "
              "unconditional runs will fall back to the coarse head's argmax.")
    _efd = int(getattr(cfg, "elem_feat_dim", 0) or 0)
    if _efd > 0:
        _ok = (hasattr(model, "elem_feat")
               and int(model.elem_feat.shape[-1]) == _efd)
        print(f"  elem_feat_dim={_efd}  table present: {_ok}")
        if not _ok:
            print("  [warn] the encoder expects element features but the buffer "
                  "is missing/mis-shaped; the re-encoded energy would be wrong.")
    return model, cfg


# ═══════════════════════════════════════════════════════════════════════════
# Generation with trajectory capture
# ═══════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def generate_with_trajectory(model, target_energy, device, num_steps=100,
                             guidance=None, snapshot_ts=None, seed=42,
                             temperature=0.0, project_lattice=True,
                             max_unique_types=MAX_UNIQUE_TYPES,
                             expand_orbits=None, target_sg=None, cond_sg=None):
    """Generate one crystal and capture the ODE trajectory.

    This drives model.sample(z0=..., trajectory_ts=...) instead of
    re-implementing the decode, so anything the model changes shows up here
    automatically. In particular the decoder no longer consumes the lattice:
    the cell is built AFTER decoding, from the same occupancy that produced the
    sites, so the volume and the atom count cannot drift apart. The visualiser
    inherits that rather than carrying its own copy of the volume logic.
    """
    if snapshot_ts is None:
        snapshot_ts = [0.0, 0.25, 0.50, 0.75, 1.0]
    torch.manual_seed(seed)
    np.random.seed(seed)
    model.eval()
    cfg = model.cfg
    if expand_orbits is None:
        expand_orbits = bool(getattr(cfg, "decode_asymmetric_unit", True)
                             and getattr(cfg, "use_wyckoff", True))
    N, D = cfg.n_sites, cfg.sec_dim
    Z = list(cfg.atomic_numbers)
    guidance = float(cfg.guidance) if guidance is None else float(guidance)
    if target_energy is None and guidance != 1.0:
        print("  [note] classifier-free guidance is inert without a target; "
              "sampling unconditionally.")

    if not hasattr(model, "_decode_latent"):
        raise RuntimeError(
            "This model file predates sample(z0=..., trajectory_ts=...). "
            "Point EMF_MODEL_MODULE at the current model file -- the visualiser "
            "no longer carries its own copy of the decode path, precisely so it "
            "cannot silently disagree with the model.")

    z0 = torch.randn(1, N, D, device=device)
    _skw = {}
    if cond_sg is not None:
        _params = set(_inspect.signature(model.sample).parameters)
        if "cond_sg" not in _params:
            raise RuntimeError(
                "This model file predates sample(cond_sg=...), so the "
                "shuffled-conditioning control cannot be run. The control is "
                "the only thing that separates learned space-group "
                "conditioning from the guarantee supplied by orbit expansion, "
                "so without it the exact-match rate should not be reported as "
                "controllability.")
        _skw["cond_sg"] = int(cond_sg)
    out = model.sample(
        1, target_y=target_energy, steps=num_steps, guidance=guidance,
        device=str(device), temperature=temperature,
        expand_orbits=expand_orbits, target_sg=target_sg,
        z0=z0, trajectory_ts=snapshot_ts, **_skw)

    traj_raw = out["trajectory"]
    snapshots = [dict(t=sd["t"], z=sd["z"].cpu(), occ=sd["occ"][0].cpu(),
                      n_sites=float(sd["n_sites"].reshape(-1)[0]),
                      frac=sd["frac"][0].cpu(), types=sd["types"][0].cpu(),
                      mask=sd["mask"][0].cpu(), L=sd["lattice"][0].cpu(),
                      v_norm=sd.get("v_norm", float("nan")))
                 for sd in traj_raw]

    last = traj_raw[-1]                       # the pre-expansion decode
    z = last["z"]
    L_dec = last["lattice"]
    # SPACE-GROUP SEMANTICS. In the sg-conditioned model `sg_pred` is the group
    # the sample was CONDITIONED on -- it is an input, drawn from the empirical
    # training prior when the caller names none, not a classifier read-out. It
    # is what shapes the Wyckoff site menu, the cell family and the expansion.
    # `sg_coarse` is the read-out head's own guess from the latent, kept only as
    # a self-consistency diagnostic. In a legacy checkpoint both are the same
    # tensor and the two rows below simply agree.
    sg_requested = int(out["sg_pred"].reshape(-1)[0])
    sg_coarse = (int(out["sg_coarse"].reshape(-1)[0]) if "sg_coarse" in out
                 else sg_requested)
    sg_refined = int(out["sg_fine"].reshape(-1)[0]) if "sg_fine" in out else sg_requested
    sg_conditioned = hasattr(getattr(model, "conditioner", None), "sg_emb")
    sg_geom = sg_requested                 # the group that drove cell + expansion
    y_pred = float(out["property"].reshape(-1)[0])

    # Per-site additive energy contribution. The property head is a Deep-Sets
    # read-out, so `e` is exactly the per-token term that is pooled into the
    # prediction; computing it here keeps the panel tied to the head rather
    # than to a reimplementation.
    _, contrib_sites = model.property(z, last["occ"])
    contrib_sites = contrib_sites.reshape(z.shape[0], z.shape[1])[0].cpu()
    mult_pred = model.mult_head(z)[0].cpu()

    # An extra unique-type cap on top of the one sample() already applied is only
    # meaningful if it is stricter.
    _model_cap = int(getattr(cfg, "unique_type_cap", 0) or 0)
    types_final = out["sampled_types"]
    probs_final = out["type_probs"]
    viz_edits = []
    if max_unique_types and (_model_cap <= 0 or max_unique_types < _model_cap):
        _before = types_final.clone()
        types_final, probs_final = _enforce_unique_types_single(
            types_final[0], probs_final[0], out["mask"][0], max_unique_types)
        types_final = types_final.unsqueeze(0)
        probs_final = probs_final.unsqueeze(0)
        _changed = int((_before != types_final).sum())
        if _changed:
            viz_edits.append(
                f"unique-type cap {max_unique_types} (stricter than the model's "
                f"{_model_cap or 'none'}) rewrote {_changed} site(s)")

    L_exp = out.get("lattice_conv", out["lattice"])
    final_exp = dict(frac=out["frac"][0].cpu(), types=types_final[0].cpu(),
                     mask=out["mask"][0].cpu(), L=L_exp[0].cpu())
    exp_stats = ({k: int(out.get(k, 0)) for k in
                  ("n_expansion_failed", "n_site_conflicts", "n_orbit_truncated",
                   "n_types_reconciled")}
                 if expand_orbits else {})

    # Realized space group of the rendered cell
    sg_realized = None
    try:
        from pymatgen.core.structure import Structure as _S
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer as _SGA
        mk = final_exp["mask"].numpy() > 0.5
        fb = final_exp["frac"].numpy()[mk]
        tb = final_exp["types"].numpy()[mk]
        if len(fb):
            zb = [int(Z[min(int(t), len(Z) - 1)]) for t in tb]
            st = _S(final_exp["L"].numpy(), zb, fb, coords_are_cartesian=False)
            sg_realized = int(_SGA(st, symprec=0.1).get_space_group_number())
    except Exception:
        pass

    # ── occupancy diagnostics ─────────────────────────────────────────────
    # Occupancy is a single monotone cut: the count head predicts n in
    # [1, n_sites] and token i is occupied iff i < n. There is no threshold to
    # mismatch and no floor to hide behind, so the only thing worth reporting is
    # the cut itself and whether it saturates the capacity.
    _occ = last["occ"][0].reshape(-1).float().cpu().numpy()
    _n_soft = float(last["n_sites"].reshape(-1)[0])
    _n_on = int((_occ > 0.5).sum())
    occ_info = dict(values=_occ, n_soft=_n_soft, n_on=_n_on, capacity=N,
                    saturated=bool(_n_on >= N))

    # ── non-emptiness ─────────────────────────────────────────────────────
    # Structural, not a guard: n >= 1 by construction in SiteDecoder.
    _n_dec = int((last["mask"][0].cpu().numpy() > 0.5).sum())

    valid, min_dist, comp_ok = None, None, None
    try:
        v = structural_validity(out["frac"], L_exp, out["mask"], types_final,
                                model.radii_lut,
                                scale=float(getattr(cfg, "validity_overlap_scale", 0.5)),
                                min_abs=float(getattr(cfg, "validity_min_dist", 0.75)))
        valid = bool(v.reshape(-1)[0])
    except Exception as e:
        print(f"  [warn] structural_validity failed ({e})")
    comp_ok_modeltable = None
    if composition_validity is not None:
        try:
            # `types_final` holds VOCABULARY indices. composition_validity now
            # maps them to atomic numbers itself, but only if it is given the
            # vocabulary -- without it, index 3 is scored as lithium and the
            # panel reports the validity of a composition the model never
            # emitted. (Checked: Na2O written in vocabulary indices comes back
            # invalid without `atomic_numbers` and valid with it.)
            _an = list(cfg.atomic_numbers) if cfg.atomic_numbers else None
            comp_ok = bool(composition_validity(
                types_final, out["mask"], model.ox_states_padded,
                model.ox_states_mask, atomic_numbers=_an,
                use_pauling_test=True, include_alloys=True,
                use_model_table=False).reshape(-1)[0])
            # The model-table gate (no Pauling test, no alloy exemption) as a
            # DIAGNOSTIC. On 600 random compositions the two gates read 72.3%
            # and 30.7%, so they must never be confused for one another.
            try:
                comp_ok_modeltable = bool(composition_validity(
                    types_final, out["mask"], model.ox_states_padded,
                    model.ox_states_mask, atomic_numbers=_an,
                    use_model_table=True).reshape(-1)[0])
            except TypeError:
                comp_ok_modeltable = None
        except TypeError:
            # a model file that predates the SMACT-faithful signature
            print("  [note] this model file has the OLD composition_validity "
                  "(no Pauling test, no alloy exemption, narrow oxidation "
                  "table). The composition verdict below is NOT a SMACT number.")
            try:
                comp_ok = bool(composition_validity(
                    types_final, out["mask"], model.ox_states_padded,
                    model.ox_states_mask).reshape(-1)[0])
            except Exception as e:
                print(f"  [warn] composition_validity failed ({e})")
        except Exception as e:
            print(f"  [warn] composition_validity failed ({e})")
    try:
        # min_image_disp is chunked; pairwise_pbc materialises (1, A, A, 27, 3)
        # in one go, which is hundreds of MB once the cell is expanded.
        if min_image_disp is not None:
            _, dmat = min_image_disp(out["frac"], out["frac"], L_exp,
                                     n_images=1, chunk=9)
        else:
            _, dmat = pairwise_pbc(out["frac"], L_exp)
        m = out["mask"][0].bool()
        pm = m.unsqueeze(0) & m.unsqueeze(1)
        pm &= ~torch.eye(pm.size(0), dtype=torch.bool, device=pm.device)
        if pm.any():
            min_dist = float(dmat[0][pm].min())
    except Exception:
        pass

    # ── reference structural validity ─────────────────────────────────────
    # The model's structural_validity above is its TRAINING criterion: a
    # covalent-radius exclusion at a 0.75 A floor plus per-atom-volume and
    # connectivity checks. That is stricter than the published gate on every
    # axis and is not comparable with literature numbers. The reference gate is
    # a flat 0.5 A minimum distance plus a non-degenerate cell, and nothing
    # else. Both are reported, labelled.
    cdvae_struct = cdvae_min_d = None
    try:
        _mk = final_exp["mask"].numpy() > 0.5
        _fr = final_exp["frac"].numpy()[_mk]
        _Lc = final_exp["L"].numpy()
        if len(_fr):
            _dm = min_image_distance_matrix(_fr % 1.0, _Lc)
            if len(_fr) > 1:
                _off = _dm + np.eye(len(_fr)) * 1e6
                cdvae_min_d = float(_off.min())
            else:
                cdvae_min_d = float("inf")
            cdvae_struct = bool(cdvae_min_d >= CIF_MIN_DIST
                                and abs(np.linalg.det(_Lc)) >= CIF_MIN_VOLUME)
    except Exception as e:
        print(f"  [warn] reference validity gate failed ({e})")

    # Read the property back off the rendered structure. The old code hard-coded
    # predicted_reenc=None with the comment "no encoder in direct flow model" --
    # there has always been an encoder, and it is the only readout here that
    # depends on the atoms rather than on the latent.
    y_reenc = None
    if hasattr(model, "property_from_structure") and bool(model._encoder_pretrained):
        try:
            # Read the surrogate off the PRE-EXPANSION asymmetric unit. The
            # encoder holds cfg.n_sites tokens, so handing it the expanded cell
            # truncates to the first n_sites in canonical order -- which sorts
            # by atomic number, and would bias the read-out toward light
            # elements. The asymmetric unit is also what the surrogate was
            # trained on during stage 3.
            _pe_src = out.get("pre_expansion")
            if _pe_src is not None:
                _fr_r, _L_r, _mk_r = (_pe_src["frac"], _pe_src["lattice"],
                                      _pe_src["mask"])
                _tp = _pe_src["type_probs"]
            else:
                _fr_r, _L_r, _mk_r = out["frac"], L_exp, out["mask"]
                _tp = probs_final
            if _tp.shape[:2] != _fr_r.shape[:2]:
                _tp = torch.nn.functional.one_hot(
                    _pe_src["sampled_types"].clamp(min=0).long()
                    if _pe_src is not None
                    else types_final.clamp(min=0).long(), cfg.num_types).float()
            # Pass the space group. The surrogate EXPANDS the asymmetric unit
            # by its orbit before re-encoding, because the encoder was trained
            # on full cells with the asymmetric unit selected by site_mask;
            # handing it the bare asymmetric unit strips most of every atom's
            # neighbourhood and the read-out is noise. Without `sg` the
            # expansion is skipped and that old path is what runs.
            _sg_r = None
            if _pe_src is not None and "sg" in _pe_src:
                _sg_r = _pe_src["sg"]
            elif "sg_pred" in out:
                _sg_r = out["sg_pred"]
            _pfs_kw = {}
            if _sg_r is not None and "sg" in set(
                    _inspect.signature(model.property_from_structure).parameters):
                _pfs_kw["sg"] = _sg_r
            _y, _ = model.property_from_structure(
                _fr_r.float(), _tp.to(_fr_r.device).float(),
                _L_r.float(), _mk_r.float(), **_pfs_kw)
            y_reenc = float(model.ydenorm(_y).reshape(-1)[0])
        except Exception as e:
            print(f"  [warn] property_from_structure failed ({e})")

    n_atoms = int(final_exp["mask"].numpy().sum())
    vol = float(abs(np.linalg.det(final_exp["L"].numpy())))
    _npk = "n_atoms_cell_est" if "n_atoms_cell_est" in out else "n_atoms_full_pred"
    n_pred_full = float(out[_npk].reshape(-1)[0]) if _npk in out else float("nan")

    # Panel B is the PRE-expansion decode (one row per site token);
    # out['sampled_types'] is post-expansion and a different length, so it
    # cannot be used here.
    #
    # This used to RE-RUN enforce_unique_types + smact_balance_types to
    # reconstruct that state, which is a copy of sample()'s post-processing and
    # drifted from it the moment either changed -- and both did: SMACT now takes
    # Wyckoff-multiplicity weights, because the decoder emits the asymmetric
    # unit whose formula is not the cell formula. Re-running it with unit
    # weights would balance a different formula than the model balanced and
    # panel B would disagree with panel C. sample() now hands the state over.
    _pre = out.get("pre_expansion")
    if _pre is not None:
        _t_post, _p_post = _pre["sampled_types"], _pre["type_probs"]
        _t_raw = _pre["sampled_types_raw"]
        _pre_mask = _pre["mask"]
        _mult_pre = _pre.get("multiplicity")
    else:
        # legacy model without the snapshot: reproduce as closely as possible
        # and say so, rather than pretending the panels are in sync.
        print("  [note] this model does not expose out['pre_expansion']; "
              "panel B post-processing is being re-derived and may differ "
              "slightly from what sample() actually applied.")
        _t_raw, _p_raw = last["types"], last["type_probs"]
        _t_post, _p_post = _t_raw.clone(), _p_raw.clone()
        _pre_mask = last["mask"]
        _mult_pre = None
        if max_unique_types and max_unique_types > 0 and enforce_unique_types is not None:
            _t_post, _p_post = enforce_unique_types(_t_post, _p_post, _pre_mask,
                                                    max_unique_types)
        # Default False: post-hoc charge repair is off in the current model, so
        # a checkpoint whose cfg predates the flag should NOT have it re-applied
        # here -- that would show a repaired composition as if the decoder had
        # produced it.
        if bool(getattr(cfg, "smact_balance_on_sample", False)):
            _t_post, _p_post = smact_balance_types(
                _t_post, _p_post, _pre_mask, model.ox_states_padded,
                model.ox_states_mask, atomic_numbers=list(cfg.atomic_numbers))

    _pre_probs = (_pre["type_probs"] if _pre is not None else None)
    final = dict(z=z.cpu(), occ=last["occ"][0].cpu(), contrib=contrib_sites,
                 frac=last["frac"][0].cpu(), types=_t_post[0].cpu(),
                 types_raw=_t_raw[0].cpu(),
                 mask=_pre_mask[0].cpu(), L=L_dec[0].cpu(),
                 mult_pred=mult_pred,
                 type_probs=(None if _pre_probs is None else _pre_probs[0].cpu()),
                 mult=(None if _mult_pre is None else _mult_pre[0].cpu()))

    _m_dec = final["mask"].numpy() > 0.5
    _t_dec = final["types"].numpy().astype(int)[_m_dec]
    n_atoms_dec = int(_m_dec.sum())
    formula_dec = " ".join(
        f"{_z_sym(e)}{c}" for e, c in sorted(Counter(
            int(Z[min(int(t), len(Z) - 1)]) for t in _t_dec).items()))
    _raw_t = final["types_raw"].numpy().astype(int)[_m_dec]
    formula_raw = " ".join(
        f"{_z_sym(e)}{c}" for e, c in sorted(Counter(
            int(Z[min(int(t), len(Z) - 1)]) for t in _raw_t).items()))

    # Wyckoff occupancy of the rendered cell: how many decoded sites sit on a
    # special position (multiplicity < |G|). This is the thing the Wyckoff
    # machinery exists to produce, and the field was hard-coded to None before.
    n_special = n_sites = None
    if _mult_pre is not None:
        try:
            _mk = (_pre_mask[0].cpu().numpy() > 0.5)
            _mv = _mult_pre[0].cpu().numpy()[_mk]
            _order = float(model.sym_valid_lut[sg_requested - 1].sum().item())
            n_sites = int(_mk.sum())
            n_special = int((_mv < _order - 0.5).sum())
        except Exception:
            n_special = n_sites = None

    # A unary cell is passed unconditionally by SMACT (see composition_validity),
    # not because of any oxidation-state argument, so
    # composition validity is inflated by exactly the failure it should catch.
    _mk_f = final_exp["mask"].numpy() > 0.5
    _zf = [int(Z[min(int(t), len(Z) - 1)])
           for t in final_exp["types"].numpy()[_mk_f]]
    n_distinct = len(set(_zf))

    # Per-site type confidence: mean max-softmax over decoded sites. Read it
    # together with token_div below -- the pair separates the two
    # composition-collapse mechanisms. LOW confidence with a unary cell means
    # the decoder never committed and a marginal loss was being satisfied by a
    # smear that argmax then rounded away; HIGH confidence with a unary cell and
    # a near-zero token_div means the flow collapsed the latent set and one
    # element is all it carries.
    type_conf = None
    try:
        _pp = out.get("type_probs")
        if _pp is not None:
            _pm = out["mask"][0].bool().cpu()
            _c = _pp[0].max(-1).values.cpu()[_pm]
            if _c.numel():
                type_conf = float(_c.mean())
    except Exception:
        pass

    # Token diversity: mean pairwise cosine DISTANCE between OCCUPIED latent
    # tokens. This is the direct successor of the capsule-collapse check, and it
    # is the number that decides which of the two unary-cell mechanisms is at
    # work. Near 0 with a confident unary cell means the flow collapsed the
    # latent set; a healthy value with low type confidence means the decoder
    # never committed.
    token_div = None
    try:
        _zn = torch.nn.functional.normalize(z, dim=-1)[0].cpu()
        _cos = _zn @ _zn.T
        _mk = torch.as_tensor(_occ > 0.5)
        _pair = (_mk.unsqueeze(1) & _mk.unsqueeze(0))
        _pair &= ~torch.eye(_pair.size(0), dtype=torch.bool)
        if bool(_pair.any()):
            token_div = float((1.0 - _cos)[_pair].mean())
    except Exception:
        pass

    # Reference density for MP-20, so the panel can state the ratio rather than
    # a bare number the reader has to look up.
    _vpa = vol / max(n_atoms, 1)
    vpa_ratio = _vpa / MP20_REF_VPA

    return dict(snapshots=snapshots, final=final, final_exp=final_exp,
                target=target_energy, predicted=y_pred,
                formula_raw=formula_raw,
                predicted_reenc=y_reenc,
                sg_requested=sg_requested, sg_coarse=sg_coarse,
                sg_conditioned=sg_conditioned,
                sg_first=sg_requested,          # back-compat alias
                sg_refined=sg_refined,
                sg_geom=sg_geom, sg_realized=sg_realized, volume=vol,
                vpa=vol / max(n_atoms, 1), valid=valid, comp_valid=comp_ok,
                comp_valid_modeltable=comp_ok_modeltable,
                min_dist=min_dist,
                vpa_floor=float(getattr(cfg, "vpa_floor", 5.0)),
                vpa_ceiling=float(getattr(cfg, "vpa_ceiling", float("inf"))),
                wyckoff=None, n_special=n_special, n_sites=n_sites,
                viz_edits=viz_edits,
                occ_info=occ_info, token_div=token_div,
                n_atoms_pre_expansion=_n_dec, n_distinct=n_distinct,
                cdvae_struct_valid=cdvae_struct, cdvae_min_dist=cdvae_min_d,
                type_conf=type_conf, vpa_ratio=vpa_ratio,
                cond_sg=(None if cond_sg is None else int(cond_sg)),
                guidance=guidance, atomic_numbers=Z, n_tokens=N,
                expanded=bool(expand_orbits), exp_stats=exp_stats,
                n_atoms_decoded=n_atoms_dec, formula_decoded=formula_dec,
                n_atoms_cell=n_atoms, n_atoms_pred_full=n_pred_full)


# ─────────────────────────────────────────────────────────────────────────────
# Panel data helpers
# ─────────────────────────────────────────────────────────────────────────────
def _project_latents(snapshots):
    """Per-site-token latents across snapshots -> [n_snap, N, 2] via PCA."""
    Zs = np.stack([s["z"].numpy()[0] for s in snapshots], 0)      # [T,N,D]
    T, N, D = Zs.shape
    if not np.all(np.isfinite(Zs)):
        n_bad = int((~np.isfinite(Zs).all(-1)).sum())
        warnings.warn(
            f"[visualizer] {n_bad} latent (snapshot, token) entries were "
            f"non-finite — the flow integration diverged for part of the "
            f"trajectory. Interpolating for plotting.", stacklevel=2)
        for j in range(N):
            traj = Zs[:, j, :]
            bad = ~np.isfinite(traj).any(-1)
            if bad.any() and not bad.all():
                good = ~bad
                for d in range(D):
                    traj[bad, d] = np.interp(np.where(bad)[0].astype(float),
                                             np.where(good)[0].astype(float),
                                             traj[good, d].astype(float))
            elif bad.all():
                traj[:] = 0.0
        Zs = np.nan_to_num(Zs, nan=0.0, posinf=0.0, neginf=0.0)
    flat = Zs.reshape(T * N, D)
    if _HAVE_PCA and D > 2 and flat.shape[0] >= 2:
        proj = PCA(n_components=2).fit_transform(flat)
    elif D >= 2:
        proj = flat[:, :2]
    else:
        proj = np.concatenate([flat, np.zeros((flat.shape[0], 2 - D))], 1)
    return proj.reshape(T, N, 2)


def _per_site_breakdown(final, n_sites, Z):
    """Per-site element and the atoms each site contributes to the full cell.

    One token is one site, so there is no [N, K] block to unpack any more. What
    replaces it is the multiplicity: a site on a special Wyckoff position stands
    for fewer full-cell atoms than a general one, and that is the quantity the
    volume is built from.
    """
    types = final["types"].numpy().astype(int).reshape(-1)[:n_sites]
    mask = final["mask"].numpy().reshape(-1)[:n_sites] > 0.5
    site_Z = np.array([int(Z[min(int(t), len(Z) - 1)]) for t in types])
    mult = final.get("mult")
    if mult is None:
        mult = final.get("mult_pred")
    mult = (np.ones(n_sites) if mult is None
            else np.asarray(mult).reshape(-1)[:n_sites].astype(float))
    atoms_per_site = np.where(mask, np.maximum(mult, 1.0), 0.0)
    comp = Counter()
    for zz, mk, mu in zip(site_Z, mask, atoms_per_site):
        if mk:
            comp[int(zz)] += int(round(mu))
    return site_Z, mask, atoms_per_site, comp


def _final_atoms_Z(final, Z):
    """Flat crystal as (frac[M,3], Z[M]) over the masked atoms."""
    m = final["mask"].numpy() > 0.5
    frac = final["frac"].numpy()[m] % 1.0
    types = final["types"].numpy().astype(int)[m]
    return frac, np.array([int(Z[min(t, len(Z) - 1)]) for t in types])


def _enforce_unique_types_single(types, probs, mask, max_unique):
    """Cap ONE crystal to `max_unique` distinct element types."""
    if not max_unique or max_unique <= 0:
        return types, probs
    if enforce_unique_types is not None:
        t, p = enforce_unique_types(types.unsqueeze(0), probs.unsqueeze(0),
                                    mask.unsqueeze(0), max_unique)
        return t[0], p[0]
    types, probs = types.clone(), probs.clone()
    mb = mask.bool()
    t_b = types[mb]
    if t_b.numel() == 0:
        return types, probs
    uniq, counts = t_b.unique(return_counts=True)
    if uniq.numel() <= max_unique:
        return types, probs
    keep_list = uniq[counts.argsort(descending=True)[:max_unique]].tolist()
    keep_set = set(keep_list)
    for j in range(types.shape[0]):
        if not bool(mb[j]) or int(types[j]) in keep_set:
            continue
        new_t = keep_list[int(probs[j, keep_list].argmax())]
        types[j] = new_t
        probs[j].zero_()
        probs[j, new_t] = 1.0
    return types, probs


# ─────────────────────────────────────────────────────────────────────────────
# Figure assembly
# ─────────────────────────────────────────────────────────────────────────────
def build_figure(traj, dpi=200):
    snaps = traj["snapshots"]
    n_snaps = len(snaps)
    N, Z = traj["n_tokens"], traj["atomic_numbers"]
    final = traj["final"]

    # No atom->motif assignment panel: the latent set has no assignment step.
    n_rows = 3
    heights = [1.15, 1.0, 0.92]
    fig = plt.figure(figsize=(4.6 * max(n_snaps, 3), 4.1 * n_rows + 2.5))
    outer = gridspec.GridSpec(n_rows, 1, height_ratios=heights, hspace=0.34,
                              figure=fig)

    # ===================== Panel A — flow trajectory =====================
    gsA = gridspec.GridSpecFromSubplotSpec(2, n_snaps, subplot_spec=outer[0],
                                           height_ratios=[1.4, 1.0], hspace=0.32,
                                           wspace=0.28)
    ax_traj = fig.add_subplot(gsA[0, :max(n_snaps - 1, 1)])
    ax_vel = fig.add_subplot(gsA[0, max(n_snaps - 1, 1):])

    proj = _project_latents(snaps)
    tvals = [s["t"] for s in snaps]
    cmap = plt.cm.viridis
    for j in range(N):
        ax_traj.plot(proj[:, j, 0], proj[:, j, 1], "-", color="#cccccc",
                     lw=0.8, alpha=0.7, zorder=1)
    for ti, t in enumerate(tvals):
        ax_traj.scatter(proj[ti, :, 0], proj[ti, :, 1], s=46, color=cmap(t),
                        edgecolors="#333", linewidths=0.4, zorder=3)
    _td = traj.get("token_div")
    _tds = "" if _td is None else f"   token diversity {_td:.2f}"
    ax_traj.set_title("A · Site-token latent flow (PCA of z(t))" + _tds)
    ax_traj.set_xlabel("PC 1")
    ax_traj.set_ylabel("PC 2")

    _legend_handles = [
        Line2D([0], [0], marker="o", linestyle="none", markersize=6,
               markerfacecolor=cmap(t), markeredgecolor="#333",
               markeredgewidth=0.4, label=f"t={t:.2f}")
        for t in tvals
    ]
    ax_traj.legend(handles=_legend_handles, loc="best", fontsize=6.5,
                   ncol=2, framealpha=0.7)

    ax_vel.plot(tvals, [s["v_norm"] for s in snaps], "o-", color="#E63946", lw=1.6)
    ax_vel.set_title(f"mean ‖v(t)‖   (guidance {traj['guidance']:.2f})", fontsize=9)
    ax_vel.set_xlabel("integration time t")
    ax_vel.set_ylabel("velocity")
    ax_vel.grid(alpha=0.25)

    for si, s in enumerate(snaps):
        ax = fig.add_subplot(gsA[1, si], projection="3d")
        m = s["mask"].numpy() > 0.5
        fr = s["frac"].numpy()[m]
        zz = np.array([int(Z[min(int(t), len(Z) - 1)])
                       for t in s["types"].numpy()[m]])
        if len(fr):
            _draw_crystal_3d(ax, fr, zz, s["L"].numpy(), draw_bonds=False,
                             show_labels=False, view=(20, -60))
        ax.set_title(f"t = {s['t']:.2f}  ({int(m.sum())} atoms)", fontsize=7.5)

    # ===================== Panel B — per-site decomposition ==================
    gsB = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=outer[1],
                                           hspace=0.5, wspace=0.32)
    occ = final["occ"].numpy().reshape(-1)[:N]
    contrib = final["contrib"].numpy().reshape(-1)[:N]
    site_Z, site_mask, atoms_per_site, comp = _per_site_breakdown(final, N, Z)
    znorm = np.linalg.norm(final["z"].numpy()[0], axis=-1)[:N]
    site_cols = [_site_color(j) for j in range(N)]
    xs = np.arange(N)
    _oi = traj.get("occ_info") or {}
    _n_soft = float(_oi.get("n_soft", occ.sum()))
    _n_on = int(_oi.get("n_on", int((occ > 0.5).sum())))

    # B1 — occupancy. One monotone cut, so the only thing to show is where the
    # count head put it. Token i is occupied iff i < n; there is no per-token
    # threshold that could sit in different units from its training target.
    axb1 = fig.add_subplot(gsB[0, 0])
    axb1.bar(xs, occ, color=site_cols, edgecolor="#222", linewidth=0.3)
    axb1.axvline(_n_soft - 0.5, color="#E63946", ls="--", lw=1.2)
    axb1.text(min(_n_soft - 0.4, N - 0.4), 1.02, f"n = {_n_soft:.2f}",
              fontsize=6, color="#E63946", ha="left", va="bottom")
    _b1t = "B1 · site occupancy (monotone cut)"
    if _oi.get("saturated"):
        _b1t += f"  ⚠ SATURATES n_sites ({N})"
    axb1.set_title(_b1t, fontsize=9)
    axb1.set_xlabel(f"site token   ({_n_on} of {N} occupied)")
    axb1.set_ylim(0, 1.08)

    axb2 = fig.add_subplot(gsB[0, 1])
    axb2.bar(xs, contrib, color=site_cols, edgecolor="#222", linewidth=0.3)
    axb2.set_title("B2 · additive energy contribution")
    axb2.set_xlabel("site token")

    axb3 = fig.add_subplot(gsB[0, 2])
    if comp:
        els = [e for e, _ in comp.most_common()]
        axb3.bar([_z_sym(e) for e in els], [comp[e] for e in els],
                 color=[_elem_color(e) for e in els], edgecolor="#222",
                 linewidth=0.4)
    axb3.set_title("B3 · cell composition (multiplicity-weighted)")
    axb3.set_ylabel("atoms in the full cell")

    # B4 — per-site element and how confidently it was chosen. This is the
    # panel that distinguishes the two unary-cell mechanisms at a glance: bars
    # near 1/num_types mean the decoder never committed, bars near 1.0 on a
    # single element mean the latent set carries only one thing to say.
    axb4 = fig.add_subplot(gsB[1, 0])
    _tp = final.get("type_probs")
    if _tp is not None:
        conf = _tp.numpy().max(-1).reshape(-1)[:N]
    else:
        conf = np.full(N, np.nan)
    axb4.bar(xs, np.where(site_mask, conf, 0.0),
             color=[_elem_color(int(z)) if m else "#dddddd"
                    for z, m in zip(site_Z, site_mask)],
             edgecolor="#222", linewidth=0.3)
    _nt = len(Z)
    axb4.axhline(1.0 / max(_nt, 2), color="#888", ls=":", lw=0.9)
    axb4.text(N - 0.4, 1.0 / max(_nt, 2) + 0.02, "uniform", fontsize=5.5,
              color="#888", ha="right")
    axb4.set_title("B4 · per-site element + confidence", fontsize=9)
    axb4.set_xlabel("site token  (bar colour = element)")
    axb4.set_ylim(0, 1.05)
    for j in range(N):
        if site_mask[j]:
            axb4.text(j, min(conf[j] + 0.02, 1.0), _z_sym(int(site_Z[j])),
                      fontsize=5.2, ha="center", va="bottom", rotation=90)

    # B5 — how many full-cell atoms each site stands for. A site on a special
    # Wyckoff position contributes fewer atoms than a general one, and this is
    # the weighting the cell volume is built from.
    axb5 = fig.add_subplot(gsB[1, 1])
    axb5.bar(xs, atoms_per_site, color=site_cols, edgecolor="#222", linewidth=0.3)
    _mult_src = "measured orbit" if final.get("mult") is not None else "head"
    axb5.set_title(f"B5 · atoms per site (multiplicity, {_mult_src})", fontsize=9)
    axb5.set_xlabel("site token")

    axb6 = fig.add_subplot(gsB[1, 2])
    axb6.bar(xs, znorm, color=site_cols, edgecolor="#222", linewidth=0.3)
    _spread = float(np.std(final["z"].numpy()[0], axis=0).mean())
    axb6.set_title(f"B6 · latent ‖z‖ per token  (spread {_spread:.2f})",
                   fontsize=9)
    axb6.set_xlabel("site token")

    # ===================== Panel C — final crystal =====================
    gsC = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[2], wspace=0.12)
    fexp = traj["final_exp"]
    frac, zZ = _final_atoms_Z(fexp, Z)
    Lf = fexp["L"].numpy()
    tag = "C"
    ax_c1 = fig.add_subplot(gsC[0], projection="3d")
    ax_c2 = fig.add_subplot(gsC[1], projection="3d")
    _expanded = bool(traj.get("expanded", True))
    _what = "orbit-expanded" if _expanded else "decoder output"

    if len(frac):
        _draw_crystal_3d(ax_c1, frac, zZ, Lf, highlight=None,
                         title=f"{tag}1 · final cell ({_what})", view=(22, -60))
        _draw_crystal_3d(ax_c2, frac, zZ, Lf, highlight=None,
                         title=f"{tag}2 · rotated", view=(14, 30))

    ax_c3 = fig.add_subplot(gsC[2])
    ax_c3.axis("off")
    formula = " ".join(f"{_z_sym(e)}{n}" for e, n in sorted(Counter(zZ).items()))
    # Occupancy and "carries atoms" are the SAME thing now: one token is one
    # site, so B1 and B5 have identical support by construction. A disagreement
    # would be a bug, not a modelling subtlety.
    n_active = int((occ > 0.5).sum())
    n_with_atoms = int((np.asarray(atoms_per_site) > 0).sum())
    tgt = traj["target"]
    tgt_s = "unconditional" if tgt is None else f"{tgt:+.3f} eV/atom"
    err_s = "—" if tgt is None else f"{abs(traj['predicted'] - tgt):.3f}"

    _req = traj.get("sg_requested", traj.get("sg_first"))
    _conditioned = bool(traj.get("sg_conditioned", False))
    sg_req_s = (f"{_req}  (conditioning input → Wyckoff sites + cell + expansion)"
                if _conditioned else
                f"{_req}  (coarse head → Wyckoff sites + cell + expansion)")
    _coarse = traj.get("sg_coarse", _req)
    sg_coarse_s = (f"{_coarse} " + ("✓" if _coarse == _req else "✗")
                   + "  (head's own guess — self-consistency only)")
    sg_ref_s = f"{traj['sg_refined']}  (refined head → reported only)"
    # Realized is compared against the group we ASKED for. That comparison is
    # the controllability check; comparing it to a read-out head instead only
    # asks whether one classifier agrees with another.
    if traj.get("sg_realized") is not None:
        mk = "✓" if traj["sg_realized"] == _req else "✗"
        sg_real_s = (f"{traj['sg_realized']} {mk}  "
                     + ("(vs requested)" if _conditioned else "(vs driving SG)"))
    else:
        sg_real_s = "—"

    reenc = traj.get("predicted_reenc")
    if reenc is None:
        reenc_s, reenc_err = "unavailable (encoder not pretrained)", "—"
    else:
        reenc_s = f"{reenc:+.3f} eV/atom"
        reenc_err = "—" if tgt is None else f"{abs(reenc - tgt):.3f}"
    if traj.get("valid") is None:
        valid_s = "—"
    else:
        valid_s = "✓" if traj["valid"] else "✗"
        if traj.get("min_dist") is not None:
            valid_s += f"   (min d {traj['min_dist']:.2f} Å)"
    _cv = traj.get("comp_valid")
    # The gate is SMACT's, i.e. CDVAE's: an integer oxidation assignment summing
    # to zero AND the Pauling electronegativity test, with unary compositions
    # and all-metal alloys passing unconditionally. A ✗ can therefore mean
    # either criterion, so the label does not claim to know which.
    comp_s = ("—" if _cv is None else
              ("✓  (SMACT: charge balance + Pauling test)" if _cv else
               "✗  (fails SMACT: no charge-balanced assignment, or the "
               "electronegativity test)"))
    _cvm = traj.get("comp_valid_modeltable")
    if _cvm is not None and _cv is not None and _cvm != _cv:
        comp_s += ("   [model-table gate says "
                   + ("✓" if _cvm else "✗")
                   + " — diagnostic only, not a SMACT number]")
    # The cell volume is clamped to [vpa_floor, vpa_ceiling] per atom. Sitting
    # exactly on either bound means the lattice head asked for something outside
    # the band and the clamp caught it -- the rendered density is the clamp's,
    # not the model's. The ceiling exists because the lattice head is trained on
    # encoder latents and runs out of distribution on rolled-out ones.
    _vf = traj.get("vpa_floor", 5.0)
    _vc = traj.get("vpa_ceiling", float("inf"))
    vpa_s = f"{traj['vpa']:.1f} Å³/atom"
    if abs(traj["vpa"] - _vf) < 0.05:
        vpa_s += f"   ⚠ PINNED AT vpa_floor ({_vf:.1f})"
    elif np.isfinite(_vc) and abs(traj["vpa"] - _vc) < 0.05:
        vpa_s += f"   ⚠ PINNED AT vpa_ceiling ({_vc:.1f})"

    _oi = traj.get("occ_info") or {}
    occ_s = (f"{_oi.get('n_on', n_active)}/{N} occupied  "
             f"(count head n = {_oi.get('n_soft', float(n_active)):.2f})")
    if _oi.get("saturated"):
        occ_s += "  ⚠ SATURATES capacity — raise cfg.n_sites"

    _nd = traj.get("n_distinct")
    if _nd is None:
        nd_s = "—"
    elif _nd <= 1:
        nd_s = (f"{_nd}  ⚠ UNARY — SMACT passes single-element compositions "
                f"UNCONDITIONALLY, so this cell scores comp-valid without "
                f"saying anything about the model, and its formation energy "
                f"is zero BY DEFINITION. MP-20 is overwhelmingly "
                f"multi-element; a unary cell is a composition-collapse "
                f"signal, not a success.")
    else:
        nd_s = str(_nd)

    _cs, _cd = traj.get("cdvae_struct_valid"), traj.get("cdvae_min_dist")
    if _cs is None:
        cdv_s = "—"
    else:
        cdv_s = "✓" if _cs else "✗"
        if _cd is not None and np.isfinite(_cd):
            cdv_s += f"   (min d {_cd:.2f} Å ≥ {CIF_MIN_DIST} Å)"

    lines = [
        ("Target energy", tgt_s),
        ("Predicted energy (latent)", f"{traj['predicted']:+.3f} eV/atom"),
        ("  ↳ abs. error", err_s),
        ("Re-encoded energy", reenc_s),
        ("  ↳ abs. error (honest)", reenc_err),
        ("SG · requested" if _conditioned else "SG · driving", sg_req_s),
        ("SG · coarse head", sg_coarse_s),
        ("SG · refined head", sg_ref_s),
        ("SG · realized", sg_real_s),
        ("Validity · reference gate (0.5 Å)", cdv_s),
        ("Validity · model criterion (0.75 Å + VPA + conn.)", valid_s),
        ("Composition validity (SMACT)", comp_s),
        ("Distinct elements in cell", nd_s),
        ("Sites in the asymmetric unit", occ_s),
        ("Cell volume", f"{traj['volume']:.1f} Å³"),
        ("Volume / atom", vpa_s),
        ("Atoms · cell (panels C1/C2)", f"{len(zZ)}"),
        ("Sites: occupied / carrying atoms", f"{n_active} / {n_with_atoms}  "
                                             f"(of {N}; these must agree)"),
        ("Formula · cell (C1/C2)", formula or "—"),
    ]
    _es = traj.get("exp_stats") or {}
    if traj.get("expanded"):
        _bits = [f"on (SG {_req})"]
        if _es.get("n_expansion_failed"):
            _bits.append("SHRANK → un-expanded")
        if _es.get("n_site_conflicts"):
            _bits.append(f"{_es['n_site_conflicts']} site conflict(s)")
        if _es.get("n_orbit_truncated"):
            _bits.append(f"{_es['n_orbit_truncated']} orbit(s) truncated")
        lines.append(("Orbit expansion", "; ".join(_bits)))
        if _es.get("n_types_reconciled"):
            lines.append(("Types reconciled",
                          f"{_es['n_types_reconciled']} site(s) — "
                          f"symmetry-equivalent positions held different "
                          f"elements; ONE element won. This is how an element "
                          f"in B3 can be absent from C1/C2."))
    else:
        lines.append(("Orbit expansion", "off — C shows the decoder output"))
    if traj.get("n_special") is not None and traj.get("n_sites"):
        _ns, _nt = traj["n_special"], traj["n_sites"]
        lines.append(("Special Wyckoff sites",
                      f"{_ns} / {_nt}  ({100.0 * _ns / max(_nt, 1):.0f}% off the "
                      f"general position)"))
    if traj.get("n_atoms_pre_expansion") == 0:
        lines.append(("Decoder output",
                      "EMPTY — this should be unreachable: the site count is "
                      "bounded below at 1. Check for atom loss downstream, not "
                      "in the count head."))
    if traj.get("cond_sg") is not None:
        lines.append(("SG · fed to the flow",
                      f"{traj['cond_sg']}  ⚠ CONTROL RUN: the flow was "
                      f"conditioned on a DIFFERENT group from the one the cell "
                      f"was built under"))
    if traj.get("vpa_ratio") is not None:
        _r = traj["vpa_ratio"]
        _flag = "  ⚠" if (_r > 1.5 or _r < 0.67) else ""
        lines.append(("Density vs MP-20",
                      f"{_r:.2f}x the reference median "
                      f"({MP20_REF_VPA:.1f} A^3/atom){_flag}"))
    if traj.get("type_conf") is not None:
        lines.append(("Per-site type confidence",
                      f"{traj['type_conf']:.3f} mean max-softmax"
                      + ("  (low: the decoder has not committed to an element)"
                         if traj["type_conf"] < 0.6 else "")))
    if traj.get("token_div") is not None:
        _tdv = traj["token_div"]
        _msg = f"{_tdv:.3f} mean pairwise cosine distance"
        if _tdv < 0.05:
            _msg += ("  ⚠ the occupied tokens are IDENTICAL — the flow "
                     "collapsed the latent set and no decoder can recover "
                     "more than one element from it")
        lines.append(("Latent token diversity", _msg))
    if traj.get("viz_edits"):
        lines.append(("Visualizer-only edits", "; ".join(traj["viz_edits"])))
    ax_c3.set_title(f"{tag}3 · final crystal summary", fontsize=10,
                    fontweight="bold", loc="left")
    y = 0.96
    dy = min(0.098, 0.93 / max(len(lines), 1))
    # Long value strings are right-aligned against left-aligned labels, so the
    # two collide once the card is full. Scale the type with the row count.
    _fs = 7.8 if len(lines) <= 16 else max(5.4, 7.8 * 16.0 / len(lines))
    for k, v in lines:
        ax_c3.text(0.02, y, k, fontsize=_fs, color="#555",
                   transform=ax_c3.transAxes)
        ax_c3.text(0.98, y, v, fontsize=_fs,
                   fontweight=("normal" if k.startswith("  ") else "bold"),
                   ha="right", transform=ax_c3.transAxes)
        y -= dy
    _foot = ("panel A = flow trajectory; panel B = decoder output, one bar per "
             "SITE of the asymmetric unit; panels C1/C2 = final cell.\n"
             "volume and V/atom describe the cell shown in C1/C2.\n")
    if _conditioned:
        _foot += ("space group is a CONDITIONING INPUT. 'realized' vs "
                  "'requested' is bounded below by orbit expansion, which "
                  "applies the requested group's operations regardless of what "
                  "the flow produced; run --sg_control for the honest number.")
    else:
        _foot += ("legacy checkpoint: space group is read off the latent, not "
                  "conditioned on.")
    ax_c3.text(0.02, max(y, 0.01), _foot,
               fontsize=6.2, color="#888", style="italic",
               transform=ax_c3.transAxes, va="top")
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# SPACE-GROUP PANEL MODE
# ═══════════════════════════════════════════════════════════════════════════
def _model_module():
    """The Python module the model classes came from (symfix patches it)."""
    if _mod is not None:
        return _mod
    return sys.modules.get("__main__")


def _install_symfix(model, cfg, dataset_path, strict=False, enabled=True):
    """Install the stored-basis symmetry correction, as the evaluation script does.

    The orbit expansion shipped with the model builds cells in a conventional
    basis that differs from the basis the Wyckoff codebook was mined in. The
    visualiser used to skip this step, so every crystal it drew came from the
    pre-fix pipeline (about 3x too many atoms per cell, special positions
    under-used, Pnma broken). This rebuilds the training dataset only to read the
    stored bases; it does not touch the weights.
    """
    if not enabled:
        print("  [symfix] DISABLED (--no_symfix): crystals follow the PRE-FIX orbit "
              "expansion and will not match the evaluation results.")
        return None

    def _fail(msg):
        if strict:
            raise RuntimeError(msg + "  Pass --no_symfix to render anyway (the "
                                     "structures will then be the pre-fix ones).")
        print(f"  [warn] {msg} Continuing WITHOUT the symmetry fix.")
        return None

    try:
        import symfix
    except ImportError:
        return _fail("symfix.py is not importable, so the stored-basis symmetry "
                     "correction cannot be installed.")
    MFGD = _src_ns.get("MultiFileGraphDataset")
    if MFGD is None:
        return _fail("the model module has no MultiFileGraphDataset, which symfix "
                     "needs to read the stored bases.")
    zs = [int(z) for z in cfg.atomic_numbers]
    atomic_to_idx = {z: i for i, z in enumerate(zs)}
    merged = {}
    for split in ("train", "val", "test"):
        p = Path(dataset_path) / split / f"{split}_config.json"
        if p.exists():
            try:
                with open(p) as f:
                    for k, v in (json.load(f).get("element_properties", {}) or {}).items():
                        merged.setdefault(str(k), v)
            except Exception:
                pass
    onehot = np.eye(len(zs), dtype=np.float32)
    phys = _src_ns.get("build_element_phys_features")
    node_vectors = (np.concatenate([onehot, phys(zs, merged)], axis=1)
                    if phys is not None else onehot)
    n_sites = getattr(cfg, "n_sites", None)
    try:
        tr_ds = MFGD(dataset_path, TARGET_NAME, "train", atomic_to_idx, node_vectors,
                     **({} if n_sites is None else {"n_sites": int(n_sites)}))
    except TypeError:
        tr_ds = MFGD(dataset_path, TARGET_NAME, "train", atomic_to_idx, node_vectors)
    rep = symfix.install_stored_basis_symmetry(model, [tr_ds], _model_module())
    try:
        print(f"  [symfix] stored-basis symmetry installed: Wyckoff codebook "
              f"{rep['codebook_classes_before']} -> {rep['codebook_classes_after']} classes")
    except Exception:
        print("  [symfix] stored-basis symmetry installed.")
    return rep


def analyze_sites(structure, symprec=0.1):
    """Space group, one representative atom per symmetry orbit (the asymmetric
    unit) and a Wyckoff label such as '4e' for every atom (conventional setting)."""
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    sga = SpacegroupAnalyzer(structure, symprec=symprec)
    ss = sga.get_symmetrized_structure()
    reps, label = [], {}
    for idxs, wsym in zip(ss.equivalent_indices, ss.wyckoff_symbols):
        reps.append(int(idxs[0]))
        for i in idxs:
            label[int(i)] = str(wsym)
    return dict(number=int(sga.get_space_group_number()),
                symbol=str(sga.get_space_group_symbol()),
                reps=sorted(reps), wyckoff=label)


def hm_mathtext(sym):
    """'P6_3/mmc' -> P6$_{3}$/mmc ;  'Fm-3m' -> Fm$\\bar{3}$m"""
    t = re.sub(r"_(\d)", r"$_{\1}$", sym)
    return re.sub(r"-(\d)", r"$\\bar{\1}$", t)


def formula_mathtext(formula):
    return re.sub(r"(\d+)", r"$_{\1}$", formula)


def _panel_candidate(traj, g, max_atoms):
    """(entry or None, reason). Entry carries everything the drawing needs."""
    Z = traj["atomic_numbers"]
    frac, zZ = _final_atoms_Z(traj["final_exp"], Z)
    if len(zZ) == 0:
        return None, "empty cell"
    if len(zZ) > max_atoms:
        return None, f"more than {max_atoms} atoms"
    if traj.get("cdvae_struct_valid") is not True:
        return None, "fails the 0.5 A distance gate"
    if traj.get("comp_valid") is not True:
        return None, "fails SMACT composition gate"
    if (traj.get("n_distinct") or 0) < 2:
        return None, "unary"
    ok, pay, _rep = audit_structure(zZ, frac, traj["final_exp"]["L"].numpy())
    if not ok:
        return None, "fails the CIF audit"
    if not _HAVE_PMG:
        raise RuntimeError("pymatgen is required for the panel (symmetry analysis).")
    st = _PmgStructure(_PmgLattice(pay["L"]), [int(z) for z in pay["Z"]], pay["frac"],
                       coords_are_cartesian=False)
    try:
        info = analyze_sites(st)
    except Exception:
        return None, "symmetry analysis failed"
    if info["number"] != int(g):
        return None, f"spglib finds group {info['number']}, not the requested {g}"
    return dict(group=int(g), formula=st.composition.reduced_formula,
                formula_cell=st.composition.formula.replace(" ", ""),
                n_atoms=len(st), sg_number=info["number"], sg_symbol=info["symbol"],
                reps=info["reps"], wyckoff={str(k): v for k, v in info["wyckoff"].items()},
                Z=[int(z) for z in pay["Z"]], frac=pay["frac"].tolist(),
                L=pay["L"].tolist()), ""


def draw_panel_crystal(ax, e, view=(20, -62), bonds=False):
    """Draw one entry: asymmetric-unit atoms outlined in red and labelled with their
    Wyckoff position, symmetry copies faded."""
    frac = np.asarray(e["frac"], dtype=float) % 1.0
    Zs = [int(z) for z in e["Z"]]
    L = np.asarray(e["L"], dtype=float)
    reps = set(int(i) for i in e["reps"])
    n_before = len(ax.collections)
    _draw_crystal_3d(ax, frac, Zs, L, draw_bonds=bonds, show_labels=False, view=view,
                     highlight=sorted(reps))
    # _draw_crystal_3d adds one scatter collection per atom, in atom order
    atoms = [c for c in ax.collections[n_before:] if isinstance(c, Path3DCollection)]
    for i, coll in enumerate(atoms):
        if i in reps:
            coll.set_edgecolor("#d62728"); coll.set_linewidth(2.6)
        else:
            coll.set_alpha(0.30); coll.set_edgecolor("#888888"); coll.set_linewidth(0.4)
    cart = frac @ L
    for i in sorted(reps):
        x, y, z = cart[i]
        ax.text(x, y, z, e["wyckoff"][str(i)], fontsize=6.6, color="#b71c1c",
                fontweight="bold", ha="left", va="bottom", zorder=20,
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.8))
    ax.set_xlabel(""); ax.set_ylabel(""); ax.set_zlabel("")
    ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
    ax.tick_params(length=0)
    ax.set_axis_off()
    try:
        ax.set_box_aspect((1, 1, 1), zoom=1.0)
    except TypeError:
        pass


def render_sg_panel(entries, groups, per_group, out_path, dpi=300):
    """entries: {group: [entry, ...]}.  Columns = groups, rows = samples."""
    ncol, nrow = len(groups), int(per_group)
    with plt.rc_context({"mathtext.default": "regular"}):
        fig = plt.figure(figsize=(1.72 * ncol + 0.1, 2.55 * nrow + 0.25))
        gs = gridspec.GridSpec(2 * nrow, ncol,
                               height_ratios=[2.7, 0.78] * nrow, hspace=0.0, wspace=0.02,
                               left=0.01, right=0.99, top=0.995, bottom=0.03)
        for c, g in enumerate(groups):
            for r in range(nrow):
                lst = entries.get(g, [])
                ax = fig.add_subplot(gs[2 * r, c], projection="3d")
                tx = fig.add_subplot(gs[2 * r + 1, c]); tx.axis("off")
                if r < len(lst):
                    e = lst[r]
                    draw_panel_crystal(ax, e)
                    tx.text(0.5, 0.95, formula_mathtext(e["formula"]), ha="center",
                            va="top", fontsize=8.5, fontweight="bold",
                            transform=tx.transAxes)
                    tx.text(0.5, 0.52, f"{hm_mathtext(e['sg_symbol'])}  (No. {e['sg_number']})",
                            ha="center", va="top", fontsize=7.5, transform=tx.transAxes)
                else:
                    ax.set_axis_off()
                    tx.text(0.5, 0.6, "no eligible sample", ha="center", va="center",
                            fontsize=7, color="#999999", transform=tx.transAxes)
        fig.text(0.5, 0.004,
                 "red outline: asymmetric-unit atom, labelled with its Wyckoff position "
                 "(conventional setting); faded atoms are symmetry copies",
                 ha="center", va="bottom", fontsize=5.6, color="#555555")
        fig.savefig(out_path, dpi=dpi)
        pdf = os.path.splitext(out_path)[0] + ".pdf"
        fig.savefig(pdf)
        plt.close(fig)
    print(f"  panel -> {out_path} and {pdf}")


def panel_caption(entries, groups, per_group, stats):
    names = []
    for g in groups:
        lst = entries.get(g, [])
        if lst:
            sym = re.sub(r"_(\d)", r"_\1", re.sub(r"-(\d)", r"\\bar\1", lst[0]["sg_symbol"]))
            names.append("$" + sym + "$")
    return (r"\textbf{c}, Example generated structures, " + str(per_group) + r" each from "
            + ", ".join(names)
            + r", sampled with the flow and the decoder conditioned on the requested group. "
              r"A sample is shown if it passes the 0.5\,\AA{} distance test and the SMACT "
              r"composition test, contains at least two elements and is recovered by spglib "
              r"in exactly the requested group; the first samples satisfying this (with "
              r"distinct formulas) are drawn. The asymmetric unit is outlined in red and "
              r"labelled with Wyckoff positions (conventional setting); faded atoms are "
              r"symmetry copies. Formula and space group are given beneath each structure.")


def run_panel(model, cfg, device, args, gen_fn=None):
    """Generate `per_group` crystals for every requested group and draw them."""
    gen_fn = gen_fn or generate_with_trajectory
    groups = [int(g) for g in args.groups]
    per = int(args.per_group)
    expand = True if args.expand_orbits is None else bool(args.expand_orbits)
    if not expand:
        print("  [warn] orbit expansion is OFF: the panel would show only the "
              "asymmetric unit, so the requested symmetry cannot be seen.")
    stem = os.path.splitext(args.panel_output)[0]
    entries, stats = {}, {}
    for g in groups:
        got, backup, seen, why = [], [], set(), Counter()
        tried = 0
        for k in range(int(args.candidates)):
            if len(got) >= per:
                break
            tried += 1
            seed = args.seed + 1000 * g + k
            traj = gen_fn(model, None, device, num_steps=args.num_steps,
                          guidance=args.guidance_scale, snapshot_ts=[0.0, 1.0],
                          seed=seed, temperature=args.temperature, project_lattice=True,
                          max_unique_types=args.max_unique_types, expand_orbits=expand,
                          target_sg=g, cond_sg=None)
            ent, reason = _panel_candidate(traj, g, args.max_atoms)
            if ent is None:
                why[reason] += 1
                continue
            ent["seed"] = seed
            ent["traj_meta"] = dict(guidance=traj.get("guidance"),
                                    n_atoms_decoded=traj.get("n_atoms_decoded"))
            if ent["formula"] in seen:
                backup.append(ent); why["duplicate formula (kept as backup)"] += 1
                continue
            seen.add(ent["formula"]); got.append(ent)
        for ent in backup:                       # fill up only if distinct ones ran out
            if len(got) >= per:
                break
            got.append(ent)
        entries[g] = got
        stats[g] = dict(tried=tried, shown=len(got), rejected=dict(why))
        print(f"  SG {g:>3}: {len(got)}/{per} found in {tried} samples; "
              f"rejected: {dict(why) if why else 'none'}")

    render_sg_panel(entries, groups, per, args.panel_output, dpi=args.dpi)
    cap = panel_caption(entries, groups, per, stats)
    payload = dict(groups=groups, per_group=per, entries=entries, stats=stats,
                   checkpoint=os.path.basename(args.checkpoint), seed=args.seed,
                   caption_latex=cap)
    json_path = stem + ".json"
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=1)
    print(f"  selection -> {json_path}")
    if not args.no_save_cif:
        for g in groups:
            for k, e in enumerate(entries.get(g, [])):
                meta = dict(model="EquiCap / DirectCrystalFlow",
                            checkpoint=os.path.basename(args.checkpoint),
                            sampling=f"{args.num_steps} RK steps, unconditional in energy, "
                                     f"space group {g} requested, orbit expansion on",
                            target="unconditional",
                            sg_requested=f"{g} (conditioning input)",
                            sg_realized=e["sg_number"], seed=e["seed"])
                write_cif(e["Z"], np.asarray(e["frac"]), np.asarray(e["L"]),
                          f"{stem}_sg{g}_{k}.cif", meta=meta)
    print("\nSuggested caption:\n" + cap)
    return payload


def redraw_panel(json_path, out_path, dpi):
    d = json.load(open(json_path))
    entries = {int(g): v for g, v in d["entries"].items()}
    render_sg_panel(entries, [int(g) for g in d["groups"]], int(d["per_group"]),
                    out_path, dpi=dpi)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Visualize DirectCrystalFlow generation (direct flow model).")
    ap.add_argument("--checkpoint", default=CHECKPOINT)
    ap.add_argument("--dataset_path", default=DATASET_PATH)
    ap.add_argument("--target_energy", type=float, default=TARGET_ENERGY,
                    help="target formation energy (eV/atom)")
    ap.add_argument("--unconditional", action="store_true",
                    help="ignore --target_energy and sample unconditionally")
    ap.add_argument("--num_crystals", type=int, default=NUM_CRYSTALS)
    ap.add_argument("--num_steps", type=int, default=NUM_STEPS)
    ap.add_argument("--guidance_scale", type=float, default=GUIDANCE_SCALE,
                    help="classifier-free guidance scale; omit to use cfg.guidance")
    ap.add_argument("--temperature", type=float, default=TEMPERATURE)
    ap.add_argument("--max_unique_types", type=int, default=MAX_UNIQUE_TYPES,
                    help="cap each rendered crystal to this many unique element "
                         "types (matches the evaluator). 0 = raw generation.")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--output", default=OUTPUT)
    ap.add_argument("--dpi", type=int, default=DPI)
    ap.add_argument("--device", default=DEVICE)
    ap.add_argument("--no_save_cif", action="store_true")
    ap.add_argument("--no_project_lattice", action="store_true")
    ap.add_argument("--target_sg", type=int, default=None,
                    help="condition generation on this space group (1-230). "
                         "With an sg-conditioned checkpoint this steers the "
                         "flow itself; omit it to draw each sample's group "
                         "from the empirical training prior.")
    ap.add_argument("--expand_orbits", dest="expand_orbits",
                    action="store_true", default=EXPAND_ORBITS)
    ap.add_argument("--no_expand_orbits", dest="expand_orbits",
                    action="store_false")
    ap.add_argument("--cif_symmetrize", action="store_true",
                    default=CIF_SYMMETRIZE,
                    help="also write a symmetry-reduced copy, kept only if "
                         "expanding it reproduces the structure")
    ap.add_argument("--cond_sg", type=int, default=COND_SG,
                    help="space group fed to the FLOW, when it should differ "
                         "from the group the cell is built under")
    ap.add_argument("--sg_control", action="store_true", default=SG_CONTROL,
                    help="for every crystal, also generate a shuffled-"
                         "conditioning control in which the flow sees an "
                         "unrelated group while the cell is still built under "
                         "the requested one. The difference in exact-match "
                         "rate is the only part attributable to learned "
                         "conditioning.")
    ap.add_argument("--panel", action="store_true", default=PANEL_MODE,
                    help="space-group panel: generate crystals under requested groups "
                         "and draw them with the asymmetric unit and Wyckoff labels")
    ap.add_argument("--groups", type=int, nargs="+", default=PANEL_GROUPS,
                    help="space-group numbers for --panel (default: 2 14 148 194 225)")
    ap.add_argument("--per_group", type=int, default=PANEL_PER_GROUP)
    ap.add_argument("--candidates", type=int, default=PANEL_CANDIDATES,
                    help="max samples tried per group to find --per_group eligible ones")
    ap.add_argument("--max_atoms", type=int, default=PANEL_MAX_ATOMS)
    ap.add_argument("--panel_output", default=PANEL_OUTPUT)
    ap.add_argument("--panel_redraw", default=None,
                    help="redraw the panel from a saved .json; no model is loaded")
    ap.add_argument("--no_symfix", action="store_true",
                    help="do NOT install the stored-basis symmetry correction "
                         "(structures then follow the pre-fix expansion)")
    args, _ = ap.parse_known_args()
    if args.panel_redraw:
        redraw_panel(args.panel_redraw, args.panel_output, args.dpi)
        return
    save_cif = SAVE_CIF and not args.no_save_cif
    project_lattice = PROJECT_LATTICE and not args.no_project_lattice
    target_energy = None if args.unconditional else args.target_energy

    device = _resolve_device(args.device)
    print(f"Device: {device}")
    print(f"Lattice projection: {'on (SG family)' if project_lattice else 'OFF (raw cell)'}")
    model, cfg = _load_model(args.checkpoint, args.dataset_path, device)
    _install_symfix(model, cfg, args.dataset_path, strict=bool(args.panel),
                    enabled=not args.no_symfix)
    if args.panel:
        run_panel(model, cfg, device, args)
        print("Done.")
        return
    _exp_eff = (bool(getattr(cfg, "decode_asymmetric_unit", True)
                     and getattr(cfg, "use_wyckoff", True))
                if args.expand_orbits is None else bool(args.expand_orbits))
    print(f"Orbit expansion   : "
          f"{'on — panels C + CIF show the EXPANDED cell' if _exp_eff else 'off — panels C + CIF show the decoder output'}"
          f"{'  [from checkpoint cfg]' if args.expand_orbits is None else ''}")

    stem, ext = os.path.splitext(args.output)
    ext = ext or ".png"
    all_pred, all_err, all_reenc = [], [], []
    all_valid, all_sg_match, all_special = [], [], []
    all_ctrl, all_cdvae, all_distinct, all_empty = [], [], [], []
    tgt_label = "unconditional" if target_energy is None else f"{target_energy:+.3f} eV/atom"

    for i in range(args.num_crystals):
        print(f"\n[{i+1}/{args.num_crystals}] generating (target {tgt_label}, "
              f"seed {args.seed + i}) ...")
        traj = generate_with_trajectory(
            model, target_energy, device, num_steps=args.num_steps,
            guidance=args.guidance_scale, snapshot_ts=SNAPSHOT_TS,
            seed=args.seed + i, temperature=args.temperature,
            project_lattice=project_lattice,
            max_unique_types=args.max_unique_types,
            expand_orbits=args.expand_orbits, target_sg=args.target_sg,
            cond_sg=args.cond_sg)

        if args.sg_control:
            _req_g = traj.get("sg_requested")
            _rng = np.random.default_rng(args.seed + i)
            _fake = int(_rng.choice([g for g in range(1, 231) if g != _req_g]))
            try:
                _ctrl = generate_with_trajectory(
                    model, target_energy, device, num_steps=args.num_steps,
                    guidance=args.guidance_scale, snapshot_ts=[1.0],
                    seed=args.seed + i, temperature=args.temperature,
                    project_lattice=project_lattice,
                    max_unique_types=args.max_unique_types,
                    expand_orbits=args.expand_orbits, target_sg=_req_g,
                    cond_sg=_fake)
                _hit_m = (traj.get("sg_realized") == _req_g)
                _hit_c = (_ctrl.get("sg_realized") == _req_g)
                all_ctrl.append(bool(_hit_c))
                print(f"    [control] flow conditioned on SG {_fake} but cell "
                      f"built under SG {_req_g}: realised "
                      f"{_ctrl.get('sg_realized')} "
                      f"({'MATCH' if _hit_c else 'MISS'}); matched run "
                      f"{'MATCH' if _hit_m else 'MISS'}")
            except RuntimeError as e:
                print(f"    [control] unavailable: {e}")
                args.sg_control = False

        all_pred.append(traj["predicted"])
        if target_energy is not None:
            all_err.append(abs(traj["predicted"] - target_energy))
        if traj.get("valid") is not None:
            all_valid.append(bool(traj["valid"]))
        if traj.get("predicted_reenc") is not None:
            all_reenc.append(traj["predicted_reenc"])
        if traj.get("sg_realized") is not None:
            all_sg_match.append(
                traj["sg_realized"] == traj.get("sg_requested",
                                                traj.get("sg_first")))
        if traj.get("n_special") is not None:
            all_special.append(traj["n_special"])
        if traj.get("cdvae_struct_valid") is not None:
            all_cdvae.append(bool(traj["cdvae_struct_valid"]))
        if traj.get("n_distinct") is not None:
            all_distinct.append(int(traj["n_distinct"]))
        all_empty.append(int(traj.get("n_atoms_pre_expansion", 1)) == 0)
        _oi = traj.get("occ_info") or {}
        if _oi.get("saturated"):
            print(f"    [warn] the asymmetric unit fills all "
                  f"{traj['n_tokens']} site tokens: the count head is against "
                  f"the capacity ceiling, so this cell may be truncated. "
                  f"Raise cfg.n_sites.")
        _tdv = traj.get("token_div")
        if _tdv is not None and _tdv < 0.05:
            print(f"    [warn] latent token diversity {_tdv:.3f}: the occupied "
                  f"tokens are effectively identical, so the decoder cannot "
                  f"emit more than one element whatever its type head does.")
        if traj.get("n_distinct") == 1:
            print("    [warn] the cell is UNARY. It will pass the charge-"
                  "balance test trivially, so composition validity is not "
                  "evidence of anything here.")

        n_cell = int((traj["final_exp"]["mask"].numpy() > 0.5).sum())
        _re = traj.get("predicted_reenc")
        _rq = traj.get("sg_requested", traj.get("sg_first"))
        _sgtag = ("SG req " if traj.get("sg_conditioned") else "SG ")
        print(f"    latent {traj['predicted']:+.3f}"
              + (f"  structure {_re:+.3f}" if _re is not None else "")
              + f"  {_sgtag}{_rq}"
              f"{' → realized ' + str(traj['sg_realized']) + ('  MATCH' if traj['sg_realized'] == _rq else '  MISS') if traj.get('sg_realized') is not None else ''}  "
              f"vol {traj['volume']:.1f}  VPA {traj['vpa']:.1f}  "
              f"atoms(cell) {n_cell}")
        _npf = traj.get("n_atoms_pred_full")
        if _npf and np.isfinite(_npf) and n_cell:
            _r = n_cell / max(_npf, 1e-6)
            if _r < 0.6 or _r > 1.7:
                print(f"    [warn] the cell holds {n_cell} atoms but the volume "
                      f"was sized for {_npf:.1f}. VPA is off by ~{_r:.1f}x.")
        if n_cell != traj.get("n_atoms_decoded", n_cell):
            if n_cell < traj.get("n_atoms_decoded", n_cell):
                print(f"    [WARN] the cell has FEWER atoms than the decoder "
                      f"emitted. Orbit expansion may have shrunk the cell.")
            else:
                print(f"    [note] orbit expansion {n_cell/max(traj.get('n_atoms_decoded', n_cell),1):.1f}× "
                      f"(decoder → cell).")
        _es = traj.get("exp_stats") or {}
        if _es.get("n_expansion_failed"):
            print("    [WARN] expansion would have shrunk the cell; the "
                  "un-expanded structure was kept instead.")
        if _es.get("n_site_conflicts"):
            print(f"    [warn] {_es['n_site_conflicts']} cross-element site "
                  f"conflict(s): different elements landed on one position.")
        if _es.get("n_sites_collapsed"):
            print(f"    [note] {_es['n_sites_collapsed']} redundant site(s) on an "
                  f"already-occupied Wyckoff orbit collapsed to one representative "
                  f"per orbit (fixes multiplicity/volume and prevents cross-element "
                  f"conflicts before expansion).")
        if _es.get("n_types_reconciled"):
            print(f"    [note] {_es['n_types_reconciled']} site(s) reconciled to a "
                  f"single element because symmetry-equivalent positions were "
                  f"assigned different types (safety net).")

        fig = build_figure(traj, dpi=args.dpi)
        fig_path = f"{stem}_{i:02d}{ext}"
        fig.savefig(fig_path, dpi=args.dpi, bbox_inches="tight", facecolor="white")
        if SHOW_INLINE:
            try:
                from IPython.display import Image as _IPyImage, display as _disp
                _disp(_IPyImage(filename=fig_path))
            except Exception:
                pass
        plt.close(fig)
        print(f"  figure -> {fig_path}")

        if save_cif:
            fexp = traj["final_exp"]
            frac, zZ = _final_atoms_Z(fexp, traj["atomic_numbers"])
            _tgt_s = ("unconditional" if target_energy is None
                      else f"{target_energy:+.4f} eV/atom (formation energy)")
            _rq = traj.get("sg_requested")
            _meta = dict(
                model="EquiCap / DirectCrystalFlow",
                checkpoint=os.path.basename(args.checkpoint),
                sampling=(f"{args.num_steps} RK steps, guidance "
                          f"{traj['guidance']:.2f}, temperature "
                          f"{args.temperature:g}, orbit expansion "
                          f"{'on' if traj.get('expanded') else 'off'}"),
                target=_tgt_s,
                y_latent=f"{traj['predicted']:+.4f} eV/atom (model latent head)",
                y_reenc=("n/a" if traj.get("predicted_reenc") is None else
                         f"{traj['predicted_reenc']:+.4f} eV/atom "
                         f"(encoder re-read of these atoms)"),
                sg_requested=(f"{_rq} (conditioning input; the cell was BUILT "
                              f"under this group, so the realised value below "
                              f"is bounded below by construction)"),
                sg_realized=(traj.get("sg_realized")
                             if traj.get("sg_realized") is not None
                             else "not determined"),
                seed=args.seed + i,
                notes=list(traj.get("viz_edits") or []))
            if traj.get("cond_sg") is not None:
                _meta["notes"].append(
                    f"CONTROL RUN: the flow was conditioned on space group "
                    f"{traj['cond_sg']}, not {_rq}.")
            if traj.get("n_distinct") == 1:
                _meta["notes"].append(
                    "single-element cell; charge balance is satisfied "
                    "trivially at oxidation state zero.")
            ok_cif, _ = write_cif(zZ, frac, fexp["L"].numpy(),
                                  f"{stem}_{i:02d}.cif", meta=_meta,
                                  symmetrize=args.cif_symmetrize)
            if not ok_cif:
                print("  [cif] no file written for this sample.")

    print("\n" + "=" * 80 + "\nSUMMARY\n" + "=" * 80)
    print(f"Target energy     : {tgt_label}")
    print(f"Generated samples : {args.num_crystals}")
    if all_pred:
        print(f"Mean prediction   : {np.mean(all_pred):+.4f} ± {np.std(all_pred):.4f}"
              "   (LATENT readout — the model reading its own latent; proves nothing)")
    if all_reenc:
        print(f"Mean from structure: {np.mean(all_reenc):+.4f} ± "
              f"{np.std(all_reenc):.4f}   (frozen encoder + frozen property head "
              f"on the rendered atoms)")
    if all_err:
        print(f"Mean abs. error   : {np.mean(all_err):.4f} ± {np.std(all_err):.4f}"
              "   (latent vs target)")
    if all_reenc and target_energy is not None:
        _e = np.abs(np.array(all_reenc) - target_energy)
        print(f"Structure abs err : {_e.mean():.4f} ± {_e.std():.4f}"
              "   (the one to quote)")
    if all_cdvae:
        print(f"Validity (reference gate, 0.5 A) : {100*np.mean(all_cdvae):.0f}%"
              "   (the definition published numbers use)")
    if all_valid:
        print(f"Validity (model training criterion): {100*np.mean(all_valid):.0f}%"
              "   (stricter on every axis; NOT literature-comparable)")
    if all_distinct:
        _u = 100.0 * np.mean(np.array(all_distinct) == 1)
        print(f"Distinct elements / cell : mean {np.mean(all_distinct):.2f}"
              f"   ({_u:.0f}% of cells are UNARY)")
        if _u > 40:
            print("    ^ MP-20 is overwhelmingly multi-element. A unary "
                  "majority is composition collapse, and it inflates")
            print("      composition validity rather than being caught by it.")
    if any(all_empty):
        print(f"Empty cells       : {sum(all_empty)}/{len(all_empty)}"
              "   (decoder emitted no atoms; check force_nonempty)")
    if all_sg_match:
        _sgc = bool(getattr(getattr(model, "conditioner", None), "sg_emb", None)
                    is not None)
        if _sgc:
            print(f"SG realized==requested : {100*np.mean(all_sg_match):.0f}%"
                  "   (SPACE-GROUP CONTROLLABILITY — did the decoder and "
                  "expansion")
            print("                          actually realise the group the "
                  "sample was conditioned on?)")
        else:
            print(f"SG realized==driving : {100*np.mean(all_sg_match):.0f}%"
                  "   (the group that set the cell + expansion)")
    if all_ctrl and all_sg_match:
        _m, _c = 100 * np.mean(all_sg_match), 100 * np.mean(all_ctrl)
        print(f"SG match, matched conditioning : {_m:.0f}%")
        print(f"SG match, shuffled conditioning: {_c:.0f}%   (tautological floor)")
        print(f"CONTROLLABILITY (difference)   : {_m - _c:+.0f} points")
        if abs(_m - _c) < 5:
            print("    ^ the control matches the conditioned run. The "
                  "exact-match rate is produced by orbit expansion and the")
            print("      lattice-family projection, NOT by space-group "
                  "conditioning. Do not report it as controllability.")
    if all_special:
        print(f"Sites on special Wyckoff positions (mean): "
              f"{np.mean(all_special):.1f}"
              "   (0 everywhere means the Wyckoff head collapsed to the "
              "general position)")
    print("=" * 80)
    print("Read in this order. 1) 'Structure abs err' -- does the property "
          "conditioning survive into")
    print("the atoms? 2) 'Distinct elements / cell' -- has the composition "
          "collapsed? 3) CONTROLLABILITY,")
    print("not the raw SG match rate. 4) Validity last, and only under the "
          "reference gate. The latent")
    print("readout is the model grading its own homework, and composition "
          "validity rises under collapse.")
    print("=" * 80 + "\nDone.")


if __name__ == "__main__":
    main()