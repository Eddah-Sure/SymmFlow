"""SymmFlow -- conditional crystal generation for MP-20.

    noise ~ N(0, I)  ->  LatentSetFlow(z, t, cond(y, sg))  ->  site tokens
                     ->  SiteDecoder  ->  asymmetric unit  ->  cell  ->  crystal

The latent is a canonically ordered SET OF SITE TOKENS, one per crystallographic
site of the asymmetric unit, with a learned empty embedding in the unused slots.
Canonical ordering is what makes the flow-matching target single valued and what
allows DIRECT per-token supervision (token i must be element i, at position i);
chamfer and nearest-neighbour type losses cannot provide that, and both have
"every site on the most common element" at a shallow minimum.

Training runs in three stages:

  1  run_encoder_pretrain   encoder + decoder + structural heads as an
                            autoencoder, then freeze_encoder()
  2  run_direct_flow        flow matching onto the frozen encoder's latents
  3  run_stage3_finetune    roll the flow out from noise, decode, and grade the
                            crystal that comes out, with the reconstruction
                            branch as an anchor

Layout (see the section banners):

   1 constants and chemistry tables        7 losses
   2 geometry                              8 data
   3 symmetry                              9 training
   4 encoder                              10 sample-time post-processing, metrics
   5 generator                            11 configuration and entry points
   6 SymmFlow

Requires torch, torch_geometric, numpy, pandas; pymatgen for the space-group
operations (without it every symmetry number is inert and the file says so).
Checkpoints from the capsule version are not loadable.

"""
from __future__ import annotations
import os
import sys
import math
import csv
import json
import warnings
from dataclasses import dataclass, field
from typing import Optional, Dict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_dense_batch
warnings.filterwarnings('ignore')



_DATA_WARNED = set()

def _data_warn(msg: str, once: bool = True):
    key = msg if once else None
    if once:
        if key in _DATA_WARNED:
            return
        _DATA_WARNED.add(key)
    print(f"[mainmodel][WARN] {msg}", file=sys.stderr, flush=True)

SYM_SNAP_TOL = 0.01

_MID_ELEM_BUDGET = 32_000_000     # elements per intermediate tensor

ELEM_FEAT_DIM = 4

ATOMIC_RADII = {
    1:0.31,2:0.28,3:1.28,4:0.96,5:0.84,6:0.76,7:0.71,8:0.66,9:0.57,10:0.58,
    11:1.66,12:1.41,13:1.21,14:1.11,15:1.07,16:1.05,17:1.02,18:1.06,19:2.03,20:1.76,
    21:1.70,22:1.60,23:1.53,24:1.39,25:1.39,26:1.32,27:1.26,28:1.24,29:1.32,30:1.22,
    31:1.22,32:1.20,33:1.19,34:1.20,35:1.20,36:1.16,37:2.20,38:1.95,39:1.90,40:1.75,
    41:1.64,42:1.54,43:1.47,44:1.46,45:1.42,46:1.39,47:1.45,48:1.44,49:1.42,50:1.39,
    51:1.39,52:1.38,53:1.39,54:1.40,55:2.44,56:2.15,57:2.07,58:2.04,59:2.03,60:2.01,
    61:1.99,62:1.98,63:1.98,64:1.96,65:1.94,66:1.92,67:1.92,68:1.89,69:1.90,70:1.87,
    71:1.87,72:1.75,73:1.70,74:1.62,75:1.51,76:1.44,77:1.41,78:1.36,79:1.36,80:1.32,
    81:1.45,82:1.46,83:1.48,84:1.40,85:1.50,86:1.50,87:2.60,88:2.21,89:2.15,
    90:2.06,91:2.00,92:1.96,93:1.90,94:1.87,95:1.80,96:1.69,
}

OXIDATION_STATES = {
    1:[1,-1],3:[1],4:[2],5:[3],6:[4,-4,2],7:[-3,3,5],8:[-2],9:[-1],11:[1],12:[2],
    13:[3],14:[4,-4],15:[-3,3,5],16:[-2,4,6],17:[-1,5,7],19:[1],20:[2],21:[3],
    22:[4,3,2],23:[5,4,3,2],24:[3,6,2],25:[2,4,7],26:[3,2],27:[2,3],28:[2,3],
    29:[2,1],30:[2],31:[3],32:[4,2],33:[3,5,-3],34:[-2,4,6],35:[-1,5],37:[1],38:[2],
    39:[3],40:[4],41:[5,3],42:[4,6],44:[4,3],45:[3],46:[2,4],47:[1],48:[2],49:[3],
    50:[4,2],51:[3,5],52:[4,6,-2],53:[-1,5,7],55:[1],56:[2],57:[3],58:[3,4],59:[3],
    60:[3],62:[3,2],63:[3,2],64:[3],65:[3],66:[3],67:[3],68:[3],69:[3],70:[3,2],
    71:[3],72:[4],73:[5],74:[6,4],75:[4,7],76:[4],77:[3,4],78:[2,4],79:[3,1],80:[2,1],
    81:[1,3],82:[2,4],83:[3],89:[3],
    2:[0],10:[0],18:[0],36:[0],54:[0],
    43:[4,7],
    61:[3],
    90:[4],
    91:[5,4],
    92:[6,4,5,3],93:[5,4,6,3],94:[4,3,6,5],
    95:[3],96:[3],
    84:[4,2,-2,6],85:[-1,1,5],86:[0],87:[1],88:[2],
    97:[3],98:[3],
}

_PAULING_EN = {
    1: 2.20, 2: 0.0, 3: 0.98, 4: 1.57, 5: 2.04, 6: 2.55, 7: 3.04, 8: 3.44,
    9: 3.98, 10: 0.0, 11: 0.93, 12: 1.31, 13: 1.61, 14: 1.90, 15: 2.19,
    16: 2.58, 17: 3.16, 18: 0.0, 19: 0.82, 20: 1.00, 21: 1.36, 22: 1.54,
    23: 1.63, 24: 1.66, 25: 1.55, 26: 1.83, 27: 1.88, 28: 1.91, 29: 1.90,
    30: 1.65, 31: 1.81, 32: 2.01, 33: 2.18, 34: 2.55, 35: 2.96, 36: 3.00,
    37: 0.82, 38: 0.95, 39: 1.22, 40: 1.33, 41: 1.60, 42: 2.16, 43: 1.90,
    44: 2.20, 45: 2.28, 46: 2.20, 47: 1.93, 48: 1.69, 49: 1.78, 50: 1.96,
    51: 2.05, 52: 2.10, 53: 2.66, 54: 2.60, 55: 0.79, 56: 0.89, 57: 1.10,
    58: 1.12, 59: 1.13, 60: 1.14, 61: 1.13, 62: 1.17, 63: 1.20, 64: 1.20,
    65: 1.10, 66: 1.22, 67: 1.23, 68: 1.24, 69: 1.25, 70: 1.10, 71: 1.27,
    72: 1.30, 73: 1.50, 74: 2.36, 75: 1.90, 76: 2.20, 77: 2.20, 78: 2.28,
    79: 2.54, 80: 2.00, 81: 1.62, 82: 2.33, 83: 2.02, 84: 2.00, 85: 2.20,
    86: 2.20, 87: 0.70, 88: 0.90, 89: 1.10, 90: 1.30, 91: 1.50, 92: 1.38,
    93: 1.36, 94: 1.28,
}

NON_FORMABLE_Z = {
    2, 10, 18, 36, 54, 86,
    0,
}

SMACT_OXIDATION_STATES = {
    1:[-1, 1], 3:[1], 4:[2], 5:[-3, -2, -1, 1, 2, 3], 6:[-4, -3, -2, -1, 1, 2, 3, 4],
    7:[-5, -4, -3, -2, -1, 1, 2, 3, 4, 5], 8:[-2, -1], 9:[-1], 11:[1], 12:[2], 13:[3],
    14:[-4, -3, -2, -1, 2, 3, 4], 15:[-3, -2, -1, 1, 2, 3, 4, 5], 16:[-2, -1, 1, 2, 3, 4, 5, 6],
    17:[-1, 1, 3, 5, 7], 19:[1], 20:[2], 21:[1, 2, 3], 22:[2, 3, 4], 23:[1, 2, 3, 4, 5],
    24:[1, 2, 3, 4, 5, 6], 25:[-1, 1, 2, 3, 4, 5, 6, 7], 26:[1, 2, 3, 4, 5, 6],
    27:[-1, 1, 2, 3, 4], 28:[1, 2, 3, 4], 29:[1, 2, 3], 30:[2, 3], 31:[-3, 1, 2, 3, 4],
    32:[-4, -3, -2, -1, 2, 3, 4], 33:[-3, -2, -1, 1, 2, 3, 4, 5], 34:[-2, -1, 1, 2, 4, 6],
    35:[-1, 1, 3, 5, 7], 36:[2], 37:[1], 38:[2, 4], 39:[1, 2, 3, 4], 40:[1, 2, 3, 4],
    41:[1, 2, 3, 4, 5, 6], 42:[-1, 1, 2, 3, 4, 5, 6], 43:[1, 2, 3, 4, 5, 7],
    44:[1, 2, 3, 4, 5, 6, 7], 45:[-1, 1, 2, 3, 4, 5], 46:[1, 2, 3, 4], 47:[1, 2, 3], 48:[2],
    49:[-3, -1, 1, 2, 3], 50:[-4, -2, -1, 2, 3, 4], 51:[-3, -2, -1, 2, 3, 4, 5],
    52:[-3, -2, -1, 1, 2, 4, 5, 6], 53:[-1, 1, 3, 5, 7], 54:[2, 4, 6, 8], 55:[1], 56:[2],
    57:[1, 2, 3, 4], 58:[2, 3, 4], 59:[2, 3, 4], 60:[2, 3, 4], 62:[2, 3], 63:[2, 3, 4],
    64:[2, 3, 4], 65:[2, 3, 4], 66:[2, 3, 4], 67:[2, 3], 68:[2, 3, 4], 69:[2, 3], 70:[2, 3],
    71:[2, 3], 72:[2, 3, 4], 73:[1, 2, 3, 4, 5], 74:[2, 3, 4, 5, 6], 75:[1, 2, 3, 4, 5, 6, 7],
    76:[1, 2, 3, 4, 5, 6, 7, 8], 77:[1, 2, 3, 4, 5, 6], 78:[2, 3, 4, 5], 79:[-1, 1, 2, 3, 5],
    80:[1, 2], 81:[-1, 1, 2, 3], 82:[-4, -1, 2, 3, 4], 83:[-3, -2, -1, 1, 2, 3, 4, 5], 84:[4],
    89:[3], 90:[2, 3, 4], 91:[4, 5], 92:[2, 3, 4, 5, 6], 93:[2, 3, 4, 5, 6, 7],
    94:[2, 3, 4, 6, 7], 95:[2, 3, 4], 96:[3], 97:[3], 98:[3],
}

SMACT_PAULING_EN = {
    1:2.2, 3:0.98, 4:1.57, 5:2.04, 6:2.55, 7:3.04, 8:3.44, 9:3.98, 11:0.93, 12:1.31, 13:1.61,
    14:1.9, 15:2.19, 16:2.58, 17:3.16, 19:0.82, 20:1.0, 21:1.36, 22:1.54, 23:1.63, 24:1.66,
    25:1.55, 26:1.83, 27:1.88, 28:1.91, 29:1.9, 30:1.65, 31:1.81, 32:2.01, 33:2.18, 34:2.55,
    35:2.96, 36:3.0, 37:0.82, 38:0.95, 39:1.22, 40:1.33, 41:1.6, 42:2.16, 43:2.1, 44:2.2,
    45:2.28, 46:2.2, 47:1.93, 48:1.69, 49:1.78, 50:1.96, 51:2.05, 52:2.1, 53:2.66, 54:2.6,
    55:0.79, 56:0.89, 57:1.1, 58:1.12, 59:1.13, 60:1.14, 62:1.17, 63:1.2, 64:1.2, 65:1.2,
    66:1.22, 67:1.23, 68:1.24, 69:1.25, 70:1.1, 71:1.0, 72:1.3, 73:1.5, 74:1.7, 75:1.9, 76:2.2,
    77:2.2, 78:2.2, 79:2.4, 80:1.9, 81:1.8, 82:1.8, 83:1.9, 84:2.0, 85:2.2, 87:0.7, 88:0.9,
    89:1.1, 90:1.3, 91:1.5, 92:1.7, 93:1.3, 94:1.3, 95:1.3, 96:1.3, 97:1.3, 98:1.3, 99:1.3,
    100:1.3, 101:1.3, 102:1.3,
}

SMACT_METAL_Z = frozenset({3, 4, 11, 12, 13, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 55, 56, 57, 58, 59, 60, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102})

def smact_oxidation_states(z, fallback=None):
    """SMACT's oxidation states for atomic number `z`, with the narrow legacy
    table as a last resort."""
    z = int(z)
    if z in SMACT_OXIDATION_STATES:
        return list(SMACT_OXIDATION_STATES[z])
    if fallback is not None and z in fallback:
        return list(fallback[z])
    return list(OXIDATION_STATES.get(z, [0])) or [0]

def build_chemistry_tables(atomic_numbers):
    """(ox_padded, ox_mask, eneg, is_metal) indexed by VOCABULARY position."""
    zs = [int(z) for z in atomic_numbers]
    ox_lists = [smact_oxidation_states(z) for z in zs]
    S_ = max(1, max(len(o) for o in ox_lists))
    ox_pad = torch.zeros(len(zs), S_)
    ox_msk = torch.zeros(len(zs), S_)
    for i, o in enumerate(ox_lists):
        ox_pad[i, :len(o)] = torch.tensor([float(v) for v in o])
        ox_msk[i, :len(o)] = 1.0
    en = torch.tensor([float(SMACT_PAULING_EN.get(z, _PAULING_EN.get(z, 1.5)))
                       for z in zs])
    metal = torch.tensor([1.0 if z in SMACT_METAL_Z else 0.0 for z in zs])
    return ox_pad, ox_msk, en, metal

def build_element_phys_features(atomic_numbers, elem_props):
    phys = np.zeros((len(atomic_numbers), 4), dtype=np.float32)
    for idx, z in enumerate(atomic_numbers):
        ep = elem_props.get(str(z), elem_props.get(int(z), {})) or {}
        r = float(ep.get("atomic_radius", ATOMIC_RADII.get(int(z), 1.5)))
        ox = ep.get("oxidation_states", None) or OXIDATION_STATES.get(int(z), [0])
        ox = [float(v) for v in ox] if len(ox) else [0.0]
        ve = float(ep.get("valence_electrons", 0))
        phys[idx] = [
            r / 3.0,
            (float(np.mean(ox)) + 4.0) / 12.0,
            float(np.max(np.abs(ox))) / 8.0,
            ve / 18.0,
        ]
    return phys


def safe_norm(x, dim=-1, eps=1e-12, keepdim=False):
    return (x.pow(2).sum(dim=dim, keepdim=keepdim) + eps).sqrt()

def lattice_params_to_matrix(p: torch.Tensor) -> torch.Tensor:
    _sq = (p.dim() == 1)
    if _sq:
        p = p.unsqueeze(0)
    ANGL_LO, ANGL_HI = 20.0, 160.0
    a, b, c = p[:, 0], p[:, 1], p[:, 2]
    ang = p[:, 3:].clamp(ANGL_LO, ANGL_HI) * math.pi / 180
    al, be, ga = ang[:, 0], ang[:, 1], ang[:, 2]
    z = torch.zeros_like(a)
    v1 = torch.stack([a, z, z], -1)
    v2 = torch.stack([b * torch.cos(ga), b * torch.sin(ga), z], -1)
    cx = c * torch.cos(be)
    cy = c * (torch.cos(al) - torch.cos(be) * torch.cos(ga)) / torch.sin(ga).clamp(min=1e-4)
    cz = torch.sqrt((c * c - cx * cx - cy * cy).clamp(min=1e-8))
    out = torch.stack([v1, v2, torch.stack([cx, cy, cz], -1)], 1)
    return out.squeeze(0) if _sq else out

def matrix_to_lattice_params(L: torch.Tensor) -> torch.Tensor:

    _sq = (L.dim() == 2)
    if _sq:
        L = L.unsqueeze(0)
    v1, v2, v3 = L[:, 0], L[:, 1], L[:, 2]
    a, b, c = safe_norm(v1), safe_norm(v2), safe_norm(v3)
    def ang(u, v, nu, nv):
        return torch.acos(((u * v).sum(-1) / (nu * nv + 1e-8)).clamp(-1, 1)) * 180 / math.pi
    out = torch.stack([a, b, c, ang(v2, v3, b, c), ang(v1, v3, a, c), ang(v1, v2, a, b)], -1)
    return out.squeeze(0) if _sq else out

def project_to_crystal_family(p, sg):
    p = p.clone()
    sg = sg.view(-1).long().clamp(1, 230)
    ab = p[:, :2].mean(1); abc = p[:, :3].mean(1)
    mono = (sg >= 3) & (sg <= 15)
    ortho = (sg >= 16) & (sg <= 74)
    tetra = (sg >= 75) & (sg <= 142)
    trig = (sg >= 143) & (sg <= 167)
    hexg = (sg >= 168) & (sg <= 194)
    cubic = sg >= 195
    p[tetra, 0] = ab[tetra]; p[tetra, 1] = ab[tetra]
    p[trig, 0] = ab[trig]; p[trig, 1] = ab[trig]
    p[hexg, 0] = ab[hexg]; p[hexg, 1] = ab[hexg]
    p[cubic, 0] = abc[cubic]; p[cubic, 1] = abc[cubic]; p[cubic, 2] = abc[cubic]
    right = ortho | tetra | cubic
    p[right, 3] = 90.0; p[right, 4] = 90.0; p[right, 5] = 90.0
    p[mono, 3] = 90.0; p[mono, 5] = 90.0
    p[trig, 3] = 90.0; p[trig, 4] = 90.0; p[trig, 5] = 120.0
    p[hexg, 3] = 90.0; p[hexg, 4] = 90.0; p[hexg, 5] = 120.0
    return p

def _image_shifts(n_images: int, device, dtype):
    rng = torch.arange(-n_images, n_images + 1, device=device, dtype=dtype)
    return torch.stack(torch.meshgrid(rng, rng, rng, indexing='ij'), -1).reshape(-1, 3)

def _min_image_disp_block(fa, fb, L, n_images=1, chunk=9):
    df = fa.unsqueeze(2) - fb.unsqueeze(1)
    shifts = _image_shifts(n_images, L.device, df.dtype)
    S = shifts.size(0)
    chunk = max(1, int(chunk))
    best_v = None
    best_d = None
    for lo in range(0, S, chunk):
        sh = shifts[lo:lo + chunk]
        cand = df.unsqueeze(3) + sh.view(1, 1, 1, -1, 3)
        dv = torch.einsum('bijsk,bkl->bijsl', cand, L)
        dd = safe_norm(dv)
        dmin, idx = dd.min(dim=-1)
        gidx = idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, 1, 3)
        vmin = torch.gather(dv, 3, gidx).squeeze(3)
        if best_d is None:
            best_d, best_v = dmin, vmin
        else:
            upd = dmin < best_d
            best_d = torch.where(upd, dmin, best_d)
            best_v = torch.where(upd.unsqueeze(-1), vmin, best_v)
    return best_v, best_d

def min_image_disp(fa: torch.Tensor, fb: torch.Tensor, L: torch.Tensor,
                   n_images: int = 1, chunk: int = 9, row_chunk: int = None):
    """Minimum-image displacement/distance between two padded coordinate sets.
    """
    A = fa.size(1)
    B = fa.size(0)
    if row_chunk is None:
        per_row = max(1, B * fb.size(1) * max(1, int(chunk)) * 3)
        row_chunk = int(max(1, min(A, _MID_ELEM_BUDGET // per_row)))
    if row_chunk >= A:
        return _min_image_disp_block(fa, fb, L, n_images=n_images, chunk=chunk)
    vs, ds = [], []
    for lo in range(0, A, row_chunk):
        v, d = _min_image_disp_block(fa[:, lo:lo + row_chunk], fb, L,
                                     n_images=n_images, chunk=chunk)
        vs.append(v)
        ds.append(d)
    return torch.cat(vs, 1), torch.cat(ds, 1)

def pairwise_pbc(frac: torch.Tensor, L: torch.Tensor, n_images: int = 1):
    return min_image_disp(frac, frac, L, n_images=n_images)

def min_image_shift(frac: torch.Tensor, L: torch.Tensor, n_images: int = 1,
                    row_chunk: int = None):
    """Per-pair min-image lattice shift s such that (f_j + s - f_i) is nearest."""
    A = frac.size(1)
    B = frac.size(0)
    n_shift = (2 * int(n_images) + 1) ** 3
    if row_chunk is None:
        per_row = max(1, B * A * n_shift * 3)
        row_chunk = int(max(1, min(A, _MID_ELEM_BUDGET // per_row)))
    shifts = _image_shifts(n_images, L.device, frac.dtype)
    outs, dists = [], []
    for lo in range(0, A, row_chunk):
        df = frac[:, lo:lo + row_chunk].unsqueeze(2) - frac.unsqueeze(1)
        cand = -df.unsqueeze(3) + shifts.view(1, 1, 1, -1, 3)
        dv = torch.einsum('bijsk,bkl->bijsl', cand, L)
        dd = safe_norm(dv)
        dmin, idx = dd.min(dim=-1)
        outs.append(shifts[idx])
        dists.append(dmin)
    return torch.cat(outs, 1), torch.cat(dists, 1)

def lattice_self_edges(L, cutoff=6.0, max_self=6, n_images=2):
    """The shortest non-zero lattice translations within `cutoff`.
    """
    B = L.size(0)
    shifts = _image_shifts(n_images, L.device, L.dtype)          # (S, 3)
    nz = shifts.abs().sum(-1) > 0
    shifts = shifts[nz]
    vecs = torch.einsum('sk,bkl->bsl', shifts, L)                # (B, S, 3)
    d = safe_norm(vecs)                                          # (B, S)
    k = int(min(max_self, d.size(1)))
    dk, ik = d.topk(k, dim=1, largest=False)
    off = shifts.unsqueeze(0).expand(B, -1, -1).gather(
        1, ik.unsqueeze(-1).expand(B, k, 3))
    return off, dk, (dk < cutoff)

def compact_by_mask(mask, *tensors):
    """Move occupied slots to the front and cut the padding off the tail.
    """
    keep_n = int(mask.bool().sum(1).max().clamp(min=1))
    if keep_n >= mask.size(1):
        return (mask,) + tensors
    order = torch.argsort(mask.bool().to(torch.int8), dim=1, descending=True,
                          stable=True)[:, :keep_n]
    out = [torch.gather(mask, 1, order)]
    for t in tensors:
        if t is None:
            out.append(None)
        elif t.dim() == 2:
            out.append(torch.gather(t, 1, order))
        else:
            out.append(torch.gather(
                t, 1, order.view(*order.shape, *([1] * (t.dim() - 2)))
                 .expand(-1, -1, *t.shape[2:])))
    return tuple(out)


def build_edges_from_geometry(frac, L, mask, cutoff=6.0, max_deg=12, n_images=1,
                              max_self=6):
    """Padded (src, dst, offset, emask) for a structure that has no dataset graph.

    """
    B, A, _ = frac.shape
    sft, d = min_image_shift(frac, L, n_images)
    mb = mask.bool()
    eye = torch.eye(A, device=frac.device, dtype=torch.bool).unsqueeze(0)
    pair = (mb.unsqueeze(2) & mb.unsqueeze(1)) & ~eye
    BIG = 1.0e4
    dm = torch.where(pair, d, torch.full_like(d, BIG))
    k = int(min(max_deg, A))
    dk, jk = dm.topk(k, dim=2, largest=False)
    ok = (dk < cutoff) & (dk < 0.5 * BIG)
    src = torch.arange(A, device=frac.device).view(1, A, 1).expand(B, A, k)
    off = torch.gather(sft, 2, jk.unsqueeze(-1).expand(B, A, k, 3))

    src_l = src.reshape(B, A * k)
    dst_l = jk.reshape(B, A * k)
    off_l = off.reshape(B, A * k, 3)
    ok_l = ok.reshape(B, A * k)

    ms = int(max_self)
    if ms > 0:
        s_off, _s_d, s_ok = lattice_self_edges(L, cutoff=cutoff, max_self=ms)
        ar = torch.arange(A, device=frac.device).view(1, A, 1).expand(B, A, ms)
        self_src = ar.reshape(B, A * ms)
        self_dst = self_src
        self_off = s_off.view(B, 1, ms, 3).expand(B, A, ms, 3).reshape(B, A * ms, 3)
        self_ok = (s_ok.view(B, 1, ms) & mb.unsqueeze(-1)).reshape(B, A * ms)
        src_l = torch.cat([src_l, self_src], 1)
        dst_l = torch.cat([dst_l, self_dst], 1)
        off_l = torch.cat([off_l, self_off], 1)
        ok_l = torch.cat([ok_l, self_ok], 1)
    return src_l, dst_l, off_l, ok_l

def edge_geometry(frac, L, src, dst, offset):
    C3 = src.unsqueeze(-1).expand(-1, -1, 3)
    fi = torch.gather(frac, 1, C3)
    fj = torch.gather(frac, 1, dst.unsqueeze(-1).expand(-1, -1, 3))
    df = fj + offset - fi
    dvec = torch.einsum('bek,bkl->bel', df, L)
    return dvec, safe_norm(dvec)



_SG_OPS_CACHE = {}

_SG_OPS_ARR_CACHE = {}

def _pymatgen_available():
    try:
        import importlib.util
        return importlib.util.find_spec('pymatgen') is not None
    except Exception:
        return False

def _get_sg_ops(sg, mined_ops=None):
    if mined_ops is not None:
        rot, trans = mined_ops
        rot = np.asarray(rot, dtype=float)
        trans = np.asarray(trans, dtype=float)
        if rot.ndim == 3 and rot.shape[0] > 0:
            return [(rot[i], trans[i]) for i in range(rot.shape[0])]
    if sg in _SG_OPS_CACHE:
        return _SG_OPS_CACHE[sg]
    ops = []
    try:
        from pymatgen.symmetry.groups import SpaceGroup
        for op in SpaceGroup.from_int_number(int(sg)).symmetry_ops:
            rot = np.asarray(op.rotation_matrix, dtype=float)
            t = np.asarray(op.translation_vector, dtype=float)
            ops.append((rot, t))
    except Exception:
        ops = []
    _SG_OPS_CACHE[sg] = ops
    return ops

def _sg_ops_arrays(sg, mined_ops=None):
    """Stacked (rot (K,3,3), trans (K,3)) for a space group, cached."""
    key = int(sg)
    if mined_ops is None and key in _SG_OPS_ARR_CACHE:
        return _SG_OPS_ARR_CACHE[key]
    ops = _get_sg_ops(key, mined_ops=mined_ops)
    if not ops:
        out = (np.zeros((0, 3, 3)), np.zeros((0, 3)))
    else:
        out = (np.stack([np.asarray(r, dtype=float) for r, _ in ops], 0),
               np.stack([np.asarray(t, dtype=float) for _, t in ops], 0))
    if mined_ops is None:
        _SG_OPS_ARR_CACHE[key] = out
    return out

def _site_symmetry_snap(f, ops_arr, tol=SYM_SNAP_TOL):
    """Pull `f` (n, 3) onto the operations that almost fix it. Vectorised over
    both sites and operations; the previous version looped over |G| in Python
    once per site, i.e. n * |G| iterations per crystal."""
    rot, trans = ops_arr
    f = np.atleast_2d(np.asarray(f, dtype=float))
    if rot.shape[0] == 0:
        return f % 1.0
    d = np.einsum('kij,nj->nki', rot, f) + trans[None] - f[:, None, :]
    d -= np.round(d)
    near = np.abs(d).max(-1) < tol                                # (n, K)
    cnt = np.maximum(near.sum(1), 1)[:, None]
    return (f + (d * near[..., None]).sum(1) / cnt) % 1.0

def _min_image_sq_matrix(fa, fb, L, invL):
    """Pairwise squared minimum-image cartesian distances between two
    fractional coordinate sets."""
    d = np.asarray(fa)[:, None, :] - np.asarray(fb)[None, :, :]
    d -= np.round(d)
    c = d @ L
    return (c * c).sum(-1)

def _orbit_min_contact(orbit, L_use, invL):
    """Shortest distance between two DISTINCT members of an orbit, in Angstrom."""
    if orbit is None or len(orbit) < 2:
        return float('inf')
    pts = np.asarray(orbit)
    dd = np.sqrt(_min_image_sq_matrix(pts, pts, L_use, invL))
    np.fill_diagonal(dd, np.inf)
    return float(dd.min())

_SNAP_LADDER = (0.02, 0.035, 0.05, 0.07, 0.10, 0.14, 0.20, 0.26)


def _physical_site_snap(f, ops_arr, L_use, invL, base_tol=SYM_SNAP_TOL,
                        min_dist=0.75, max_tol=0.26):
    """Snap one site onto a special position.
    """
    f1 = np.asarray(f, dtype=float).reshape(1, 3)
    tol = float(base_tol)
    s = _site_symmetry_snap(f1, ops_arr, tol=tol)[0]
    orb = _orbit_of(s, ops_arr, L_use, invL, tol_sq=1e-4)
    d = _orbit_min_contact(orb, L_use, invL)
    if d >= min_dist or len(orb) <= 1:
        return s, orb, tol, d, False
    best = (s, orb, tol, d)
    for t in _SNAP_LADDER:
        if t <= tol or t > max_tol:
            continue
        s2 = _site_symmetry_snap(f1, ops_arr, tol=t)[0]
        o2 = _orbit_of(s2, ops_arr, L_use, invL, tol_sq=1e-4)
        d2 = _orbit_min_contact(o2, L_use, invL)
        if d2 > best[3]:
            best = (s2, o2, t, d2)
        if d2 >= min_dist:
            return s2, o2, t, d2, True
    return best[0], best[1], best[2], best[3], True


def _orbit_of(f, ops_arr, L_use, invL, tol_sq=None, merge_dist=0.75):
    """Distinct images of one site under the group, as an (m, 3) array.
    """
    ts = float(tol_sq) if tol_sq is not None else float(merge_dist) ** 2
    rot, trans = ops_arr
    if rot.shape[0] == 0:
        return np.asarray(f, dtype=float).reshape(1, 3) % 1.0
    img = (np.einsum('kij,j->ki', rot, np.asarray(f, dtype=float)) + trans) % 1.0
    d2 = _min_image_sq_matrix(img, img, L_use, invL)
    dup = (d2 < ts) & (np.tril(np.ones_like(d2, dtype=bool), -1))
    return img[~dup.any(1)]

_WYCKOFF_CACHE = {}

_WYCKOFF_TENSORS = None

_WYCKOFF_MAX = 27

_WYCKOFF_AXIS_FIXED = (
    0.0, 1.0 / 8, 1.0 / 6, 1.0 / 4, 1.0 / 3, 3.0 / 8, 1.0 / 2,
    5.0 / 8, 2.0 / 3, 3.0 / 4, 5.0 / 6, 7.0 / 8,
)

_WYCKOFF_AXIS_FREE = (0.137, 0.271, 0.393)

_WYCKOFF_PTS = None

def _wyckoff_axis_values():
    """Grid of candidate coordinate values, including derived ones.
    """
    vals = set(float(v) for v in _WYCKOFF_AXIS_FIXED)
    for x in _WYCKOFF_AXIS_FREE:
        for v in (x, 2.0 * x, -x, 1.0 - 2.0 * x,
                  0.5 + x, 0.5 - x, 0.5 + 2.0 * x, 0.5 - 2.0 * x,
                  1.0 / 3.0 + x, 2.0 / 3.0 - x):
            vals.add(float(v % 1.0))
    return tuple(sorted(vals))

_WYCKOFF_AXIS_VALUES = _wyckoff_axis_values()

def _wyckoff_candidate_points():
    global _WYCKOFF_PTS
    if _WYCKOFF_PTS is not None:
        return _WYCKOFF_PTS
    g = np.asarray(_WYCKOFF_AXIS_VALUES, dtype=np.float64)
    pts = np.stack(np.meshgrid(g, g, g, indexing='ij'), -1).reshape(-1, 3)
    _, keep = np.unique(np.round(pts, 6), axis=0, return_index=True)
    _WYCKOFF_PTS = pts[np.sort(keep)]
    return _WYCKOFF_PTS

def _site_projectors(rot, trans, pts, tol=1e-4, chunk=4096):
    K = rot.shape[0]
    Ps, os_, nfs = [], [], []
    for lo in range(0, pts.shape[0], int(chunk)):
        p = pts[lo:lo + int(chunk)]
        img = p @ np.transpose(rot, (0, 2, 1)) + trans[:, None, :]
        d = img - p[None]
        d = d - np.round(d)
        stab = (np.abs(d).max(-1) < tol)
        cnt = np.maximum(stab.sum(0), 1).astype(np.float64)
        P = (stab.T.astype(np.float64) @ rot.reshape(K, 9)).reshape(-1, 3, 3)
        P = P / cnt[:, None, None]
        P = np.where(np.abs(P) < 1e-9, 0.0, P)
        o = p - np.squeeze(P @ p[:, :, None], -1)
        o = o - np.floor(o)
        nf = np.clip(np.rint(np.trace(P, axis1=1, axis2=2)).astype(np.int64), 0, 3)
        Ps.append(P); os_.append(o); nfs.append(nf)
    return np.concatenate(Ps, 0), np.concatenate(os_, 0), np.concatenate(nfs, 0)

def _orbit_key(P, o, rot, rot_inv, trans, dec=4):
    Pp = (rot @ P) @ rot_inv
    op = (np.squeeze(rot @ o[None, :, None], -1) + trans) % 1.0
    feats = np.round(np.concatenate([Pp.reshape(len(rot), 9), op], 1), dec) + 0.0
    order = np.lexsort(feats.T[::-1])
    return tuple(feats[order[0]].tolist())

def _wyckoff_sites(sg, max_wyckoff=_WYCKOFF_MAX):
    key = (int(sg), int(max_wyckoff))
    if key in _WYCKOFF_CACHE:
        return _WYCKOFF_CACHE[key]

    free_site = [('a', 3, np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32))]
    ops = _get_sg_ops(int(sg))
    if not ops:
        _WYCKOFF_CACHE[key] = free_site
        return free_site

    rot = np.stack([np.asarray(r, dtype=np.float64) for r, _ in ops], 0)
    trans = np.stack([np.asarray(t, dtype=np.float64) for _, t in ops], 0)
    pts = _wyckoff_candidate_points()

    try:
        P, o, nfree = _site_projectors(rot, trans, pts)
    except Exception:
        _WYCKOFF_CACHE[key] = free_site
        return free_site

    sel = np.nonzero(nfree < 3)[0]
    if sel.size == 0:
        _WYCKOFF_CACHE[key] = free_site
        return free_site
    Ps, os_, nfs = P[sel], o[sel], nfree[sel]
    feat = np.concatenate([np.round(Ps.reshape(-1, 9), 4) + 0.0,
                           np.round(os_, 4) + 0.0], axis=1)
    _, first = np.unique(feat, axis=0, return_index=True)
    first = np.sort(first)
    uniq = [(int(nfs[c]), Ps[c], os_[c]) for c in first]


    uniq.sort(key=lambda u: (u[0], tuple(np.round(u[2], 6)),
                             tuple(np.round(u[1].ravel(), 6))))
    try:
        rot_inv = np.linalg.inv(rot)
    except np.linalg.LinAlgError:
        rot_inv = None
    seen_orbit, sites = set(), []
    for nf, Pm, om in uniq:
        if nf >= 3:
            continue
        if rot_inv is not None:
            ok = _orbit_key(Pm, om, rot, rot_inv, trans)
            if ok in seen_orbit:
                continue
            seen_orbit.add(ok)
        sites.append((nf, Pm, om))
        if len(sites) >= int(max_wyckoff) - 1:
            break
    sites.append((3, np.eye(3), np.zeros(3)))

    out = [(chr(ord('a') + i) if i < 26 else 'A', int(nf),
            Pm.astype(np.float32), om.astype(np.float32))
           for i, (nf, Pm, om) in enumerate(sites)]
    if not out:
        out = free_site
    _WYCKOFF_CACHE[key] = out
    return out

def _build_wyckoff_tensors(max_wyckoff=_WYCKOFF_MAX):
    W = int(max_wyckoff)
    proj = torch.eye(3, dtype=torch.float32).view(1, 1, 3, 3).repeat(230, W, 1, 1)
    off = torch.zeros(230, W, 3, dtype=torch.float32)
    nfree = torch.full((230, W), 3, dtype=torch.int32)
    valid = torch.zeros(230, W, dtype=torch.bool)
    gen = torch.zeros(230, dtype=torch.long)

    n_real = 0
    _t0 = __import__('time').time()
    for sg in range(1, 231):
        sites = _wyckoff_sites(sg, W)
        for wi, (_letter, nf, P, o) in enumerate(sites[:W]):
            proj[sg - 1, wi] = torch.as_tensor(P, dtype=torch.float32)
            off[sg - 1, wi] = torch.as_tensor(o, dtype=torch.float32)
            nfree[sg - 1, wi] = int(nf)
            valid[sg - 1, wi] = True
        gi = max(0, min(len(sites), W) - 1)
        gen[sg - 1] = gi
        valid[sg - 1, gi] = True          # the general position is always available
        n_real += len(sites)
    print(f"  [wyckoff] enumerated {n_real} sites across 230 space groups "
          f"in {__import__('time').time() - _t0:.1f}s (cached for next time).")
    if n_real <= 230:
        _data_warn(
            "Wyckoff table built with one site per space group -- pymatgen is "
            "probably unavailable, so the decoder's Wyckoff conditioning is "
            "inactive (anchors stay unconstrained).")
    return proj, off, nfree, valid, gen

def wyckoff_multiplicity_table(P_lut, o_lut, valid_lut, probe=(0.137, 0.271, 0.393),
                               tol_frac=1e-3, verbose=True):
    """Multiplicity of every (space group, Wyckoff class) in the codebook.
    """
    import time as _time
    R, T, ok = _get_sym_op_tensors()
    W = int(P_lut.size(1))
    u = torch.as_tensor(probe, dtype=torch.float32).view(1, 3)
    mult = torch.ones(230, W, dtype=torch.float32)
    t0 = _time.time()
    for sg0 in range(230):
        okg = ok[sg0]
        M = int(okg.sum())
        if M == 0:
            continue
        Rg, Tg = R[sg0][okg], T[sg0][okg]                     # (M,3,3) (M,3)
        rep_pt = (torch.einsum('wij,bj->wi', P_lut[sg0], u.expand(W, 3))
                  + o_lut[sg0]) % 1.0                          # (W,3)
        img = (torch.einsum('mij,wj->mwi', Rg, rep_pt) + Tg.unsqueeze(1)) % 1.0
        d = img.unsqueeze(0) - img.unsqueeze(1)                 # (M,M,W,3)
        d = d - d.round()
        close = d.abs().amax(-1) < tol_frac                     # (M,M,W)
        earlier = torch.tril(torch.ones(M, M, dtype=torch.bool), -1)
        dup = (close & earlier.unsqueeze(-1)).any(1)            # (M,W)
        mult[sg0] = (~dup).sum(0).clamp(min=1).float()
    mult = torch.where(valid_lut, mult, torch.ones_like(mult))
    if verbose:
        print(f"  [wyckoff] multiplicity table for {int(valid_lut.sum())} classes "
              f"in {_time.time() - t0:.1f}s "
              f"(mean {float(mult[valid_lut].mean()):.2f}, "
              f"max {int(mult[valid_lut].max())}).")
    return mult


def wyckoff_sites(sg, max_wyckoff=_WYCKOFF_MAX):
    sites = _wyckoff_sites(sg, max_wyckoff)
    P = np.stack([s[2] for s in sites], 0).astype(np.float32)
    o = np.stack([s[3] for s in sites], 0).astype(np.float32)
    nfree = np.array([s[1] for s in sites], dtype=np.int32)
    return P, o, nfree

def _wyckoff_cache_path(max_wyckoff):
    root = os.environ.get('EMF_CACHE_DIR') or os.path.join(
        os.path.expanduser('~'), '.cache', 'emergent_motif_flow')
    return os.path.join(root, f'wyckoff_lut_v5_w{int(max_wyckoff)}.npz')

def _get_wyckoff_tensors(max_wyckoff=_WYCKOFF_MAX, use_cache=True):
    global _WYCKOFF_TENSORS
    if _WYCKOFF_TENSORS is not None:
        return _WYCKOFF_TENSORS

    path = _wyckoff_cache_path(max_wyckoff)
    if use_cache and os.path.exists(path):
        try:
            with np.load(path) as z:
                _proj = torch.as_tensor(z['proj'], dtype=torch.float32)
                _off = torch.as_tensor(z['off'], dtype=torch.float32)
                _nf = torch.as_tensor(z['nfree'], dtype=torch.int32)
                _ok = torch.as_tensor(z['valid']).bool()
                _gen = torch.as_tensor(z['gen']).long()
            if _proj.shape == (230, int(max_wyckoff), 3, 3):
                if int(_ok.sum()) <= 230 * 2 and _pymatgen_available():
                    _data_warn("Discarding the cached Wyckoff table: it was built "
                               "without pymatgen (one site per space group) but "
                               "pymatgen is available now. Rebuilding.")
                else:
                    _ok.scatter_(1, _gen.view(-1, 1), True)
                    _WYCKOFF_TENSORS = (_proj, _off, _nf, _ok, _gen)
                    return _WYCKOFF_TENSORS
        except Exception:
            pass

    _WYCKOFF_TENSORS = _build_wyckoff_tensors(max_wyckoff)
    if use_cache:
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            proj, off, nfree, valid, gen = _WYCKOFF_TENSORS
            np.savez_compressed(path, proj=proj.numpy(), off=off.numpy(),
                                nfree=nfree.numpy(), valid=valid.numpy(),
                                gen=gen.numpy())
        except Exception:
            pass
    return _WYCKOFF_TENSORS

def build_wyckoff_codebook(datasets, max_w=27, verbose=True):
    """(space group, Wyckoff letter) -> (projector, offset), taken from the data.
    """
    from collections import Counter
    acc = {}
    for ds in datasets:
        sgs = getattr(ds, 'space_groups', None)
        for gi, g in enumerate(ds.graph_data):
            sg = int(sgs[gi]) if sgs is not None else int(getattr(g, 'space_group', 1))
            sg = min(max(sg, 1), 230)
            wi = np.asarray(getattr(g, 'wyckoff_indices', None)).reshape(-1)
            am = np.asarray(getattr(g, 'asym_unit_mask', None), dtype=bool).reshape(-1)
            Pm = np.asarray(getattr(g, 'wyckoff_proj', None), dtype=np.float64)
            om = np.asarray(getattr(g, 'wyckoff_offset', None), dtype=np.float64)
            nf = np.asarray(getattr(g, 'wyckoff_n_free', None)).reshape(-1)
            if wi.size != am.size or Pm.shape[0] != am.size:
                continue
            for i in np.nonzero(am)[0]:
                w = int(wi[i])
                if not (0 <= w < max_w):
                    continue
                key = (sg - 1, w)
                sig = (tuple(np.round(Pm[i].reshape(-1), 4)),
                       tuple(np.round(om[i] % 1.0, 4)), int(nf[i]))
                acc.setdefault(key, Counter())[sig] += 1

    Pl = torch.eye(3).view(1, 1, 3, 3).repeat(230, max_w, 1, 1).contiguous()
    Ol = torch.zeros(230, max_w, 3)
    Vl = torch.zeros(230, max_w, dtype=torch.bool)
    Nf = torch.full((230, max_w), 3, dtype=torch.long)
    # How often the data actually USES each class. The generated branch has no
    # per-site Wyckoff label to be graded against, so this empirical marginal is
    # what tells the Wyckoff head which classes a real crystal in this group
    # would populate (see `wyckoff_prior_loss`).
    Cn = torch.zeros(230, max_w)
    n_amb = 0
    for (sg0, w), ctr in acc.items():
        sig, cnt = ctr.most_common(1)[0]
        Cn[sg0, w] = float(sum(ctr.values()))
        if cnt < sum(ctr.values()):
            n_amb += 1
        Pl[sg0, w] = torch.tensor(sig[0], dtype=torch.float32).view(3, 3)
        Ol[sg0, w] = torch.tensor(sig[1], dtype=torch.float32)
        Nf[sg0, w] = int(sig[2])
        Vl[sg0, w] = True
    # Every group must offer at least one class, or its masked softmax is empty.
    for sg0 in range(230):
        if not bool(Vl[sg0].any()):
            Vl[sg0, max_w - 1] = True
    gen = Nf.masked_fill(~Vl, -1).argmax(dim=1)
    if verbose:
        print(f"  [wyckoff] codebook from data: {int(Vl.sum())} (group, letter) "
              f"classes over {len(acc)} observed"
              + (f"; {n_amb} had several offsets, modal value taken" if n_amb else ""))
    return Pl, Ol, Nf, Vl, gen, Cn

_SYM_TENSORS = None

_SYM_MAX_OPS = 192   # max |G| over the 230 groups

def _build_sym_op_tensors(max_ops=_SYM_MAX_OPS):
    M = int(max_ops)
    R = torch.eye(3, dtype=torch.float32).view(1, 1, 3, 3).repeat(230, M, 1, 1)
    T = torch.zeros(230, M, 3, dtype=torch.float32)
    ok = torch.zeros(230, M, dtype=torch.bool)
    for sg in range(1, 231):
        ops = _get_sg_ops(int(sg)) or [(np.eye(3), np.zeros(3))]
        # identity first, so index 0 always reproduces the asymmetric unit itself
        ops = sorted(ops, key=lambda rt: (np.abs(rt[0] - np.eye(3)).sum()
                                          + np.abs(rt[1]).sum()))
        for i, (r, t) in enumerate(ops[:M]):
            R[sg - 1, i] = torch.as_tensor(np.asarray(r), dtype=torch.float32)
            T[sg - 1, i] = torch.as_tensor(np.asarray(t), dtype=torch.float32)
            ok[sg - 1, i] = True
    ok[:, 0] = True
    return R, T, ok

def _sym_cache_path(max_ops):
    root = os.environ.get('EMF_CACHE_DIR') or os.path.join(
        os.path.expanduser('~'), '.cache', 'emergent_motif_flow')
    return os.path.join(root, f'symops_lut_v2_m{int(max_ops)}.npz')

def _get_sym_op_tensors(max_ops=_SYM_MAX_OPS, use_cache=True):
    global _SYM_TENSORS
    if _SYM_TENSORS is not None:
        return _SYM_TENSORS
    path = _sym_cache_path(max_ops)
    if use_cache and os.path.exists(path):
        try:
            with np.load(path) as z:
                R = torch.as_tensor(z['rot'], dtype=torch.float32)
                T = torch.as_tensor(z['trans'], dtype=torch.float32)
                ok = torch.as_tensor(z['valid']).bool()
            if R.shape == (230, int(max_ops), 3, 3):
                # identity-only table == built without pymatgen
                if int(ok.sum()) <= 230 and _pymatgen_available():
                    _data_warn("Discarding the cached space-group operation "
                               "table: it holds only the identity for every "
                               "group (built without pymatgen), but pymatgen is "
                               "available now. Rebuilding.")
                else:
                    _SYM_TENSORS = (R, T, ok)
                    return _SYM_TENSORS
        except Exception:
            pass
    _SYM_TENSORS = _build_sym_op_tensors(max_ops)
    _n_ops = int(_SYM_TENSORS[2].sum())
    print(f"  [symops] {_n_ops} operations across 230 space groups.")
    if _n_ops <= 230:
        _data_warn(
            "The space-group operation table holds only the identity -- pymatgen "
            "is unavailable, so orbit multiplicity, symmetry-aware repulsion and "
            "expand_generated() are all inert. Install pymatgen and delete "
            f"{_sym_cache_path(max_ops)} before trusting any symmetry number.")
    if use_cache:
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            R, T, ok = _SYM_TENSORS
            np.savez_compressed(path, rot=R.numpy(), trans=T.numpy(),
                                valid=ok.numpy())
        except Exception:
            pass
    return _SYM_TENSORS

def _first_occurrence(f, ok, L=None, tol_ang=1e-2, elem_budget=_MID_ELEM_BUDGET):
    """Mark, for each atom, the images that are not duplicates of an earlier one.
    """
    B, M, A, _ = f.shape
    dup = torch.zeros(B, M, A, dtype=torch.bool, device=f.device)
    per_row = max(1, B * M * A * 3)
    chunk = int(max(1, min(M, elem_budget // per_row)))
    idx = torch.arange(M, device=f.device)
    okc = ok.view(B, 1, M, 1)
    for lo in range(0, M, chunk):
        hi = min(lo + chunk, M)
        d = f[:, lo:hi].unsqueeze(2) - f.unsqueeze(1)        # (B, c, M, A, 3)
        d = d - d.round()
        if L is not None:
            close = safe_norm(torch.einsum('bghak,bkl->bghal', d, L)) < tol_ang
        else:
            close = d.abs().amax(-1) < 1e-3
        earlier = (idx.view(1, M) < idx[lo:hi].view(-1, 1)).view(1, hi - lo, M, 1)
        dup[:, lo:hi] = (close & earlier & okc).any(2)
    return ok.view(B, M, 1) & ~dup

def _clamp_st(x, lo, hi):
    
    xc = torch.minimum(torch.maximum(x, lo), hi)
    return x + (xc - x).detach()


def orbit_multiplicity(frac, mask, sg, R_lut, T_lut, ok_lut, L=None,
                       snap_tol=SYM_SNAP_TOL, tol_ang=1e-2, merge_dist=None):
    """Wyckoff multiplicity of each decoded atom under its space group.

    """
    B, A, _ = frac.shape
    sgi = sg.view(-1).clamp(1, 230).long() - 1
    R, T, ok = R_lut[sgi], T_lut[sgi], ok_lut[sgi]
    M = R.size(1)

    # site-symmetry snap: pull the coordinate onto the operations that almost
    # fix it, matching _site_symmetry_snap on the numpy side
    img = (torch.einsum('bmij,baj->bmai', R, frac) + T.unsqueeze(2))
    dv = img - frac.unsqueeze(1)
    dv = dv - dv.round()
    near = (dv.abs().amax(-1) < snap_tol) & ok.unsqueeze(-1)      # (B, M, A)
    nz = near.float().sum(1).clamp(min=1.0)
    frac = (frac + (dv * near.unsqueeze(-1).float()).sum(1) / nz.unsqueeze(-1)) % 1.0

    f = (torch.einsum('bmij,baj->bmai', R, frac) + T.unsqueeze(2)) % 1.0
    keep = torch.zeros(B, M, A, dtype=torch.bool, device=frac.device)
    keep[:, 0] = ok[:, 0].unsqueeze(-1).expand(B, A)
    _present = ok.any(0).nonzero()
    M_eff = int(_present.max().item()) + 1 if _present.numel() else 1

    # Sequential keep-mask update: coincidence of exact group images is an
    # equivalence relation, so a pairwise reduction over already-kept images
    # is enough to find the orbit's distinct representatives.
    _t = float(tol_ang if merge_dist is None else max(tol_ang, merge_dist))
    for g in range(1, M_eff):
        d = f[:, g:g + 1] - f[:, :g]
        d = (d + 0.5) % 1.0 - 0.5
        if L is not None:
            dd = safe_norm(torch.einsum('bgak,bkl->bgal', d, L))
            close = dd < _t
        else:
            close = d.abs().amax(-1) < 1e-3
        conflict = (close & keep[:, :g]).any(1)
        keep[:, g] = ok[:, g].unsqueeze(-1) & ~conflict
    mult = keep.sum(1).float()
    return mult * mask.float()

def symmetry_orbit(frac, mask, types, sg, R_lut, T_lut, ok_lut, n_ops=8,
                   stochastic=True, L=None, dedup=True, tol_ang=1e-2,
                   select='random'):
    """Differentiable partial orbit of the asymmetric unit under the space group.
    """
    B, A, _ = frac.shape
    sgi = sg.view(-1).clamp(1, 230).long() - 1
    R, T, ok = R_lut[sgi], T_lut[sgi], ok_lut[sgi]
    M = R.size(1)
    n = int(max(1, min(n_ops, M)))
    if n == 1:
        idx = torch.zeros(B, 1, dtype=torch.long, device=frac.device)
    elif select == 'nearest' and n < M:
        with torch.no_grad():
            fi = (torch.einsum('bmij,baj->bmai', R, frac)
                  + T.unsqueeze(2)) - frac.unsqueeze(1)
            fi = (fi + 0.5) % 1.0 - 0.5
            if L is not None:
                di = safe_norm(torch.einsum('bmak,bkl->bmal', fi, L))
            else:
                di = safe_norm(fi)
            di = di.masked_fill(~mask.bool().unsqueeze(1), 1e4)
            score = di.amin(-1)                                   # (B, M)
            score = score.masked_fill(~ok, 1e5)
            score[:, 0] = -1.0                                    # identity first
            idx = score.argsort(dim=1)[:, :n]
    elif stochastic and n < M:
        w = ok[:, 1:].float() + 1e-6
        pick = torch.multinomial(w, n - 1, replacement=False) + 1
        idx = torch.cat([torch.zeros(B, 1, dtype=torch.long, device=frac.device),
                         pick], 1)
    else:
        idx = torch.arange(n, device=frac.device).view(1, n).expand(B, n)
    Rn = torch.gather(R, 1, idx.view(B, n, 1, 1).expand(B, n, 3, 3))
    Tn = torch.gather(T, 1, idx.view(B, n, 1).expand(B, n, 3))
    okn = torch.gather(ok, 1, idx).float()
    f = (torch.einsum('bnij,baj->bnai', Rn, frac) + Tn.unsqueeze(2)) % 1.0
    m = mask.unsqueeze(1) * okn.unsqueeze(-1)          # (B, n, A)

    if dedup and n > 1:
        with torch.no_grad():
            keep = torch.ones(B, n, A, dtype=torch.bool, device=frac.device)
            for g in range(1, n):
                d = f[:, g:g + 1] - f[:, :g]                     # (B, g, A, 3)
                d = (d + 0.5) % 1.0 - 0.5
                if L is not None:
                    dd = safe_norm(torch.einsum('bgak,bkl->bgal', d, L))
                    close = dd < tol_ang
                else:
                    close = d.abs().amax(-1) < 1e-3
                keep[:, g] = ~(close & keep[:, :g]).any(1)
        m = m * keep.to(m.dtype)

    t = types.unsqueeze(1).expand(B, n, A)
    return f.reshape(B, n * A, 3), m.reshape(B, n * A), t.reshape(B, n * A)


class RBF(nn.Module):
    def __init__(self, num_rbf=16, cutoff=6.0):
        super().__init__()
        self.cutoff = cutoff
        self.register_buffer('centers', torch.linspace(0, cutoff, num_rbf))
        self.register_buffer('width', torch.tensor(cutoff / num_rbf))

    def forward(self, dist):
        d = dist.unsqueeze(-1).clamp(max=self.cutoff)
        rbf = torch.exp(-((d - self.centers) / self.width) ** 2)
        env = (0.5 * (torch.cos(math.pi * d / self.cutoff) + 1.0))
        return rbf * env

class BasicGNNConv(nn.Module):
    """Plain invariant message-passing layer (GCN/MPNN-style).

    Messages are built purely from scalar node features and the scalar edge
    descriptor (RBF-expanded bond length). No per-channel vectors are
    tracked, so this has no O(3) equivariance machinery at all -- it relies
    entirely on invariant inputs.
    """

    def __init__(self, channels, num_rbf=16, cutoff=6.0):
        super().__init__()

        self.C = channels
        self.rbf = RBF(num_rbf, cutoff)
        self.msg = nn.Sequential(
            nn.Linear(2 * channels + num_rbf, channels), nn.SiLU(),
            nn.Linear(channels, channels))
        self.upd_s = nn.Sequential(nn.Linear(2 * channels, channels), nn.SiLU())
        self.norm = nn.LayerNorm(channels)

    def forward(self, s, edges, mask):
        B, A, C = s.shape
        src, dst, dvec, dist, em = edges
        idx_s = src.unsqueeze(-1).expand(-1, -1, C)
        idx_d = dst.unsqueeze(-1).expand(-1, -1, C)
        s_i = torch.gather(s, 1, idx_s)
        s_j = torch.gather(s, 1, idx_d)
        rbf = self.rbf(dist)

        m = self.msg(torch.cat([s_i, s_j, rbf], -1))
        w = em.float().unsqueeze(-1)

        s_agg = torch.zeros_like(s).scatter_add_(1, idx_s, m * w)
        s = self.norm(s + self.upd_s(torch.cat([s, s_agg], -1))) * mask.unsqueeze(-1)
        return s

class EquivariantGNNConv(nn.Module):
    """PaiNN-style message passing with a per-node vector channel.

    Each node carries scalars s (B, A, C) and vectors v (B, A, C, 3). The vector
    channel is updated with the unit displacement, so it rotates with the crystal;
    every scalar read out of it is an invariant. `BasicGNNConv` keeps only
    distances, which are reflection-invariant, so a structure and its mirror image
    produce identical latents (measured: max |dtoken| = 0.0). Chirality is recovered
    by the pseudo-scalar in `CrystalGNNEncoder`'s read-out.

      phi  = MLP(s_j) -> three C-blocks;   W = Linear(rbf(r_ij)) -> three C-blocks
      ds_i += sum_j phi_s * W_s
      dv_i += sum_j (phi_vv * W_vv) v_j + (phi_vs * W_vs) u_ij
      a_vv, a_sv, a_ss = MLP([s, ||V v||])
      v <- v + a_vv (U v);   s <- LayerNorm(s + a_ss + a_sv <U v, V v>)
    """

    def __init__(self, channels, num_rbf=16, cutoff=6.0):
        super().__init__()
        self.C = channels
        self.rbf = RBF(num_rbf, cutoff)
        self.msg = nn.Sequential(nn.Linear(channels, channels), nn.SiLU(),
                                 nn.Linear(channels, 3 * channels))
        self.filt = nn.Linear(num_rbf, 3 * channels)
        self.U = nn.Linear(channels, channels, bias=False)
        self.V = nn.Linear(channels, channels, bias=False)
        self.upd = nn.Sequential(nn.Linear(2 * channels, channels), nn.SiLU(),
                                 nn.Linear(channels, 3 * channels))
        self.norm = nn.LayerNorm(channels)

    def forward(self, s, v, edges, mask):
        B, A, C = s.shape
        src, dst, dvec, dist, em = edges
        E = src.size(1)
        w = em.to(s.dtype).unsqueeze(-1)

       
        phi = self.msg(s)
        phi_j = torch.gather(phi, 1, dst.unsqueeze(-1).expand(-1, -1, 3 * C))
        Wf = self.filt(self.rbf(dist))
        x = phi_j * Wf
        xs, xvv, xvs = x.split(C, dim=-1)

        u = dvec / dist.clamp(min=1e-8).unsqueeze(-1)            # (B, E, 3)
        v_j = torch.gather(v.reshape(B, A, C * 3), 1,
                           dst.unsqueeze(-1).expand(-1, -1, C * 3)
                           ).reshape(B, E, C, 3)
        dm_v = (xvv.unsqueeze(-1) * v_j
                + xvs.unsqueeze(-1) * u.unsqueeze(-2))           # (B, E, C, 3)

        idx_s = src.unsqueeze(-1).expand(-1, -1, C)
        s_agg = torch.zeros_like(s).scatter_add_(1, idx_s, xs * w)
        v_agg = torch.zeros(B, A, C * 3, device=s.device, dtype=s.dtype
                            ).scatter_add_(
            1, src.unsqueeze(-1).expand(-1, -1, C * 3),
            (dm_v * w.unsqueeze(-1)).reshape(B, E, C * 3)).reshape(B, A, C, 3)
        s = s + s_agg
        v = v + v_agg

        
        vt = v.transpose(-1, -2)                                 # (B, A, 3, C)
        Uv = self.U(vt).transpose(-1, -2)
        Vv = self.V(vt).transpose(-1, -2)
        Vn = safe_norm(Vv, dim=-1)                               # (B, A, C)
        a_vv, a_sv, a_ss = self.upd(torch.cat([s, Vn], -1)).split(C, dim=-1)

        v = (v + a_vv.unsqueeze(-1) * Uv) * mask.unsqueeze(-1).unsqueeze(-1)
        s = self.norm(s + a_ss + a_sv * (Uv * Vv).sum(-1)) * mask.unsqueeze(-1)
        return s, v

class _SetBlock(nn.Module):
    """Pre-norm self-attention + FFN over a latent set.
    """

    def __init__(self, dim, heads=4, mult=2, dropout=0.0):
        super().__init__()
        heads = max(1, min(int(heads), dim))
        while dim % heads != 0 and heads > 1:
            heads -= 1
        self.n1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True,
                                          dropout=dropout)
        self.n2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * mult), nn.SiLU(),
                                 nn.Linear(dim * mult, dim))

    def forward(self, x):
        h = self.n1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.ffn(self.n2(x))

def canonical_site_order(types, frac, mask, n_sites):
    """Deterministic ordering of a crystal's sites; sort key (Z, x, y, z).
    """
    B, A = mask.shape
    dev = mask.device
   
    q = (torch.round(frac.double() * 1e4).long() % 10000)
    key = (types.long() * 10 ** 15
           + q[..., 0] * 10 ** 10 + q[..., 1] * 10 ** 5 + q[..., 2])
    PAD = torch.iinfo(torch.int64).max
    key = torch.where(mask.bool(), key, torch.full_like(key, PAD))
    order = key.argsort(dim=1, stable=True)
    n = int(min(int(n_sites), A))
    perm = order[:, :n]
    keep = torch.gather(mask.bool(), 1, perm)
    if n < int(n_sites):
        pad = int(n_sites) - n
        perm = torch.cat([perm, perm.new_zeros(B, pad)], 1)
        keep = torch.cat([keep, torch.zeros(B, pad, dtype=torch.bool, device=dev)], 1)
    return perm, keep

class FracFourier(nn.Module):
    """Periodic sin/cos features of an atom's own fractional coordinate.
    """

    def __init__(self, n_freq=6):
        super().__init__()
        self.n_freq = int(n_freq)
        self.register_buffer(
            'freqs', torch.arange(1, self.n_freq + 1).float(), persistent=False)

    @property
    def out_dim(self):
        return 6 * self.n_freq

    def forward(self, frac):
        ang = 2.0 * math.pi * (frac % 1.0).unsqueeze(-1) * self.freqs
        return torch.cat([ang.sin(), ang.cos()], -1).flatten(-2)

class CrystalGNNEncoder(nn.Module):
    """Invariant message-passing GNN -> a fixed-length set of SITE tokens.

   
    """

    def __init__(self, num_types, hidden=128, n_layers=4, out_dim=64,
                 cutoff=6.0, num_rbf=16, elem_feat_dim=0, n_sites=20,
                 n_attn_layers=2, heads=4, pos_freqs=6, use_pos=True,
                 conv_type='equivariant'):
        super().__init__()
        self.n_sites = int(n_sites)
        self.conv_type = str(conv_type).lower()
        self.embed = nn.Embedding(num_types, hidden)
        self.elem_feat_dim = int(elem_feat_dim)
        if self.elem_feat_dim > 0:
            self.elem_proj = nn.Linear(self.elem_feat_dim, hidden)

        self.use_pos = bool(use_pos) and int(pos_freqs) > 0
        if self.use_pos:
            self.pos_feat = FracFourier(pos_freqs)
            self.pos_proj = nn.Linear(self.pos_feat.out_dim, hidden)
            self.pos_skip = nn.Linear(self.pos_feat.out_dim, hidden)

        _equi = self.conv_type == 'equivariant'
        self.convs = nn.ModuleList([
            (EquivariantGNNConv(hidden, num_rbf, cutoff) if _equi
             else BasicGNNConv(hidden, num_rbf, cutoff))
            for _ in range(int(n_layers))])

        self.chir = nn.Linear(hidden, 3, bias=False) if _equi else None
        _read = (2 * hidden + 1) if _equi else hidden
        self.site_proj = nn.Sequential(
            nn.Linear(_read, out_dim), nn.SiLU(), nn.Linear(out_dim, out_dim))
        self.empty_emb = nn.Parameter(torch.randn(self.n_sites, out_dim) * 0.02)
        self.pos_emb = nn.Parameter(torch.randn(self.n_sites, out_dim) * 0.02)
        self.blocks = nn.ModuleList([
            _SetBlock(out_dim, heads) for _ in range(int(n_attn_layers))])
        self.out_norm = nn.LayerNorm(out_dim)

    def forward(self, atom_types, frac, L, mask, src, dst, offset, emask,
                elem_feat=None, type_probs=None, site_mask=None,
                order_frac=None, canonical=True):
        
        if type_probs is not None:
            s = type_probs @ self.embed.weight
            if self.elem_feat_dim > 0 and elem_feat is not None:
                s = s + self.elem_proj(type_probs @ elem_feat.to(s.dtype))
            hard_types = type_probs.argmax(-1)
        else:
            s = self.embed(atom_types)
            if self.elem_feat_dim > 0 and elem_feat is not None:
                s = s + self.elem_proj(elem_feat.to(s.dtype)[atom_types])
            hard_types = atom_types
        pos = self.pos_feat(frac) if self.use_pos else None
        if pos is not None:
            s = s + self.pos_proj(pos)
        s = s * mask.unsqueeze(-1)

        dvec, dist = edge_geometry(frac, L, src, dst, offset)
        edges = (src, dst, dvec, dist, emask)
        if self.conv_type == 'equivariant':
            vvec = torch.zeros(s.size(0), s.size(1), s.size(2), 3,
                               device=s.device, dtype=s.dtype)
            for conv in self.convs:
                s, vvec = conv(s, vvec, edges, mask)
        else:
            vvec = None
            for conv in self.convs:
                s = conv(s, edges, mask)

        mb = mask.bool()
        sm = mb if site_mask is None else (mb & site_mask.bool())
        B, A = sm.shape
        if canonical:
            perm, keep = canonical_site_order(
                hard_types, frac if order_frac is None else order_frac,
                sm, self.n_sites)
        else:
            n = int(min(self.n_sites, A))
            perm = torch.arange(n, device=sm.device).view(1, n).expand(B, n)
            keep = sm[:, :n]
            if n < self.n_sites:
                pad = self.n_sites - n
                perm = torch.cat([perm, perm.new_zeros(B, pad)], 1)
                keep = torch.cat([keep, torch.zeros(B, pad, dtype=torch.bool,
                                                    device=sm.device)], 1)

        s_in = s if pos is None else s + self.pos_skip(pos)
        if vvec is not None:
            v_norm = safe_norm(vvec, dim=-1)                      # (B, A, H)
            wch = self.chir(vvec.transpose(-1, -2)).transpose(-1, -2)

            wch = wch / safe_norm(wch, dim=-1, keepdim=True).clamp(min=1e-6)
            chi = torch.linalg.det(wch).unsqueeze(-1)             # (B, A, 1)
            s_in = torch.cat([s_in, v_norm, chi], -1)
        h_site = self.site_proj(s_in)
        C = h_site.size(-1)
        tok = torch.gather(h_site, 1, perm.unsqueeze(-1).expand(-1, -1, C))
        tok = torch.where(keep.unsqueeze(-1), tok,
                          self.empty_emb.unsqueeze(0).to(tok.dtype))
        tok = tok + self.pos_emb.unsqueeze(0).to(tok.dtype)
        for blk in self.blocks:
            tok = blk(tok)
        return self.out_norm(tok), keep

class LatentStandardizer(nn.Module):
    """Whitens the latent with RUNNING statistics in both modes.
    """

    def __init__(self, dim, momentum=0.01, eps=1e-5):
        super().__init__()
        self.eps, self.momentum = eps, momentum
        self.register_buffer('mean', torch.zeros(dim))
        self.register_buffer('var', torch.ones(dim))
        self.register_buffer('initialized', torch.tensor(False))
        # Not a buffer: it must not enter the state_dict, and it is a counter
        # for the trainer to read, not state the model depends on.
        self._rejected = 0

    def forward(self, z):
        if self.training:
            self._update(z)
        return (z - self.mean) / (self.var + self.eps).sqrt()

    @torch.no_grad()
    def _update(self, z):
        
        flat = z.detach().reshape(-1, z.size(-1))
        if flat.numel() == 0:
            return
        finite_rows = torch.isfinite(flat).all(-1)
        n_ok = int(finite_rows.sum())
        if n_ok == 0:
            self._rejected += 1
            return
        if n_ok < flat.size(0):
            self._rejected += 1
            flat = flat[finite_rows]
        flat = flat.double()
        m = flat.mean(0).to(self.mean.dtype)
        v = flat.var(0, unbiased=False).clamp(min=self.eps).to(self.var.dtype)
        if not (torch.isfinite(m).all() and torch.isfinite(v).all()):
            self._rejected += 1
            return
        if (not bool(self.initialized)) or not self.stats_are_finite():
            self.mean.copy_(m)
            self.var.copy_(v)
            self.initialized.fill_(True)
        else:
            self.mean.mul_(1 - self.momentum).add_(self.momentum * m)
            self.var.mul_(1 - self.momentum).add_(self.momentum * v)

    def stats_are_finite(self):
        return bool(torch.isfinite(self.mean).all()
                    and torch.isfinite(self.var).all())

    @torch.no_grad()
    def reset_running_stats(self):
        self.mean.zero_()
        self.var.fill_(1.0)
        self.initialized.fill_(False)

class FlowBlock(nn.Module):
    def __init__(self, dim, hidden, cond_dim, t_dim=64, heads=4):
        super().__init__()
        self.t_dim = t_dim
        self.time = nn.Sequential(nn.Linear(t_dim, hidden), nn.SiLU(), nn.Linear(hidden, dim))
        self.radius = nn.Linear(1, dim)
        nn.init.zeros_(self.radius.weight); nn.init.zeros_(self.radius.bias)
        self.cond_proj = nn.Linear(cond_dim, dim)
        nn.init.zeros_(self.cond_proj.weight); nn.init.zeros_(self.cond_proj.bias)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.n1, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU())
        self.head = nn.Linear(hidden, dim)

    def _temb(self, t):
        half = self.t_dim // 2
        f = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device).float() / max(half - 1, 1))
        a = (t.view(-1, 1) * 1000.0) * f.view(1, -1)
        return torch.cat([torch.sin(a), torch.cos(a)], -1)

    def forward(self, z, t, cond, key_padding_mask=None):
        r = torch.log1p(safe_norm(z, keepdim=True))
        h = (z + self.time(self._temb(t)).unsqueeze(1)
             + self.cond_proj(cond).unsqueeze(1) + self.radius(r))
        hn = self.n1(h)
        h = h + self.attn(hn, hn, hn, key_padding_mask=key_padding_mask,
                          need_weights=False)[0]
        return self.head(self.net(self.n2(h)))

class LatentSetFlow(nn.Module):
    """Velocity field over the latent SITE set. This is the generator.
    """

    def __init__(self, dim, hidden=256, n_layers=4, cond_dim=128, t_dim=64,
                 heads=4, n_tokens=0):
        super().__init__()
        self.blocks = nn.ModuleList([FlowBlock(dim, hidden, cond_dim, t_dim, heads)
                                     for _ in range(n_layers)])
        self.n_tokens = int(n_tokens)
        if self.n_tokens > 0:
            self.token_emb = nn.Parameter(torch.randn(self.n_tokens, dim) * 0.02)

    def forward(self, z, t, cond, key_padding_mask=None):
        v = torch.zeros_like(z)
        h = z
        if self.n_tokens > 0 and z.size(1) == self.n_tokens:
            h = h + self.token_emb.unsqueeze(0)
        for blk in self.blocks:
            dv = blk(h, t, cond, key_padding_mask=key_padding_mask)
            v = v + dv
            h = h + dv
        return v

class PropertyConditioner(nn.Module):
    """Conditioning vector for the flow: property y AND space group.

    y and sg are dropped independently for classifier-free guidance, so the model
    learns p(z), p(z|y), p(z|sg) and p(z|y, sg).
    """

    def __init__(self, prop_dim=1, cond_dim=128, hidden=128, n_sg=230):
        super().__init__()
        self.cond_dim = cond_dim
        self.enc = nn.Sequential(nn.Linear(prop_dim, hidden), nn.SiLU(),
                                 nn.Linear(hidden, cond_dim))
        self.null = nn.Parameter(torch.zeros(cond_dim))
        self.sg_emb = nn.Embedding(n_sg, cond_dim)
        nn.init.normal_(self.sg_emb.weight, std=0.02)
        self.sg_null = nn.Parameter(torch.zeros(cond_dim))
        self.mix = nn.Sequential(nn.LayerNorm(cond_dim),
                                 nn.Linear(cond_dim, cond_dim), nn.SiLU(),
                                 nn.Linear(cond_dim, cond_dim))

    def forward(self, y, B, device, drop_prob=0.0, training=None, sg=None,
                sg_drop_prob=None):
        if training is None:
            training = self.training
        if sg_drop_prob is None:
            sg_drop_prob = drop_prob

        if y is None:
            c = self.null.view(1, -1).expand(B, -1)
        else:
            if y.dim() == 1:
                y = y.unsqueeze(-1)
            c = self.enc(y)
            if drop_prob > 0 and training:
                d = torch.rand(B, 1, device=device) < drop_prob
                c = torch.where(d, self.null.view(1, -1).to(c.dtype), c)

        if sg is None:
            s = self.sg_null.view(1, -1).expand(B, -1)
        else:
            sg_i = torch.as_tensor(sg, device=device).view(-1).long().clamp(1, 230) - 1
            if sg_i.numel() == 1 and B > 1:
                sg_i = sg_i.expand(B)
            s = self.sg_emb(sg_i)
            if sg_drop_prob > 0 and training:
                d = torch.rand(B, 1, device=device) < sg_drop_prob
                s = torch.where(d, self.sg_null.view(1, -1).to(s.dtype), s)
        return self.mix(c + s)

@torch.no_grad()
def assign_unique_wyckoff(wk_logits, valid, nfree, occ_mask, prior=None,
                          prior_weight=0.0):
    """Pick one Wyckoff class per token, with fixed-point classes used ONCE.
    """
    B, M, W = wk_logits.shape
    sc = wk_logits.float()
    if prior is not None and prior_weight > 0:
        sc = sc + float(prior_weight) * torch.log(
            prior.float().clamp(min=1e-6)).unsqueeze(1)
    sc = sc.masked_fill(~valid.unsqueeze(1), float('-inf'))
    order = sc.argsort(dim=-1, descending=True).cpu().numpy()      # (B,M,W)
    conf = sc.amax(-1)
    tok_rank = conf.masked_fill(~occ_mask.bool(), float('-inf'))                    .argsort(dim=-1, descending=True).cpu().numpy()
    okv = valid.cpu().numpy()
    nf = nfree.cpu().numpy()
    occ = occ_mask.bool().cpu().numpy()
    out = sc.argmax(-1).cpu().numpy()
    for b in range(B):
        taken = set()
        general = int(nf[b].argmax()) if okv[b].any() else 0
        for t in tok_rank[b]:
            if not occ[b, t]:
                continue
            chosen = None
            for w in order[b, t]:
                w = int(w)
                if not okv[b, w]:
                    continue
                if nf[b, w] <= 0 and w in taken:
                    continue
                chosen = w
                break
            if chosen is None:
                chosen = general
            if nf[b, chosen] <= 0:
                taken.add(chosen)
            out[b, t] = chosen
    return torch.as_tensor(out, dtype=torch.long, device=wk_logits.device)


class SiteDecoder(nn.Module):
    """Latent site tokens -> asymmetric unit: count, Wyckoff site, position, element.

    """

    def __init__(self, dim, num_types, n_sites=20, hidden=256, dropout=0.1,
                 use_wyckoff=True, gen_jitter=0.0, wyckoff_warmup=3000,
                 n_attn_layers=2, heads=4, count_sharpness=4.0,
                 unique_wyckoff=True, wyckoff_prior_weight=0.0):
        super().__init__()
        self.unique_wyckoff = bool(unique_wyckoff)
        self.wyckoff_prior_weight = float(wyckoff_prior_weight)
        self.n_sites = int(n_sites)
        self.num_types = int(num_types)
        self.use_wyckoff = bool(use_wyckoff)
        self.gen_jitter = float(gen_jitter)
        self.count_sharpness = float(count_sharpness)

        self.feat = nn.Sequential(nn.Linear(dim, hidden), nn.SiLU(),
                                  nn.LayerNorm(hidden), nn.Dropout(dropout),
                                  nn.Linear(hidden, hidden), nn.SiLU())
        self.blocks = nn.ModuleList([_SetBlock(hidden, heads, dropout=dropout)
                                     for _ in range(int(n_attn_layers))])
        self.norm = nn.LayerNorm(hidden)

        self.count = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.SiLU(),
                                   nn.Linear(hidden, 1))

        self.anchor = nn.Linear(hidden, 3)
        self.type_head = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(),
                                       nn.Linear(hidden, num_types))

        self.max_wyckoff = 27
        self.wyckoff_head = nn.Linear(hidden, self.max_wyckoff)
        self.wyckoff_warmup = int(wyckoff_warmup)
        self.register_buffer('_wyck_step', torch.zeros((), dtype=torch.long))
        self._wyck_lambda = 0.0 if self.wyckoff_warmup > 0 else 1.0
        wk_proj, wk_off, wk_nfree, wk_valid, wk_gen = _get_wyckoff_tensors(self.max_wyckoff)

        self.register_buffer('wyckoff_proj_lut', wk_proj, persistent=True)
        self.register_buffer('wyckoff_off_lut', wk_off, persistent=True)
        self.register_buffer('wyckoff_n_free', wk_nfree, persistent=True)
        self.register_buffer('wyckoff_valid', wk_valid, persistent=True)
        self.register_buffer('wyckoff_general', wk_gen, persistent=True)
        self.register_buffer('_wyckoff_from_data', torch.tensor(False),
                             persistent=True)
        # Exact multiplicity of each class, and how often the data uses it.
        # Both are tables, not predictions -- see wyckoff_multiplicity_table.
        self.register_buffer('wyckoff_mult_lut',
                             wyckoff_multiplicity_table(wk_proj, wk_off, wk_valid),
                             persistent=True)
        _pr = wk_valid.float()
        self.register_buffer('wyckoff_prior_lut',
                             _pr / _pr.sum(1, keepdim=True).clamp(min=1.0),
                             persistent=True)

    def install_wyckoff_codebook(self, datasets):
        """Replace the enumerated Wyckoff tables with ones built from the data."""
        Pl, Ol, Nf, Vl, gen, Cn = build_wyckoff_codebook(datasets, self.max_wyckoff)
        dev = self.wyckoff_proj_lut.device
        self.wyckoff_proj_lut.copy_(Pl.to(dev))
        self.wyckoff_off_lut.copy_(Ol.to(dev))
        self.wyckoff_n_free.copy_(Nf.to(dev, self.wyckoff_n_free.dtype))
        self.wyckoff_valid.copy_(Vl.to(dev))
        self.wyckoff_general.copy_(gen.to(dev, self.wyckoff_general.dtype))
        self._wyckoff_from_data.fill_(True)
        # The codebook changed, so the derived tables must be rebuilt against
        # it: multiplicity is a function of (P, o) and the group, and the class
        # prior is only meaningful once the classes are the mined ones.
        self.wyckoff_mult_lut.copy_(
            wyckoff_multiplicity_table(Pl, Ol, Vl).to(dev))
        _pr = Cn.to(dev).clamp(min=0.0)
        _pr = torch.where(Vl.to(dev), _pr + 1e-3, torch.zeros_like(_pr))
        self.wyckoff_prior_lut.copy_(_pr / _pr.sum(1, keepdim=True).clamp(min=1e-6))
        # Coverage against the annotations themselves. Every (group, letter) the
        # data uses must be in the codebook, or `wyckoff_class_ce` will drop
        # those sites and the Wyckoff head trains on a biased subset.
        try:
            seen, miss = 0, 0
            for ds in datasets:
                sgs = getattr(ds, 'space_groups', None)
                for gi, g in enumerate(getattr(ds, 'graph_data', [])):
                    sg0 = int(sgs[gi]) if sgs is not None else 1
                    sg0 = min(max(sg0, 1), 230) - 1
                    wi = np.asarray(getattr(g, 'wyckoff_indices', [])).reshape(-1)
                    am = np.asarray(getattr(g, 'asym_unit_mask', []), dtype=bool).reshape(-1)
                    if wi.size != am.size:
                        continue
                    for w_ in wi[am]:
                        w_ = int(w_)
                        seen += 1
                        if not (0 <= w_ < self.max_wyckoff and bool(Vl[sg0, w_])):
                            miss += 1
            if seen:
                print(f"  [wyckoff] codebook covers "
                      f"{100.0 * (seen - miss) / seen:.1f}% of annotated "
                      f"asymmetric-unit sites ({miss} unrepresentable).")
        except Exception:
            pass

    def advance_wyckoff(self, n=1):
        """Advance the warmup counter once per OPTIMISER step (the decoder runs
        more than once per step: generated + reconstruction)."""
        if self.wyckoff_warmup > 0:
            self._wyck_step += int(n)
            self._wyck_lambda = min(
                1.0, float(self._wyck_step.item()) / float(self.wyckoff_warmup))

    def _wyckoff_lambda(self):
        """0 -> 1 ramp on the Wyckoff constraint. At 0 the anchor is a free
        general position, at 1 it is projected onto the selected site.

        Read from a cached float. Calling .item() on the step buffer inside
        forward() forces a device->host sync on every decoder pass."""
        if self.wyckoff_warmup <= 0:
            return 1.0
        return self._wyck_lambda

    def forward(self, z, generation=False, temperature=0.0,
                formability_mask=None, sg=None, lattice=None,
                wyckoff_teacher=None):
       
        B, M, _ = z.shape
        h = self.feat(z)
        for blk in self.blocks:
            h = blk(h)
        h = self.norm(h)

        # --- how many sites -------------------------------------------------
        pooled = torch.cat([h.mean(1), h.amax(1)], -1)
        n_soft = 1.0 + (M - 1) * torch.sigmoid(self.count(pooled))        # (B, 1)
        idx = torch.arange(M, device=z.device, dtype=h.dtype).view(1, M)
        occ_soft = torch.sigmoid((n_soft - idx - 0.5) * self.count_sharpness)
        n_round = n_soft.detach().round().clamp(1.0, float(M))
        occ_hard = (idx < n_round).to(h.dtype)
        # straight-through: the forward value is the cell that is actually
        # emitted, the backward pass still moves the count head.
        occ_st = occ_soft + (occ_hard - occ_soft).detach()
        valid = occ_hard

        # --- where ----------------------------------------------------------
        anchor = self.anchor(h) % 1.0
        eye3 = torch.eye(3, device=h.device, dtype=anchor.dtype)
        site_P = eye3.view(1, 1, 3, 3).expand(B, M, 3, 3)
        site_o = torch.zeros(B, M, 3, device=h.device, dtype=anchor.dtype)
        site_P_use = site_P
        wk_probs = None
        wk_logits = None
        wk_idx = None
        wk_mult = None
        wk_nfree = None
        if sg is not None and self.use_wyckoff:
            sg_idx = torch.as_tensor(sg, device=h.device).detach()
            sg_idx = sg_idx.to(torch.long).view(-1).clamp(1, 230) - 1
            P_sg = self.wyckoff_proj_lut[sg_idx]
            o_sg = self.wyckoff_off_lut[sg_idx]
            ok_sg = self.wyckoff_valid[sg_idx]

            wk_logits = self.wyckoff_head(h)

            wk_logits = wk_logits.masked_fill(~ok_sg.unsqueeze(1), -1e4)
            wk_probs = F.softmax(wk_logits, dim=-1)
            if generation and self.unique_wyckoff:
                # Exclusive assignment of the fixed-point classes. Two tokens
                # cannot be handed the same rank-0 Wyckoff site, which is what
                # puts two atoms at one coordinate (see assign_unique_wyckoff).
                wk_idx = assign_unique_wyckoff(
                    wk_logits, ok_sg, self.wyckoff_n_free[sg_idx], valid,
                    prior=self.wyckoff_prior_lut[sg_idx],
                    prior_weight=self.wyckoff_prior_weight)
            else:
                wk_idx = wk_logits.argmax(-1)
            wk_hard = F.one_hot(wk_idx, wk_logits.size(-1)).to(wk_probs.dtype)
            w_st = wk_probs + (wk_hard - wk_probs).detach()
            P = torch.einsum('bnw,bwij->bnij', w_st, P_sg)
            o = torch.einsum('bnw,bwj->bnj', w_st, o_sg)
            if wyckoff_teacher is not None:
                P_geo = wyckoff_teacher[0].to(P.dtype)
                o_geo = wyckoff_teacher[1].to(o.dtype)
            else:
                P_geo, o_geo = P, o

         
            lam = 1.0 if generation else self._wyckoff_lambda()
            if lam < 1.0:
                P_use = (1.0 - lam) * eye3.view(1, 1, 3, 3) + lam * P_geo
                o_use = lam * o_geo
            else:
                P_use, o_use = P_geo, o_geo

            rel = (anchor - o_use + 0.5) % 1.0 - 0.5
            anchor = (torch.einsum('bnij,bnj->bni', P_use, rel) + o_use) % 1.0
            site_P, site_o, site_P_use = P, o, P_use
            wk_mult = self.wyckoff_mult_lut[sg_idx].gather(1, wk_idx)
            wk_nfree = self.wyckoff_n_free[sg_idx].gather(1, wk_idx).to(anchor.dtype)

        if generation and self.gen_jitter > 0:
            anchor = (anchor + (torch.rand_like(anchor) - 0.5) * self.gen_jitter) % 1.0
        frac = anchor

        type_logits = self.type_head(h)
        if formability_mask is not None:
            type_logits = type_logits + formability_mask.view(1, 1, -1).to(type_logits.dtype)
        probs = F.softmax(type_logits, -1)

        if generation and temperature > 0:
            scaled = type_logits / temperature
            top_k = min(10, self.num_types)
            topv, topi = scaled.topk(top_k, dim=-1)
            fill = torch.full_like(scaled, -1e9)
            fill.scatter_(-1, topi, topv)
            flat = F.softmax(fill, -1).reshape(-1, self.num_types)
            samp = torch.multinomial(flat.clamp(min=1e-8), 1).view(B, M)
        else:
            samp = probs.argmax(-1)


        probs_hard = F.one_hot(samp, self.num_types).to(probs.dtype)
        probs_st = probs + (probs_hard - probs).detach()

        return {'frac': frac, 'type_logits': type_logits, 'type_probs': probs,
                'type_probs_st': probs_st, 'mask': valid,
                'sampled_types': samp, 'occ_soft': occ_st,
                'site_proj': site_P_use, 'anchor': anchor,
                'site_P': site_P, 'site_o': site_o,
                'n_sites_soft': n_soft.squeeze(-1),
                'wyckoff_probs': wk_probs, 'wyckoff_logits': wk_logits,
                'wyckoff_index': wk_idx, 'wyckoff_mult': wk_mult,
                'wyckoff_nfree': wk_nfree}

class SitePropertyPredictor(nn.Module):
    """Deep-Sets read-out: per-site contribution, pooled with occupancy weights."""

    def __init__(self, dim, hidden=128, extensive=False):
        super().__init__()
        self.extensive = extensive
        self.phi = nn.Sequential(nn.Linear(dim, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, 1))

    def forward(self, caps, weights, mask=None):
        e = self.phi(caps)
        w = weights.clamp(min=0)
        if mask is not None:
            w = w * mask.to(w.dtype)
        w = w.unsqueeze(-1)
        total = (w * e).sum(1)
        return (total if self.extensive else total / w.sum(1).clamp(min=1e-6)), e

class LatticeHead(nn.Module):
    def __init__(self, dim, hidden=128, angle_lo=20.0, angle_hi=160.0):
        super().__init__()
        self.angle_lo, self.angle_hi = angle_lo, angle_hi
        self.vol = nn.Sequential(nn.Linear(dim, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        self.shape = nn.Sequential(nn.Linear(dim, hidden), nn.SiLU(),
                                   nn.Linear(hidden, hidden), nn.SiLU(),
                                   nn.Linear(hidden, 6))

    def forward(self, caps, weights, mask=None):
        w = weights.clamp(min=0)
        if mask is not None:
            w = w * mask.to(w.dtype)
        w = w.unsqueeze(-1)
        v_c = F.softplus(self.vol(caps)) + 1e-2
        V = (w * v_c).sum(1).clamp(min=1e-3)
        pooled = (caps * w).sum(1) / w.sum(1).clamp(min=1e-6)
        raw = self.shape(pooled)
        abc = F.softplus(raw[:, :3]) + 0.5
        ang = self.angle_lo + (self.angle_hi - self.angle_lo) * torch.sigmoid(raw[:, 3:])
        S = lattice_params_to_matrix(torch.cat([abc, ang], -1))
        V_S = torch.linalg.det(S).abs().clamp(min=1e-3)
        scale = (V.squeeze(-1) / V_S).clamp(min=1e-6).pow(1 / 3)
        L = S * scale.view(-1, 1, 1)
        return L, V, v_c

class SpaceGroupPredictor(nn.Module):
    def __init__(self, dim, hidden=128, pos_dim=0):
        super().__init__()
        self.attn = nn.Linear(dim, 1)
        self.pos_dim = pos_dim
        in_dim = dim + (pos_dim if pos_dim > 0 else 0)
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, 230))

    def forward(self, caps, mask=None, pos_feat=None):
        a = self.attn(caps)
        if mask is not None:
            a = a.masked_fill(~mask.unsqueeze(-1), -1e9)
        w = F.softmax(a, dim=1)
        pooled = (w * caps).sum(1)
        if self.pos_dim > 0:
            if pos_feat is None:
                pos_feat = torch.zeros(pooled.size(0), self.pos_dim,
                                       device=pooled.device, dtype=pooled.dtype)
            pooled = torch.cat([pooled, pos_feat.view(pooled.size(0), -1)], -1)
        return self.net(pooled)

class SiteMultiplicityHead(nn.Module):
    """How many FULL-CELL atoms site i stands for, i.e. its Wyckoff multiplicity.
    """

    def __init__(self, dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.SiLU(),
                                 nn.Linear(hidden, 1))

    def forward(self, z):
        return 1.0 + F.softplus(self.net(z).squeeze(-1))

class CoordinateRefiner(nn.Module):
    """Local relaxation of the decoded coordinates."""

    def __init__(self, num_types, dim=64, num_rbf=16, cutoff=6.0, steps=2,
                 scale=0.3):
        super().__init__()
        self.steps, self.scale, self.cutoff = steps, float(scale), cutoff
        self.rbf = RBF(num_rbf, cutoff)
        self.embed = nn.Linear(num_types, dim)
        # The first message layer is split into three factors so the
        # (B, M, M, 2*dim + num_rbf) concatenation is never materialised; only
        # its (B, M, M, dim) output is.
        self.msg_i = nn.Linear(dim, dim)
        self.msg_j = nn.Linear(dim, dim, bias=False)
        self.msg_r = nn.Linear(num_rbf, dim, bias=False)
        self.msg_o = nn.Linear(dim, dim)
        self.upd = nn.Sequential(nn.Linear(2 * dim, dim), nn.SiLU())
        self.norm = nn.LayerNorm(dim)
        self.delta = nn.Linear(dim, 3)
        nn.init.zeros_(self.delta.weight); nn.init.zeros_(self.delta.bias)

    def forward(self, frac, type_probs, mask, L, site_proj=None):
        B, M, _ = frac.shape
        mb = mask.bool()
        h = self.embed(type_probs) * mask.unsqueeze(-1)
        eye = torch.eye(M, device=frac.device, dtype=torch.bool).unsqueeze(0)
        invL = torch.linalg.inv(L.to(frac.dtype))
        for _ in range(self.steps):
            _, dist = pairwise_pbc(frac, L)
            pair = (mb.unsqueeze(2) & mb.unsqueeze(1)) & ~eye
            within = (pair & (dist < self.cutoff) & (dist > 1e-6)).float().unsqueeze(-1)
            msg = self.msg_o(F.silu(self.msg_i(h).unsqueeze(2)
                                    + self.msg_j(h).unsqueeze(1)
                                    + self.msg_r(self.rbf(dist))))
            agg = (msg * within).sum(2) / within.sum(2).clamp(min=1.0)
            h = self.norm(h + self.upd(torch.cat([h, agg], -1))) * mask.unsqueeze(-1)
            d_cart = self.scale * torch.tanh(self.delta(h))
            d = torch.einsum('baj,bji->bai', d_cart, invL)
            if site_proj is not None:
                d = torch.einsum('baij,baj->bai', site_proj.to(d.dtype), d)
            frac = (frac + d * mask.unsqueeze(-1)) % 1.0
        return frac

def _flow_mse(v_pred, v_tgt, keep, occ_weight=3.0):
    """Flow-matching MSE with occupied tokens weighted up.
    """
    err = (v_pred - v_tgt).pow(2).mean(-1)                     # (B, M)
    if keep is None or float(occ_weight) == 1.0:
        return err.mean()
    w = 1.0 + (float(occ_weight) - 1.0) * keep.to(err.dtype)
    return (err * w).sum() / w.sum().clamp(min=1.0)

def _rollout_step(model, z, tt, dt, solver, vel):
    if solver == 'euler':
        return z + dt * vel(z, tt)
    if solver == 'rk4':
        k1 = vel(z, tt)
        k2 = vel(z + 0.5 * dt * k1, tt + 0.5 * dt)
        k3 = vel(z + 0.5 * dt * k2, tt + 0.5 * dt)
        k4 = vel(z + dt * k3, tt + dt)
        return z + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    k1 = vel(z, tt)
    k2 = vel(z + 0.5 * dt * k1, tt + 0.5 * dt)
    return z + dt * k2



class DirectCrystalFlow(nn.Module):
    """Fully flow-based crystal generator.

    Architecture:
      z0 ~ N(0,1)  ->  LatentSetFlow(z0, t, cond)  ->  z1 (site tokens)
      z1  ->  SiteDecoder  ->  asymmetric unit  ->  cell  ->  crystal

    No encoder is used. The flow model learns to generate latents directly.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

      
        self.encoder = CrystalGNNEncoder(
            cfg.num_types, hidden=cfg.enc_hidden, n_layers=cfg.enc_layers,
            out_dim=cfg.sec_dim, cutoff=cfg.cutoff, num_rbf=cfg.num_rbf,
            elem_feat_dim=cfg.elem_feat_dim, n_sites=cfg.n_sites,
            n_attn_layers=int(getattr(cfg, 'enc_attn_layers', 2)),
            pos_freqs=int(getattr(cfg, 'pos_freqs', 6)),
            use_pos=bool(getattr(cfg, 'use_pos_features', False)),
            conv_type=str(getattr(cfg, 'conv_type', 'equivariant')))
        self.enc_proj = nn.Sequential(
            nn.Linear(cfg.sec_dim, cfg.sec_dim), nn.SiLU(),
            nn.Linear(cfg.sec_dim, cfg.sec_dim))
        self.enc_standardizer = LatentStandardizer(cfg.sec_dim)

        self.register_buffer('_encoder_pretrained', torch.tensor(False))

        # Core flow model - this IS the generator
        self.flow = LatentSetFlow(
            cfg.sec_dim, cfg.flow_hidden, cfg.flow_layers, cfg.cond_dim,
            cfg.t_dim, n_tokens=cfg.n_sites)

        # Property + space-group conditioning
        self.conditioner = PropertyConditioner(1, cfg.cond_dim)

        # Decoder: latent site tokens -> crystal structure
        self.decoder = SiteDecoder(
            cfg.sec_dim, cfg.num_types, n_sites=cfg.n_sites,
            dropout=cfg.decoder_dropout,
            use_wyckoff=getattr(cfg, 'use_wyckoff', True),
            gen_jitter=getattr(cfg, 'gen_anchor_jitter', 0.0),
            wyckoff_warmup=int(getattr(cfg, 'wyckoff_warmup', 3000)),
            n_attn_layers=int(getattr(cfg, 'dec_attn_layers', 2)),
            count_sharpness=float(getattr(cfg, 'count_sharpness', 4.0)),
            unique_wyckoff=bool(getattr(cfg, 'unique_wyckoff', True)),
            wyckoff_prior_weight=float(getattr(cfg, 'wyckoff_prior_weight', 0.0)))

        # Auxiliary heads
        self.property = SitePropertyPredictor(cfg.sec_dim, extensive=False)
        self.lattice = LatticeHead(cfg.sec_dim)
        self.sg = SpaceGroupPredictor(cfg.sec_dim, pos_dim=32)
        self.sg_coarse = SpaceGroupPredictor(cfg.sec_dim, pos_dim=0)
        self.mult_head = SiteMultiplicityHead(cfg.sec_dim)
        self.sg_pos_encoder = nn.Sequential(
            nn.Linear(8, 64), nn.SiLU(), nn.Linear(64, 32))
        self.refiner = (CoordinateRefiner(cfg.num_types, dim=cfg.sec_dim,
                                          num_rbf=cfg.num_rbf, cutoff=cfg.cutoff,
                                          steps=cfg.refiner_steps, scale=cfg.refiner_scale)
                        if cfg.use_refiner else None)

        import copy as _copy
        self.property_frozen = _copy.deepcopy(self.property)
        for _p in self.property_frozen.parameters():
            _p.requires_grad = False

        self.register_buffer('y_mean', torch.tensor(0.0))
        self.register_buffer('y_std', torch.tensor(1.0))
        self.register_buffer('sg_class_weights', torch.ones(230))
      
        self.register_buffer('sg_prior', torch.ones(230) / 230.0)
        self.register_buffer('_stage3_step', torch.zeros((), dtype=torch.long))
        # Element physics features, indexed by vocabulary position. Previously
        # computed in main() and never passed to the encoder.
        if cfg.elem_feat_dim > 0 and getattr(cfg, 'elem_feat_table', None) is not None:
            _ef = torch.as_tensor(cfg.elem_feat_table, dtype=torch.float32)
            assert _ef.shape == (cfg.num_types, cfg.elem_feat_dim), (
                f"elem_feat_table {tuple(_ef.shape)} != "
                f"({cfg.num_types}, {cfg.elem_feat_dim})")
        else:
            _ef = torch.zeros(cfg.num_types, max(1, cfg.elem_feat_dim))
        self.register_buffer('elem_feat', _ef, persistent=True)

        # Element properties
        zs = cfg.atomic_numbers if cfg.atomic_numbers is not None else list(range(1, cfg.num_types + 1))
        assert len(zs) == cfg.num_types, "atomic_numbers length must equal num_types"
        radii = torch.tensor([ATOMIC_RADII.get(int(z), 1.0) for z in zs])
        _oxl = [smact_oxidation_states(z) for z in zs]
        ox = torch.tensor([float(o[0]) for o in _oxl])
        ox_min = torch.tensor([float(min(o)) for o in _oxl])
        ox_max = torch.tensor([float(max(o)) for o in _oxl])
        self.register_buffer('radii_lut', radii.float(), persistent=False)
        self.register_buffer('ox_lut', ox.float(), persistent=False)
        self.register_buffer('ox_min_lut', ox_min.float(), persistent=False)
        self.register_buffer('ox_max_lut', ox_max.float(), persistent=False)

        _miss = [int(z) for z in zs if int(z) not in SMACT_OXIDATION_STATES]
        if _miss:
            _data_warn(
                f"{len(_miss)} vocabulary elements are absent from SMACT's "
                f"oxidation-state table (first: Z={_miss[:5]}); they fall back "
                f"to the legacy table or to [0]. Charge balance treats a [0] "
                f"element as always balanced.")
        ox_pad, ox_msk, en_lut, metal_lut = build_chemistry_tables(zs)
        self.register_buffer('ox_states_padded', ox_pad, persistent=False)
        self.register_buffer('ox_states_mask', ox_msk, persistent=False)
       
        self.register_buffer('eneg_lut', en_lut, persistent=False)
        self.register_buffer('metal_lut', metal_lut, persistent=False)
        print(f"  Chemistry tables: {float(ox_msk.sum(1).mean()):.2f} oxidation "
              f"states/element (SMACT), {int(metal_lut.sum())}/{len(zs)} metals")

        _form = torch.zeros(cfg.num_types)
        for _vi, _z in enumerate(zs):
            if int(_z) in NON_FORMABLE_Z:
                _form[_vi] = float('-inf')
        self.register_buffer('formability_mask', _form, persistent=False)

        _R, _T, _ok = _get_sym_op_tensors()
        self.register_buffer('sym_rot_lut', _R, persistent=False)
        self.register_buffer('sym_trans_lut', _T, persistent=False)
        self.register_buffer('sym_valid_lut', _ok, persistent=False)

    def set_sg_class_weights(self, counts, power=0.5, max_ratio=8.0):
        """Inverse-frequency weights, capped, with unseen groups neutralised.

        """
        if counts is None:
            return
        if isinstance(counts, dict):
            arr = np.array([counts.get(s, 0) for s in range(1, 231)], dtype=np.float64)
        else:
            arr = np.asarray(counts, dtype=np.float64).reshape(-1)[:230]
            if arr.size < 230:
                arr = np.pad(arr, (0, 230 - arr.size), constant_values=0.0)
        seen = arr > 0
        w = 1.0 / np.power(arr + 1.0, power)
        if seen.any():
            ws = np.minimum(w[seen], w[seen].min() * float(max_ratio))
            w[seen] = ws
            w[~seen] = ws.min()
            w = w / max(w[seen].mean(), 1e-8)
        else:
            w = np.ones_like(w)
        self.sg_class_weights.copy_(torch.as_tensor(w, dtype=torch.float32))
        prior = arr / max(arr.sum(), 1.0)
        if prior.sum() <= 0:
            prior = np.ones(230) / 230.0
        self.sg_prior.copy_(torch.as_tensor(prior, dtype=torch.float32))
        print(f"  Space-group loss weights: capped at {max_ratio:.0f}:1, "
              f"{int((~seen).sum())} unseen group(s) neutralised; "
              f"empirical prior stored for generation")

    @staticmethod
    def _sg_pos_feat(frac, lattice, mask, detach=True, precomputed_d=None):
        ctx = (torch.enable_grad() if (not detach and torch.is_grad_enabled())
               else torch.no_grad())
        with ctx:
            B = frac.size(0)
            L = lattice
            lp = matrix_to_lattice_params(L)
            if precomputed_d is not None:
                d = precomputed_d
            else:
                _, d = min_image_disp(frac, frac, L, n_images=1)
            m = mask.bool().unsqueeze(2) & mask.bool().unsqueeze(1)
            eye = torch.eye(d.size(1), device=d.device, dtype=torch.bool).unsqueeze(0)
            m = m & ~eye
            BIG = 1.0e4
            dmask = torch.where(m, d, torch.full_like(d, BIG))
            any_pair = m.reshape(B, -1).any(-1)
            dmin = dmask.reshape(B, -1).min(-1).values
            dmin = torch.where(any_pair, dmin, torch.full_like(dmin, 5.0))
            k = int(max(1, (d.size(1) * d.size(2)) // 4))
            small = dmask.reshape(B, -1).topk(k, dim=-1, largest=False).values
            keep = (small < 0.5 * BIG).to(small.dtype)
            dmean = (small * keep).sum(-1) / keep.sum(-1).clamp(min=1.0)
            dmean = torch.where(any_pair, dmean, dmin)
            feat = torch.stack([lp[:, 0] / 10.0, lp[:, 1] / 10.0, lp[:, 2] / 10.0,
                                lp[:, 3] / 90.0, lp[:, 4] / 90.0, lp[:, 5] / 90.0,
                                dmin / 5.0, dmean / 5.0], -1)
        return feat

    def _sg_first_pass(self, z):
        logits = self.sg_coarse(z)
        return logits, logits.argmax(-1) + 1

    @torch.no_grad()
    def fit_property(self, ys):
        self.y_mean.copy_(ys.mean()); self.y_std.copy_(ys.std().clamp(min=1e-3))

    def ynorm(self, y):   return (y - self.y_mean) / self.y_std
    def ydenorm(self, y): return y * self.y_std + self.y_mean

    def _encode_target(self, batch, add_noise: bool = None):
        """Flow-matching target z1: the encoder's latent site set for a real crystal."""
        cfg = self.cfg
        if not bool(self._encoder_pretrained):
            raise RuntimeError(
                "DirectCrystalFlow.encoder has not been pretrained "
                "(self._encoder_pretrained is False). Training the flow "
                "against an untrained encoder's output fits noise to noise "
                "and will silently produce a useless model. Call "
                "run_encoder_pretrain(...) first, or load a checkpoint that "
                "already did so.")
        site_mask = (_asym_unit_mask_from_batch(batch, batch['mask'].bool())
                     if getattr(cfg, 'decode_asymmetric_unit', True)
                     else batch['mask'].bool())
        tok, keep = self.encoder(
            batch['types'], batch['frac'], batch['lattice'], batch['mask'],
            batch['src'], batch['dst'], batch['offset'], batch['emask'],
            elem_feat=(self.elem_feat if cfg.elem_feat_dim > 0 else None),
            site_mask=site_mask)
        mu = self.enc_standardizer(self.enc_proj(tok))
        if add_noise is None:
            add_noise = self.training
        if add_noise and cfg.latent_noise_std > 0:
            mu = mu + torch.randn_like(mu) * cfg.latent_noise_std
        return mu, keep

    def _empty_ref(self, dtype, device):
        """The encoder's learned 'this slot is unoccupied' embedding, passed
        through the same enc_proj/enc_standardizer that built z1 -- i.e. the
        exact point in the FLOW's latent space that an empty slot's target
        sits at. No new parameters; this is the fixed reference the flow's
        attention mask below measures distance against.
        """
        ref = self.enc_standardizer(self.enc_proj(
            self.encoder.empty_emb.to(device=device, dtype=dtype)))
        return ref

    def _occupancy_bias(self, z, hard_keep=None, scale=6.0, sharpness=3.0):
      
        if hard_keep is not None:
            return (~hard_keep.bool()).to(z.dtype) * -1e4
        ref = self._empty_ref(z.dtype, z.device)                  # (M, D)
        d = safe_norm(z - ref.unsqueeze(0))                        # (B, M)
        d_rel = d / d.mean(dim=1, keepdim=True).clamp(min=1e-6)
        return -float(scale) * torch.sigmoid(float(sharpness) * (1.0 - d_rel))

    def reconstruct(self, batch, encode_grad=True, add_noise=None):
        """Encode a real crystal and decode it again.
        """
        cfg = self.cfg
        at, frac, L, mask = batch['types'], batch['frac'], batch['lattice'], batch['mask']
        enc_frac = frac
        _an = float(getattr(cfg, 'aug_coord_noise', 0.0))
        if self.training and encode_grad and _an > 0:
            enc_frac = (frac + torch.randn_like(frac) * _an) % 1.0

        site_mask = (_asym_unit_mask_from_batch(batch, mask.bool())
                     if bool(getattr(cfg, 'decode_asymmetric_unit', True))
                     else mask.bool())

        _ctx = (torch.enable_grad() if (encode_grad and torch.is_grad_enabled())
                else torch.no_grad())
        with _ctx:
            tok, keep = self.encoder(
                at, enc_frac, L, mask, batch['src'], batch['dst'],
                batch['offset'], batch['emask'],
                elem_feat=(self.elem_feat if cfg.elem_feat_dim > 0 else None),
                site_mask=site_mask, order_frac=frac)
            z = self.enc_standardizer(self.enc_proj(tok))
        if not encode_grad:
            z = z.detach()

        if add_noise is None:
            add_noise = self.training
        z_dec = z
        if add_noise and cfg.latent_noise_std > 0:
            z_dec = z + torch.randn_like(z) * cfg.latent_noise_std

        decode_asym = bool(getattr(cfg, 'decode_asymmetric_unit', True))
        keep_t, tgt_types, tgt_frac, tgt_mult, tgt_P, tgt_o, tgt_wyck = \
            _canonical_site_targets(batch, cfg.n_sites, decode_asym=decode_asym)

        _tf = float(getattr(cfg, 'wyckoff_teacher', 1.0))
        _use_teacher = (_tf > 0.0 and getattr(self.decoder, 'use_wyckoff', True)
                        and (_tf >= 1.0 or not self.training
                             or float(torch.rand(())) < _tf))
        dec = self.decoder(z_dec, generation=False, sg=batch['sg'],
                           wyckoff_teacher=((tgt_P, tgt_o) if _use_teacher else None))
        if self.refiner is not None:
            dec['frac'] = self.refiner(
                dec['frac'], dec['type_probs'], dec['mask'], L,
                dec.get('site_proj') if getattr(cfg, 'refiner_project_sites', True) else None)

        pm = dec['mask'].bool()
        tm = site_mask

        _gauge = str(getattr(cfg, 'site_align_gauge', 'auto')).lower()
        if _gauge == 'auto':
            _use_site_gauge = not bool(getattr(cfg, 'use_pos_features', True))
        else:
            _use_site_gauge = _gauge not in ('off', 'none', 'false', '0')
        if getattr(cfg, 'translation_invariant_recon', True):
            pf, shift = align_pred_frac(dec['frac'], frac, L, pm, tm,
                                        int(getattr(cfg, 'align_iters', 3)),
                                        return_shift=True)
        else:
            pf, shift = dec['frac'], None
        if _use_site_gauge:
            pf_site = (dec['frac'] + canonical_align_shift(
                dec['frac'], tgt_frac, keep_t,
                int(getattr(cfg, 'align_iters', 3)))) % 1.0
        else:
            pf_site = dec['frac'] % 1.0

        mult_pred = self.mult_head(z_dec)
        w_site = dec['occ_soft'] * mult_pred
        L_pred, V_pred, _ = self.lattice(z_dec, w_site)

        return dict(z=z, z_dec=z_dec, dec=dec, keep=keep_t, pm=pm, tm=tm,
                    pf=pf, pf_site=pf_site, shift=shift,
                    tgt_types=tgt_types, tgt_frac=tgt_frac,
                    tgt_mult=tgt_mult, tgt_P=tgt_P, tgt_o=tgt_o,
                    tgt_wyck=tgt_wyck,
                    wyckoff_teacher=_use_teacher,
                    mult_pred=mult_pred, w_site=w_site,
                    L_pred=L_pred, V_pred=V_pred)

    def freeze_encoder(self, freeze_property=True):
        """Freeze the encoder stack (and the property head) and mark the model pretrained.
        """
        for mod in (self.encoder, self.enc_proj, self.enc_standardizer):
            mod.eval()
            for p in mod.parameters():
                p.requires_grad = False
        if freeze_property:
            self.property.eval()
            for p in self.property.parameters():
                p.requires_grad = False
        self.refresh_property_surrogate()
        self._encoder_pretrained.fill_(True)
        return self

    @torch.no_grad()
    def refresh_property_surrogate(self):
        self.property_frozen.load_state_dict(self.property.state_dict())
        for p in self.property_frozen.parameters():
            p.requires_grad = False
        self.property_frozen.eval()

    def property_from_structure(self, frac, type_probs, L, mask, sg=None):
        """Read the property off the DECODED CRYSTAL
        """
        cfg = self.cfg
        if sg is not None and bool(getattr(cfg, 'surrogate_expand_orbits', True)):
            n_ops = int(max(1, getattr(cfg, 'surrogate_expand_ops', 8)))
            f_e, m_e, _ = symmetry_orbit(
                frac, mask, type_probs.argmax(-1), sg,
                self.sym_rot_lut, self.sym_trans_lut, self.sym_valid_lut,
                n_ops=n_ops, stochastic=False, L=L, dedup=True)
            A0 = frac.size(1)
            p_e = type_probs.repeat(1, n_ops, 1)[:, :f_e.size(1)]
           
            site_m = torch.zeros_like(m_e, dtype=torch.bool)
            site_m[:, :A0] = mask.bool()
           
            m_e, f_e, p_e, site_m = compact_by_mask(m_e, f_e, p_e, site_m)
        else:
            f_e, m_e, p_e = frac, mask.to(frac.dtype), type_probs
            site_m = mask.bool()
        if int(site_m.sum(1).max()) > cfg.n_sites:
            _data_warn(
                f"property_from_structure was given more than n_sites "
                f"({cfg.n_sites}) asymmetric-unit sites; the surrogate sees the "
                f"first {cfg.n_sites} in canonical order only.")
        src, dst, off, em = build_edges_from_geometry(
            f_e, L, m_e, cutoff=cfg.cutoff,
            max_deg=int(getattr(cfg, 'surrogate_max_deg', 12)),
            max_self=int(getattr(cfg, 'surrogate_max_self', 6)))
        tok, keep = self.encoder(
            None, f_e, L, m_e, src, dst, off, em, type_probs=p_e,
            elem_feat=(self.elem_feat if cfg.elem_feat_dim > 0 else None),
            site_mask=site_m)
        z = self.enc_standardizer(self.enc_proj(tok))
        y, _ = self.property_frozen(z, keep.to(z.dtype))
        return y.squeeze(-1), z

    @torch.no_grad()
    def diagnostics(self, batch, rollout_steps=16, verbose=True):
        """Measure, on one batch, the failure modes this architecture exists to remove.

          ref/gen_token_spread  std ACROSS tokens of an encoded and a generated latent
                                set; gen << ref means the flow collapsed the set
          token_diversity       mean pairwise cosine distance between occupied tokens,
                                generated and encoded; 0.0 is total collapse
          n_sites_gen/ref       decoded vs true asymmetric-unit size
          n_elem_gen            distinct elements per generated cell
          full_over_cell        predicted full-cell count over the realised
                                multiplicity-weighted count: the VPA inflation factor
          cond_*_norm           conditioning pathway norms; near zero means the flow
                                never learned to read y or sg
        """
        was_training = self.training
        self.eval()
        cfg = self.cfg
        dev = batch['y'].device
        B = batch['y'].size(0)
        out = {}

        site_mask = (_asym_unit_mask_from_batch(batch, batch['mask'].bool())
                     if getattr(cfg, 'decode_asymmetric_unit', True)
                     else batch['mask'].bool())
        tok, keep = self.encoder(
            batch['types'], batch['frac'], batch['lattice'], batch['mask'],
            batch['src'], batch['dst'], batch['offset'], batch['emask'],
            elem_feat=(self.elem_feat if cfg.elem_feat_dim > 0 else None),
            site_mask=site_mask)
        z1 = self.enc_standardizer(self.enc_proj(tok))
        out['ref_token_spread'] = float(z1.std(dim=1).mean())
        out['n_sites_ref'] = float(keep.float().sum(1).mean())

        yn = self.ynorm(batch['y'])
        sg_b = batch['sg'].view(-1).clamp(1, 230)
        cond = self.conditioner(yn, B, dev, drop_prob=0.0, training=False,
                                sg=sg_b, sg_drop_prob=0.0)
        z = torch.randn(B, cfg.n_sites, cfg.sec_dim, device=dev)
        dt = 1.0 / rollout_steps
        ts = torch.linspace(0, 1, rollout_steps + 1, device=dev)
        for i in range(rollout_steps):
            z = _rollout_step(self, z, ts[i:i + 1], dt, 'rk2',
                              lambda zz, s: self.flow(
                                  zz, s.view(1, 1).expand(zz.size(0), 1), cond,
                                  key_padding_mask=(
                                      self._occupancy_bias(zz)
                                      if getattr(cfg, 'flow_occ_mask', True) else None)))
        out['gen_token_spread'] = float(z.std(dim=1).mean())
        _zr = F.normalize(z1, dim=-1)
        _cr = torch.einsum('bid,bjd->bij', _zr, _zr)
        _pr = keep.float().unsqueeze(2) * keep.float().unsqueeze(1)
        _pr = _pr * (1.0 - torch.eye(cfg.n_sites, device=dev).unsqueeze(0))
        out['ref_token_diversity'] = float(
            ((1.0 - _cr) * _pr).sum() / _pr.sum().clamp(min=1.0))

        dec = self.decoder(z, generation=False,
                           formability_mask=self.formability_mask, sg=sg_b)
        msk = dec['mask']
        out['n_sites_gen'] = float(msk.sum(1).mean())

        zn = F.normalize(z, dim=-1)
        cos = torch.einsum('bid,bjd->bij', zn, zn)
        pair = msk.unsqueeze(2) * msk.unsqueeze(1)
        pair = pair * (1.0 - torch.eye(cfg.n_sites, device=dev).unsqueeze(0))
        out['token_diversity'] = float(
            ((1.0 - cos) * pair).sum() / pair.sum().clamp(min=1.0))

        mult_pred = self.mult_head(z)
        w_site = dec['occ_soft'] * mult_pred
        n_full = w_site.sum(1, keepdim=True).clamp(min=1.0)
        L, V, _ = self.lattice(z, w_site)
        L = self._finalize_lattice(
            L, torch.maximum(V, n_full * float(getattr(cfg, 'vpa_floor', 5.0))), sg_b)

        out['type_confidence'] = float(
            (dec['type_probs'].max(-1).values * msk).sum() / msk.sum().clamp(min=1.0))
        mult = orbit_multiplicity(dec['frac'], msk, sg_b, self.sym_rot_lut,
                                  self.sym_trans_lut, self.sym_valid_lut, L=L,
                                  snap_tol=float(getattr(cfg, 'sym_snap_tol', SYM_SNAP_TOL)))
        n_cell = (msk * mult.clamp(min=1.0)).sum(1, keepdim=True).clamp(min=1.0)
        out['full_over_cell'] = float((n_full / n_cell).mean())
        out['n_atoms_cell'] = float(n_cell.mean())
        cw = msk * mult.clamp(min=1.0)
        cnt = (F.one_hot(dec['sampled_types'], cfg.num_types).float()
               * cw.unsqueeze(-1)).sum(1)
        out['n_elem_gen'] = float((cnt > 0.5).float().sum(-1).mean())
        _true_n = batch['mask'].float().sum(-1).clamp(min=1.0)
        out['vpa_gen'] = float((torch.linalg.det(L).abs() / n_cell.squeeze(-1)).mean())
        out['vpa_ref'] = float((torch.linalg.det(batch['lattice']).abs() / _true_n).mean())

        out['cond_to_tokens_norm'] = float(sum(
            b.cond_proj.weight.norm() for b in self.flow.blocks))
        out['sg_emb_norm'] = float(self.conditioner.sg_emb.weight.norm())
        out['y_enc_norm'] = float(self.conditioner.enc[-1].weight.norm())

        if verbose:
            print("\n" + "=" * 66)
            print("  DirectCrystalFlow diagnostics (GNN site-token model)")
            print("=" * 66)
            print(f"  token spread  generated / encoded    : "
                  f"{out['gen_token_spread']:.3f} / {out['ref_token_spread']:.3f}")
            if out['gen_token_spread'] < 0.5 * out['ref_token_spread']:
                print("    *** the flow has COLLAPSED the latent set. ***")
            print(f"  token diversity  generated / encoded  : "
                  f"{out['token_diversity']:.3f} / "
                  f"{out['ref_token_diversity']:.3f}")
            if out['token_diversity'] < 0.05:
                print("    *** occupied tokens are essentially IDENTICAL: the "
                      "decoder cannot emit more than one element. ***")
            print(f"  sites per cell  generated / reference : "
                  f"{out['n_sites_gen']:.2f} / {out['n_sites_ref']:.2f}  "
                  f"(capacity {cfg.n_sites})")
            print(f"  per-site type confidence              : "
                  f"{out['type_confidence']:.3f}")
            print(f"  distinct elements / generated cell    : "
                  f"{out['n_elem_gen']:.2f}")
            print(f"  n_full / n_cell (VPA inflation)       : "
                  f"{out['full_over_cell']:.2f}")
            print(f"  VPA generated / reference             : "
                  f"{out['vpa_gen']:.1f} / {out['vpa_ref']:.1f}")
            print(f"  conditioning norms  cond={out['cond_to_tokens_norm']:.3f}"
                  f"  sg_emb={out['sg_emb_norm']:.3f}"
                  f"  y_enc={out['y_enc_norm']:.3f}")
            if out['cond_to_tokens_norm'] < 1e-2:
                print("    *** the conditioning projection is still at its ZERO "
                      "init: the flow never learned to read y or sg. ***")
            print("=" * 66 + "\n")

        if was_training:
            self.train()
        return out

    def _finalize_lattice(self, L_pred, V_pred, sg_pred):
        L_params = matrix_to_lattice_params(L_pred)
        L_params = project_to_crystal_family(L_params, sg_pred)
        L = lattice_params_to_matrix(L_params)
        V_now = torch.linalg.det(L).abs().clamp(min=1e-3)
        L = L * (V_pred.squeeze(-1) / V_now).clamp(min=1e-6).pow(1 / 3).view(-1, 1, 1)
        return L

    def flow_losses(self, batch, encoder=None):
        """Flow matching against the frozen encoder's latent site set."""
        cfg = self.cfg
        y = batch['y']
        B = y.size(0)

        z1, _keep = self._encode_target(batch, add_noise=True)
        z1 = z1.detach()          # the encoder is frozen; never leak into it

        z0 = torch.randn_like(z1)
        t = torch.rand(B, 1, device=z1.device)
        te = t.unsqueeze(1)
        zt = (1 - te) * z0 + te * z1

        yn = self.ynorm(y)
        cond = self.conditioner(yn, B, z1.device, drop_prob=cfg.cfg_dropout,
                                sg=batch.get('sg'),
                                sg_drop_prob=float(getattr(cfg, 'cfg_sg_dropout',
                                                           cfg.cfg_dropout)))

        v_pred = self.flow(zt, t, cond, key_padding_mask=(
            self._occupancy_bias(zt, hard_keep=_keep)
            if getattr(cfg, 'flow_occ_mask', True) else None))
        v_tgt = z1 - z0
        loss = _flow_mse(v_pred, v_tgt, _keep,
                         float(getattr(cfg, 'flow_occ_weight', 3.0)))
        comps = {'flow': float(loss.detach())}

        if 'sg' in batch:
            sg_true = batch['sg'].view(-1).clamp(1, 230) - 1
            sg_logits, _ = self._sg_first_pass(z1)
            sg_l = F.cross_entropy(sg_logits, sg_true)
            loss = loss + cfg.weights.get('sg', 1.0) * sg_l
            comps['sg'] = float(sg_l.detach())
        return loss, comps

    def _recon_losses(self, batch, w):
        """Reconstruction anchor: encode a real crystal, decode, grade it.

        Returned as a dict of already-weighted terms so that stage 3 and the
        encoder pretraining share one definition of "did the decoder reproduce
        this crystal".
        """
        cfg = self.cfg
        r = self.reconstruct(batch, encode_grad=False, add_noise=False)
        dec, pm, tm, pf = r['dec'], r['pm'], r['tm'], r['pf']
        keep = r['keep']
        out = {}

        ch = periodic_chamfer(pf, batch['frac'], batch['lattice'], pm, tm,
                              pw=dec['occ_soft'])
        # SURPLUS SITES ONLY -- see pretrain_encoder_losses for why the
        # nearest-neighbour type CE must not touch slots that already have an
        # exact canonical target.
        ty = type_loss_nn(dec['type_logits'], batch['types'], pf, batch['frac'],
                          batch['lattice'], pm & ~keep, tm,
                          formability_mask=self.formability_mask,
                          reverse_weight=float(getattr(cfg, 'type_reverse_weight', 0.0)))
        out['recon'] = w.get('recon', 1.0) * (ch + w.get('type', 0.3) * ty)

        st, sp = site_alignment_losses(dec, r['pf_site'], keep, r['tgt_types'],
                                       r['tgt_frac'], batch['lattice'])
        out['site_type'] = w.get('site_type', 1.0) * st
        out['site_pos'] = w.get('site_pos', 1.0) * sp

        kf = keep.to(r['mult_pred'].dtype)
        out['mult'] = w.get('mult', 0.5) * (
            F.smooth_l1_loss(r['mult_pred'], r['tgt_mult'], beta=1.0,
                             reduction='none') * kf).sum() / kf.sum().clamp(min=1.0)
        # NOTE the key: the generative branch has its own `count` term and
        # both land in one dict, so this one must not shadow it.
        out['recon_count'] = w.get('count', 1.0) * F.smooth_l1_loss(
            dec['n_sites_soft'], kf.sum(-1).clamp(min=1.0), beta=1.0)

        tp_ = matrix_to_lattice_params(batch['lattice'].detach())
        pp_ = matrix_to_lattice_params(r['L_pred'])
        out['lattice'] = w.get('lattice', 1.0) * (
            F.smooth_l1_loss(torch.log(pp_[:, :3].clamp(min=1e-2)),
                             torch.log(tp_[:, :3].clamp(min=1e-2)), beta=0.3)
            + F.smooth_l1_loss(pp_[:, 3:] / 180.0, tp_[:, 3:] / 180.0, beta=0.1)
            + F.smooth_l1_loss(
                torch.log(r['V_pred'].squeeze(-1).clamp(min=1e-2)),
                torch.log(torch.linalg.det(batch['lattice'].detach()).abs().clamp(min=1e-2)),
                beta=0.3))

        _wl = dec.get('wyckoff_logits')
        _wce, _ = wyckoff_class_ce(_wl, r['tgt_wyck'], batch['sg'],
                                   self.decoder.wyckoff_valid)
        if _wce is not None:
            out['wyckoff_ce'] = w.get('wyckoff_ce', 1.0) * _wce

        if w.get('wyckoff', 0.0) > 0:
            out['wyckoff'] = w.get('wyckoff', 0.5) * wyckoff_proj_loss(
                dec['site_P'], dec['site_o'], pf, batch['frac'],
                batch.get('wyckoff_proj'), batch.get('wyckoff_offset'),
                batch['lattice'], pm, tm, 1, shift=r['shift'])
        return out

    def stage3_losses(self, batch, rollout_steps=8, gen_weight=1.0, encoder=None,
                      apply_warmup=True):
        """Generative fine-tuning: roll the flow out from noise, decode, and
        grade the crystal that comes out.
        """
        cfg = self.cfg
        recon_every_k = int(getattr(cfg, 'recon_every_k', 1))
        if torch.is_grad_enabled():
            self._stage3_step += 1
            _compute_recon = (int(self._stage3_step.item()) % recon_every_k == 0)
        else:
            _compute_recon = True
        y = batch['y']
        B = y.size(0)
        D = cfg.sec_dim
        M = cfg.n_sites
        yn = self.ynorm(y)
        w = cfg.weights
        sg_batch = batch['sg'].view(-1).clamp(1, 230)

        with torch.no_grad():
            z1, _keep_fm = self._encode_target(batch, add_noise=True)
        z0 = torch.randn_like(z1)
        t = torch.rand(B, 1, device=y.device)
        te = t.unsqueeze(1)
        zt = (1 - te) * z0 + te * z1
        cond_fm = self.conditioner(yn, B, y.device, drop_prob=cfg.cfg_dropout,
                                   training=torch.is_grad_enabled(),
                                   sg=sg_batch,
                                   sg_drop_prob=float(getattr(
                                       cfg, 'cfg_sg_dropout', cfg.cfg_dropout)))
        flow_loss = _flow_mse(self.flow(zt, t, cond_fm, key_padding_mask=(
            self._occupancy_bias(zt, hard_keep=_keep_fm)
            if getattr(cfg, 'flow_occ_mask', True) else None)),
            z1 - z0, _keep_fm,
                              float(getattr(cfg, 'flow_occ_weight', 3.0)))

      
        cond = self.conditioner(yn, B, y.device, drop_prob=0.0, training=False,
                                sg=sg_batch, sg_drop_prob=0.0)
        cond_null = self.conditioner(None, B, y.device, sg=sg_batch,
                                     sg_drop_prob=0.0)
        z = torch.randn(B, M, D, device=y.device)
        dt = 1.0 / rollout_steps
        ts = torch.linspace(0, 1, rollout_steps + 1, device=y.device)
        solver = str(getattr(cfg, 'stage3_solver', 'rk2')).lower()
        _g3 = getattr(cfg, 'stage3_guidance', None)
        _g3 = 1.0 if _g3 is None else float(_g3)
        _kgrad = int(max(1, min(getattr(cfg, 'stage3_grad_steps', 2), rollout_steps)))
        _first_grad = rollout_steps - _kgrad
        _use_occ_mask = bool(getattr(cfg, 'flow_occ_mask', True))

        def _vel(zz, s):
            tt = s.view(1, 1).expand(zz.size(0), 1)
            kpm = self._occupancy_bias(zz) if _use_occ_mask else None
            vc = self.flow(zz, tt, cond, key_padding_mask=kpm)
            if _g3 != 1.0:
                vu = self.flow(zz, tt, cond_null, key_padding_mask=kpm)
                return vu + _g3 * (vc - vu)
            return vc

     
        _pre_solver = str(getattr(cfg, 'stage3_prefix_solver', 'euler')).lower()
        for i in range(rollout_steps):
            tt = ts[i:i + 1]
            if i == _first_grad:
                z = z.detach()
            _ctx = (torch.no_grad() if i < _first_grad
                    else torch.enable_grad() if torch.is_grad_enabled()
                    else torch.no_grad())
            with _ctx:
                z = _rollout_step(self, z, tt, dt,
                                  _pre_solver if i < _first_grad else solver,
                                  _vel)

        _detach_dec = bool(getattr(cfg, 'stage3_decode_detach', True))
        z_dec = z.detach() if _detach_dec else z

        # ---- decode, then build the cell around what was decoded -------------
        sg_pred = sg_batch
        dec = self.decoder(z_dec, generation=False,
                           formability_mask=self.formability_mask, sg=sg_pred)
        occ = dec['occ_soft']
        L, V, n_full, _ = self._build_cell(
            z_dec, dec, sg_pred, refine=(self.refiner is not None))
        _, dist_pp = min_image_disp(dec['frac'], dec['frac'], L, n_images=1)

        gen_losses = {}
        if _compute_recon:
            gen_losses.update(self._recon_losses(batch, w))
        else:
            gen_losses['recon'] = torch.zeros((), device=y.device)

     
        y_dec, _ = self.property(z, occ.detach())
        gen_losses['property'] = (w.get('property', 1.5)
                              * float(getattr(cfg, 'stage3_property_scale', 1.5))
                              * F.smooth_l1_loss(y_dec.squeeze(-1), yn, beta=0.5))

    
        _pk = int(max(1, getattr(cfg, 'property_every_k', 4)))
        _do_prop = (not torch.is_grad_enabled()
                    or int(self._stage3_step.item()) % _pk == 0)
        y_str = None
        if _do_prop and (w.get('property_struct', 0.0) > 0
                         or w.get('property_rank', 0.0) > 0):
            y_str, _ = self.property_from_structure(
                dec['frac'], dec.get('type_probs_st', dec['type_probs']),
                L, dec['mask'], sg=sg_pred)
            # Scaled by k so the term keeps the same average weight in the
            # objective whether or not it is amortised.
            gen_losses['property_struct'] = (
                w.get('property_struct', 3.0) * float(_pk)
                * F.smooth_l1_loss(y_str, yn, beta=0.5))

        if y_str is not None and w.get('property_rank', 0.0) > 0 and B > 1:
           
            yv = yn.detach()
            di = yv.unsqueeze(1) - yv.unsqueeze(0)
            dj = y_str.unsqueeze(1) - y_str.unsqueeze(0)
            pair_mask = (di.abs() > 1e-3).float()
            margin = float(getattr(cfg, 'rank_margin', 0.15))
            pair_loss = F.relu(margin - torch.sign(di) * dj) * pair_mask
            gen_losses['property_rank'] = w.get('property_rank', 2.0) * float(_pk) * (
                pair_loss.sum() / pair_mask.sum().clamp(min=1.0))

        _wm = dec.get('wyckoff_mult')
        if _wm is not None:
            cell_w = dec['mask'] * _wm.to(dec['mask'].dtype).clamp(min=1.0)
        else:
            with torch.no_grad():
                _mult = orbit_multiplicity(
                    dec['frac'], dec['mask'], sg_pred, self.sym_rot_lut,
                    self.sym_trans_lut, self.sym_valid_lut, L=L,
                    snap_tol=float(getattr(cfg, 'sym_snap_tol', SYM_SNAP_TOL)),
                    merge_dist=float(getattr(cfg, 'min_contact_abs', 0.75)))
            cell_w = dec['mask'] * _mult.clamp(min=1.0)
        n_cell = cell_w.sum(1, keepdim=True).clamp(min=1.0)

        
        _probs_st = dec.get('type_probs_st', dec['type_probs'])

        _nops = int(max(1, getattr(cfg, 'repulsion_ops', 16)))
        f_img, m_img, t_img = symmetry_orbit(
            dec['frac'], dec['mask'], dec['sampled_types'], sg_pred,
            self.sym_rot_lut, self.sym_trans_lut, self.sym_valid_lut,
            n_ops=_nops, stochastic=False, L=L, dedup=True,
            select=str(getattr(cfg, 'repulsion_op_select', 'nearest')))
        rep_l = hardcore_repulsion(
            dec['frac'], L, dec['mask'], dec['sampled_types'], self.radii_lut,
            scale=float(getattr(cfg, 'repulsion_scale', 0.70)),
            n_images=1, other=(f_img, m_img, t_img), chunk=9)
        gen_losses['repulsion'] = w.get('repulsion', 1.5) * rep_l

        ct, cf, cn, ce_ng = charge_neutrality_loss(
            _probs_st, cell_w, self.ox_states_padded, self.ox_states_mask,
            return_components=True,
            eneg_lut=self.eneg_lut, metal_lut=self.metal_lut,
            w_eneg=float(getattr(cfg, 'charge_eneg_weight', 1.0)),
            include_alloys=bool(getattr(cfg, 'smact_include_alloys', True)),
            exempt_unary=bool(getattr(cfg, 'smact_exempt_unary', True)))
        gen_losses['charge'] = w.get('charge', 2.0) * ct

        vpa = (torch.linalg.det(L).abs() / n_cell.squeeze(-1)).clamp(min=1e-3)

       
        _nt = dec['type_probs'].size(-1)
        _tm_cell = batch['mask'].float()
        _true_counts = (F.one_hot(batch['types'].clamp(0, _nt - 1), _nt).float()
                        * _tm_cell.unsqueeze(-1)).sum(1)               # (B, C)
        _pred_counts = (_probs_st * cell_w.unsqueeze(-1)).sum(1)
        _tm_site = (_asym_unit_mask_from_batch(batch, batch['mask'].bool())
                    if getattr(cfg, 'decode_asymmetric_unit', True)
                    else batch['mask'].bool())

        if bool(getattr(cfg, 'set_assignment', True)) and B > 1:
           
            _asg = composition_set_assignment(
                _pred_counts.detach(), _true_counts.detach(),
                size_weight=float(getattr(cfg, 'assign_size_weight', 0.25)),
                cond_y=(yn.detach().view(-1)
                        if bool(getattr(cfg, 'assign_respect_conditioning', True))
                        else None),
                cond_sg=(sg_batch.detach().view(-1)
                         if bool(getattr(cfg, 'assign_respect_conditioning', True))
                         else None),
                y_weight=float(getattr(cfg, 'assign_y_weight', 0.5)),
                sg_penalty=float(getattr(cfg, 'assign_sg_penalty', 10.0)))
        else:
            _asg = torch.arange(B, device=y.device)

      
        _pf = _pred_counts / _pred_counts.sum(-1, keepdim=True).clamp(min=1e-6)
        _tf = _true_counts / _true_counts.sum(-1, keepdim=True).clamp(min=1e-6)
        comp_l = F.smooth_l1_loss(_pf, _tf[_asg], beta=0.05,
                                  reduction='none').sum(-1).mean()
        gen_losses['composition'] = w.get('composition', 2.0) * comp_l

        _true_vpa = (torch.linalg.det(batch['lattice'].detach()).abs()
                     / _tm_cell.sum(-1).clamp(min=1.0))[_asg].clamp(min=1e-3)
        vpa_l = F.smooth_l1_loss(torch.log(vpa), torch.log(_true_vpa), beta=0.3)
        gen_losses['vpa'] = w.get('vpa', 1.0) * vpa_l

        _true_n_site = _tm_site.float().sum(-1)[_asg].clamp(1.0, float(M))
        count_l = F.smooth_l1_loss(dec['n_sites_soft'], _true_n_site, beta=1.0)
        gen_losses['count'] = w.get('count', 1.0) * count_l

        
        sg_true = sg_batch - 1
        _sg_detach = not bool(getattr(cfg, 'sg_pos_grad', True))
        sg_pos = self.sg_pos_encoder(self._sg_pos_feat(
            dec['frac'] if not _sg_detach else dec['frac'].detach(),
            L, dec['mask'].bool(), detach=_sg_detach, precomputed_d=dist_pp))
        sg_align = F.cross_entropy(self.sg(z_dec, pos_feat=sg_pos), sg_true)
        gen_losses['sg_align'] = w.get('sg_align', 1.0) * sg_align

        gen = sum(gen_losses.values())
      
        _gw = int(getattr(cfg, 'stage3_gen_warmup', 0))
        if _gw > 0 and apply_warmup:
            gen_weight = float(gen_weight) * min(
                1.0, float(int(self._stage3_step.item())) / float(_gw))
        total = flow_loss + gen_weight * gen

        comps = dict(flow=float(flow_loss.detach()),
                     repulsion=float(rep_l.detach()),
                     charge=float(ct.detach()),
                     vpa=float(vpa_l.detach()),
                     count=float(count_l.detach()),
                     sg_align=float(sg_align.detach()),
                     charge_feas=float(cf.mean().detach()),
                     charge_neut=float(cn.mean().detach()),
                     charge_eneg=float(ce_ng.mean().detach()))
        with torch.no_grad():
            comps['n_sites_gen'] = float(dec['mask'].sum(1).mean())
            comps['n_sites_soft'] = float(dec['n_sites_soft'].mean())
            comps['n_elem_gen'] = float((_pred_counts > 0.5).float().sum(-1).mean())
            comps['vpa_gen'] = float(vpa.mean())
            comps['vpa_ref'] = float(_true_vpa.mean())
            comps['n_cell_gen'] = float(n_cell.mean())
            comps['type_conf'] = float(dec['type_probs'].max(-1).values.mul(
                dec['mask']).sum() / dec['mask'].sum().clamp(min=1.0))
            _zn = F.normalize(z, dim=-1)
            _cos = torch.einsum('bid,bjd->bij', _zn, _zn)
            _pair = dec['mask'].unsqueeze(2) * dec['mask'].unsqueeze(1)
            _pair = _pair * (1.0 - torch.eye(M, device=y.device).unsqueeze(0))
            comps['token_div'] = float(
                ((1.0 - _cos) * _pair).sum() / _pair.sum().clamp(min=1.0))
            if dec.get('wyckoff_mult') is not None:
                comps['mult_gen'] = float(
                    (cell_w.sum() / dec['mask'].sum().clamp(min=1.0)))
        
        for _k in ('recon', 'composition', 'lattice', 'wyckoff',
                   'property_struct', 'mult', 'site_type', 'site_pos',
                   'recon_count', 'wyckoff_ce', 'property_rank'):
            if _k in gen_losses:
                _wk = float(w.get('count' if _k == 'recon_count' else _k, 1.0))
                if _k in ('property_struct', 'property_rank'):
                    _wk *= float(_pk)
                comps[_k] = float(gen_losses[_k].detach()) / max(_wk, 1e-8)
        comps['property'] = float(gen_losses['property'].detach()) / max(
            float(w.get('property', 1.5))
            * float(getattr(cfg, 'stage3_property_scale', 1.5)), 1e-8)

        return total, comps

    def _build_cell(self, z, dec, sg_pred, refine=True, correct_density=True,
                    band='train'):
        """Cell construction AFTER decoding.
        """
        cfg = self.cfg
        if str(band) == 'sample':
            vf = float(getattr(cfg, 'sample_vpa_floor', 3.0))
            vc = float(getattr(cfg, 'sample_vpa_ceiling', 120.0))
        else:
            vf = float(getattr(cfg, 'vpa_floor', 10.0))
            vc = float(getattr(cfg, 'vpa_ceiling', 60.0))
       
        wm = dec.get('wyckoff_mult')
        if wm is not None and bool(getattr(cfg, 'use_wyckoff_multiplicity', True)):
            mult_pred = wm.to(dec['occ_soft'].dtype)
        else:
            mult_pred = self.mult_head(z)
        w_site = dec['occ_soft'] * mult_pred
        n_full = w_site.sum(1, keepdim=True).clamp(min=1.0)
        L, V, _ = self.lattice(z, w_site)
       
        V = V.clamp(min=1e-3)
        V = _clamp_st(V, n_full * vf, n_full * vc)
        L = self._finalize_lattice(L, V, sg_pred)
        if correct_density and wm is None:
            mult = orbit_multiplicity(
                dec['frac'], dec['mask'], sg_pred,
                self.sym_rot_lut, self.sym_trans_lut, self.sym_valid_lut, L=L,
                snap_tol=float(getattr(cfg, 'sym_snap_tol', SYM_SNAP_TOL)),
                merge_dist=float(getattr(cfg, 'min_contact_abs', 0.75)))
            n_cell = (dec['mask'] * mult.clamp(min=1.0)).sum(1, keepdim=True).clamp(min=1.0)
            V = (V / n_full) * n_cell
            V = _clamp_st(V, n_cell * vf, n_cell * vc)
            L = self._finalize_lattice(L, V, sg_pred)

        if refine and self.refiner is not None:
            dec['frac'] = self.refiner(
                dec['frac'], dec['type_probs'], dec['mask'], L,
                dec.get('site_proj') if getattr(cfg, 'refiner_project_sites', True) else None)
        return L, V, n_full, w_site

    @torch.no_grad()
    def _decode_latent(self, z, cond, temperature=0.0, target_sg=None,
                       refine=True, t=None, vel=None):
        """The decode half of sample(), reusable at any point along the ODE.

        `cond` is kept in the signature for call-site compatibility; nothing
        in the decode path reads it now that the capsule gate is gone.
        """
        B = z.size(0)
        _, _sg_coarse = self._sg_first_pass(z)
        if target_sg is None:
            sg_pred = _sg_coarse
        else:
            _ts = torch.as_tensor(target_sg, device=z.device).reshape(-1).long().clamp(1, 230)
            sg_pred = _ts.expand(B).clone() if _ts.numel() == 1 else _ts

        dec = self.decoder(z, generation=True, temperature=temperature,
                           formability_mask=self.formability_mask, sg=sg_pred)
        L, V, n_full, _ = self._build_cell(z, dec, sg_pred, refine=refine,
                                           band='sample')

        out = dict(t=t, z=z.detach(), occ=dec['occ_soft'].detach(),
                   frac=dec['frac'].detach(), mask=dec['mask'].detach(),
                   types=dec['sampled_types'].detach(),
                   type_probs=dec['type_probs'].detach(), lattice=L.detach(),
                   volume=V.detach(),
                   n_sites=dec['n_sites_soft'].detach(),
                   n_atoms_full_pred=n_full.squeeze(-1).detach(),
                   sg_pred=sg_pred.detach(), sg_coarse=_sg_coarse.detach())
        if vel is not None and t is not None:
            tt = torch.as_tensor(float(t), device=z.device)
            out['v_norm'] = float(vel(z, tt).norm(dim=-1).mean())
        return out

    @torch.no_grad()
    def sample(self, n, target_y=None, steps=100, guidance=None, device='cpu',
                temperature=0.0, expand_orbits=True, target_sg=None,
                balance_composition=None, z0=None, trajectory_ts=None,
                cond_sg=None):
        """Generate n crystals."""
        self.eval()
        cfg = self.cfg
        guidance = cfg.guidance if guidance is None else guidance
        D = cfg.sec_dim
        M = cfg.n_sites

        tp = None
        if target_y is not None:
            tp = torch.full((n, 1), float(target_y), device=device)
            tp = self.ynorm(tp)
        if target_sg is not None:
            _ts = torch.as_tensor(target_sg, device=device).reshape(-1).long().clamp(1, 230)
            sg_cond = _ts.expand(n).clone() if _ts.numel() == 1 else _ts[:n]
        elif bool(getattr(cfg, 'sample_sg_from_prior', True)):
            _p = self.sg_prior.to(device).clamp(min=0)
            _p = _p / _p.sum().clamp(min=1e-8)
            sg_cond = torch.multinomial(_p, n, replacement=True) + 1
        else:
            sg_cond = None
        if cond_sg is None:
            sg_flow = sg_cond
        else:
            _cs = torch.as_tensor(cond_sg, device=device).reshape(-1).long().clamp(1, 230)
            sg_flow = _cs.expand(n).clone() if _cs.numel() == 1 else _cs[:n]
        cond = self.conditioner(tp, n, device, sg=sg_flow)          # (y, sg)
        cond_noy = self.conditioner(None, n, device, sg=sg_flow)    # ( -, sg)
        cond_nosg = self.conditioner(tp, n, device, sg=None)        # (y,  -)

        if z0 is not None:
            z = z0.to(device=device, dtype=torch.float32).reshape(n, M, D).clone()
        else:
            z = torch.randn(n, M, D, device=device)
        dt = 1.0 / steps
        ts = torch.linspace(0, 1, steps + 1, device=device)

        _solver = str(getattr(cfg, 'sample_solver', 'rk4')).lower()
        g_y = float(guidance)
        g_sg = float(getattr(cfg, 'sg_guidance', 1.0))
        _do_y = (tp is not None) and (g_y != 1.0)
        _do_sg = (sg_flow is not None) and (g_sg != 1.0)

        _use_occ_mask_s = bool(getattr(cfg, 'flow_occ_mask', True))

        def vel(zz, tt):
            t = tt.view(1, 1).expand(zz.size(0), 1)
            kpm = self._occupancy_bias(zz) if _use_occ_mask_s else None
            vc = self.flow(zz, t, cond, key_padding_mask=kpm)
            if not (_do_y or _do_sg):
                return vc
            v = vc
            if _do_y:
                v = v + (g_y - 1.0) * (vc - self.flow(zz, t, cond_noy, key_padding_mask=kpm))
            if _do_sg:
                v = v + (g_sg - 1.0) * (vc - self.flow(zz, t, cond_nosg, key_padding_mask=kpm))
            return v

        snap_at = sorted({int(round(float(t) * steps)) for t in (trajectory_ts or [])}
                         & set(range(steps + 1)))
        traj = []
        if 0 in snap_at:
            traj.append(self._decode_latent(z, cond, temperature=0.0,
                                            target_sg=sg_cond, refine=False,
                                            t=0.0, vel=vel))
        for i in range(steps):
            z = _rollout_step(self, z, ts[i:i + 1], dt, _solver, vel)
            if (i + 1) in snap_at and (i + 1) < steps:
                traj.append(self._decode_latent(z, cond, temperature=0.0,
                                                target_sg=sg_cond, refine=False,
                                                t=(i + 1) / steps, vel=vel))

        sg_logits_0, sg_coarse_pred = self._sg_first_pass(z)

        sg_pred = sg_coarse_pred if sg_cond is None else sg_cond.to(z.device)

        dec = self.decoder(z, generation=True, temperature=temperature,
                           formability_mask=self.formability_mask, sg=sg_pred)
        L, V, n_full, _ = self._build_cell(z, dec, sg_pred, refine=True,
                                           band='sample')

        _wm = dec.get('wyckoff_mult')
        if _wm is not None:
            mult = _wm.to(dec['mask'].dtype) * dec['mask']
        else:
            mult = orbit_multiplicity(
                dec['frac'], dec['mask'], sg_pred, self.sym_rot_lut,
                self.sym_trans_lut, self.sym_valid_lut, L=L,
                snap_tol=float(getattr(cfg, 'sym_snap_tol', SYM_SNAP_TOL)))
        n_cell = (dec['mask'] * mult.clamp(min=1.0)).sum(1, keepdim=True).clamp(min=1.0)

        _vf = float(getattr(cfg, 'sample_vpa_floor', 3.0))
        _vc = float(getattr(cfg, 'sample_vpa_ceiling', 120.0))
        _vpa0 = (V.view(-1) / n_cell.view(-1).clamp(min=1.0)).clamp(_vf, _vc)
        _f0, _m0 = dec['frac'].clone(), dec['mask'].clone()
        _mult0 = mult.clone()
        _tries = int(max(1, getattr(cfg, 'resolve_passes', 3)))
        _res = None
        for _pass in range(_tries):
            dec['frac'], dec['mask'] = _f0.clone(), _m0.clone()
            n_cell = (_m0 * _mult0.clamp(min=1.0)).sum(1, keepdim=True).clamp(min=1.0)
            V = (_vpa0 * n_cell.view(-1)).clamp(min=1e-3).view(-1, 1)
            L = self._finalize_lattice(L, V, sg_pred)
            _res = resolve_site_conflicts(
                dec, sg_pred, L, self.radii_lut,
                min_dist=float(getattr(cfg, 'validity_min_dist', 0.75)),
                overlap_scale=float(getattr(cfg, 'validity_overlap_scale', 0.5)),
                n_relocate=int(getattr(cfg, 'site_relocate_tries', 32)),
                snap_tol=float(getattr(cfg, 'sym_snap_tol', SYM_SNAP_TOL)),
                max_atoms=int(getattr(cfg, 'expand_max_atoms', 400)),
                seed=_pass)
            _dropped_b = (_m0.sum(1) - dec['mask'].sum(1))
            
            if (_dropped_b > 0).any() and _pass + 1 < _tries:
                _grow = float(getattr(cfg, 'resolve_inflate', 1.6))
                _next = torch.where(_dropped_b > 0, _vpa0 * _grow, _vpa0)
                if bool((_next.clamp(max=_vc) > _vpa0 + 1e-6).any()):
                    _vpa0 = _next.clamp(_vf, _vc)
                    continue
            break
        
        mult = _res['mult'] * dec['mask']
        n_cell = (dec['mask'] * mult.clamp(min=1.0)).sum(1, keepdim=True).clamp(min=1.0)
        V = torch.linalg.det(L).abs().clamp(min=1e-3).view(-1, 1)
        n_full = n_cell.clone()

        _cap = int(getattr(cfg, 'unique_type_cap', 0) or 0)
        if _cap > 0:
            dec['sampled_types'], dec['type_probs'] = enforce_unique_types(
                dec['sampled_types'], dec['type_probs'], dec['mask'], _cap)

        raw_types = dec['sampled_types'].clone()
        _do_balance = (bool(getattr(cfg, 'smact_balance_on_sample', True))
                       if balance_composition is None else bool(balance_composition))
        if _do_balance:
            dec['sampled_types'], dec['type_probs'] = smact_balance_types(
                dec['sampled_types'], dec['type_probs'], dec['mask'],
                self.ox_states_padded, self.ox_states_mask,
                atomic_numbers=list(cfg.atomic_numbers),
                site_weights=mult)

        y_pred, _ = self.property(z, dec['occ_soft'])
        y_pred = self.ydenorm(y_pred)

        _sg_fine = self.sg(z, pos_feat=self.sg_pos_encoder(
            self._sg_pos_feat(dec['frac'], L, dec['mask'].bool())))

        coords = torch.einsum('bak,bkl->bal', dec['frac'], L)
        out = {'frac': dec['frac'], 'coords': coords, 'lattice': L,
                'type_probs': dec['type_probs'], 'sampled_types': dec['sampled_types'],
                'sampled_types_raw': raw_types,
                'mask': dec['mask'], 'sg_pred': sg_pred,
                'sg_coarse': sg_coarse_pred,
                'sg_fine': _sg_fine.argmax(-1) + 1,
                'n_sites_pred': dec['n_sites_soft'],
                'n_atoms_full_pred': n_full.squeeze(-1),
                'n_atoms_cell_est': n_cell.squeeze(-1),
                'wyckoff_multiplicity': mult,
                'property': y_pred, 'volume': V,
                'latent': z, 'occ': dec['occ_soft']}


        out['pre_expansion'] = {
            'frac': dec['frac'].clone(), 'lattice': L.clone(),
            'mask': dec['mask'].clone(),
            'sampled_types': dec['sampled_types'].clone(),
            'sampled_types_raw': raw_types.clone(),
            'type_probs': dec['type_probs'].clone(),
            'multiplicity': mult.clone(), 'sg': sg_pred.clone(),
            'balanced': bool(_do_balance), 'unique_type_cap': int(_cap)}

        if trajectory_ts is not None:
            traj.append(self._decode_latent(z, cond, temperature=temperature,
                                            target_sg=sg_cond, refine=True,
                                            t=1.0, vel=vel))
            out['trajectory'] = traj

        out['n_sites_relocated'] = _res['n_relocated']
        out['n_sites_dropped'] = _res['n_dropped']

        if expand_orbits:
          
            _nf_pri = (dec.get('wyckoff_nfree') if dec.get('wyckoff_nfree')
                       is not None else torch.zeros_like(dec['mask']))
            _pri = dec['type_probs'].detach().amax(-1) - 4.0 * _nf_pri
            out = expand_generated(
                out, sg_pred, self.cfg.num_types,
                resolved_orbits=_res['orbits'],
                priority=_pri.cpu().numpy(),
                snap_tol=float(getattr(cfg, 'sym_snap_tol', SYM_SNAP_TOL)),
                reconcile_types=bool(getattr(cfg, 'reconcile_types_on_sample', True)),
                max_atoms=int(getattr(cfg, 'expand_max_atoms', 200)),
                target_vpa=(V.squeeze(-1) / n_cell.squeeze(-1)).clamp(min=1e-3),
                rescale_volume=bool(getattr(cfg, 'rescale_volume_after_expand', True)),
                merge_dist=float(getattr(cfg, 'min_contact_abs', 0.75)))
            if (out['n_expansion_failed'] or out['n_orbit_truncated']
                    or out['n_site_conflicts']):
                _data_warn(
                    f"symmetry expansion: {out['n_expansion_failed']} failed, "
                    f"{out['n_orbit_truncated']} truncated, "
                    f"{out['n_site_conflicts']} unresolved site conflicts out "
                    f"of {n} (resolver: {_res['n_relocated']} sites relocated, "
                    f"{_res['n_dropped']} dropped; raise expand_max_atoms if "
                    f"truncation dominates).",
                    once=False)

        return out



def align_pred_frac(pf, tf, L, pm, tm, iters=3, return_shift=False):
    """Global fractional shift aligning the predicted set onto the true set.

    The encoder only ever sees interatomic displacements, so its latent carries
    no absolute origin; comparing absolute fractional coordinates therefore adds
    an irreducible offset error. The shift is detached: it is a nuisance
    parameter, not something the decoder should be able to game.
    """
    if not iters:
        return ((pf % 1.0, pf.new_zeros(pf.size(0), 1, 3)) if return_shift
                else pf % 1.0)
    pf = pf % 1.0
    tf = tf % 1.0
    shift = pf.new_zeros(pf.size(0), 1, 3)
    with torch.no_grad():
        for _ in range(int(iters)):
            cur = (pf + shift) % 1.0
            _, d = min_image_disp(cur, tf, L)
            d = d.masked_fill(~tm.unsqueeze(1), 1.0e4)
            j = d.argmin(2)
            tgt = torch.gather(tf, 1, j.unsqueeze(-1).expand(-1, -1, 3))
            df = (tgt - cur + 0.5) % 1.0 - 0.5
            w = pm.float().unsqueeze(-1)
            shift = shift + (df * w).sum(1, keepdim=True) / w.sum(1, keepdim=True).clamp(min=1.0)
    shift = shift.detach()
    out = (pf + shift) % 1.0
    return (out, shift) if return_shift else out

def canonical_align_shift(pred_frac, tgt_frac, keep, iters=3):
    """Global fractional shift s with (pred_i + s) ~ tgt_i, from correspondence.
    """
    with torch.no_grad():
        w = keep.to(pred_frac.dtype).unsqueeze(-1)
        wsum = w.sum(1, keepdim=True).clamp(min=1e-6)
        shift = pred_frac.new_zeros(pred_frac.size(0), 1, 3)
        for _ in range(max(1, int(iters))):
            d = 2.0 * math.pi * (tgt_frac - (pred_frac + shift))
            s = (d.sin() * w).sum(1, keepdim=True) / wsum
            c = (d.cos() * w).sum(1, keepdim=True) / wsum
            step = torch.atan2(s, c) / (2.0 * math.pi)
            shift = shift + step
            if float(step.abs().max()) < 1e-6:
                break
    return shift.detach()

def _canonical_site_targets(batch, n_sites, decode_asym=True):
    """True asymmetric-unit sites in the SAME canonical order the encoder uses.

    Returns (keep, types_s, frac_s, mult_s, P_s, o_s):
      keep    (B, M) bool     -- slot i holds a real site
      types_s (B, M) long     -- element index of site i
      frac_s  (B, M, 3)       -- fractional coordinates of site i
      mult_s  (B, M) float    -- mined Wyckoff multiplicity, 1.0 if absent
      P_s     (B, M, 3, 3)    -- TRUE Wyckoff projector of site i
      o_s     (B, M, 3)       -- TRUE Wyckoff offset of site i
      w_s     (B, M) long     -- TRUE Wyckoff class index, -100 where unmatched
                                 or where no lookup table was supplied
    """
    mask = batch['mask'].bool()
    site_mask = _asym_unit_mask_from_batch(batch, mask) if decode_asym else mask
    perm, keep = canonical_site_order(batch['types'], batch['frac'],
                                      site_mask, n_sites)
    types_s = torch.gather(batch['types'], 1, perm)
    frac_s = torch.gather(batch['frac'], 1, perm.unsqueeze(-1).expand(-1, -1, 3))
    wm = batch.get('wyckoff_multiplicity', None)
    if wm is not None and torch.is_tensor(wm):
        wm = wm.squeeze(-1) if wm.dim() == 3 else wm
    if (wm is not None and torch.is_tensor(wm)
            and tuple(wm.shape[:2]) == tuple(mask.shape[:2])):
        mult_s = torch.gather(wm.float(), 1, perm).clamp(min=1.0)
    else:
        mult_s = torch.ones(keep.shape, dtype=torch.float32, device=mask.device)

    types_s = torch.where(keep, types_s, torch.zeros_like(types_s))
    frac_s = frac_s * keep.unsqueeze(-1).to(frac_s.dtype)
    mult_s = torch.where(keep, mult_s, torch.ones_like(mult_s))

    B, M = keep.shape
    dev = mask.device
    eye = torch.eye(3, device=dev).view(1, 1, 3, 3)
    wp, wo = batch.get('wyckoff_proj'), batch.get('wyckoff_offset')
    if wp is not None and torch.is_tensor(wp) and wp.dim() == 4:
        P_s = torch.gather(wp.float(), 1, perm.view(B, M, 1, 1).expand(-1, -1, 3, 3))
    else:
        P_s = eye.expand(B, M, 3, 3).clone()
    if wo is not None and torch.is_tensor(wo) and wo.dim() == 3:
        o_s = torch.gather(wo.float(), 1, perm.unsqueeze(-1).expand(-1, -1, 3))
    else:
        o_s = torch.zeros(B, M, 3, device=dev)
    # Padding slots get the free general position, never a clamped one.
    P_s = torch.where(keep.view(B, M, 1, 1), P_s, eye.expand(B, M, 3, 3))
    o_s = o_s * keep.unsqueeze(-1).to(o_s.dtype)

    wi = batch.get('wyckoff_index', None)
    if wi is not None and torch.is_tensor(wi) and wi.dim() == 3:
        wi = wi.squeeze(-1)          # the batch builder densifies to (B, A, 1)
    if (wi is not None and torch.is_tensor(wi) and wi.dim() == 2
            and tuple(wi.shape) == tuple(mask.shape)):
        w_s = torch.gather(wi.long(), 1, perm)
        w_s = torch.where(keep, w_s, torch.full_like(w_s, -100))
    else:
        w_s = torch.full((B, M), -100, dtype=torch.long, device=dev)
    return keep, types_s, frac_s, mult_s, P_s, o_s, w_s

def site_alignment_losses(dec, pf, keep, tgt_types, tgt_frac, L):
    """DIRECT per-token supervision, slot i against true site i.
    """
    kf = keep.to(pf.dtype)
    denom = kf.sum().clamp(min=1.0)

    ce = F.cross_entropy(dec['type_logits'].transpose(1, 2),
                         tgt_types.clamp(min=0), reduction='none')
    ce_l = (ce * kf).sum() / denom

    df = (pf - tgt_frac + 0.5) % 1.0 - 0.5
    dcart = torch.einsum('bij,bjk->bik', df, L.to(df.dtype))
    dist = safe_norm(dcart)
    pos_l = (F.smooth_l1_loss(dist, torch.zeros_like(dist), beta=0.5,
                              reduction='none') * kf).sum() / denom
    return ce_l, pos_l

def periodic_chamfer(pf, tf, L, pm, tm, pw=None, cover_all_slots=True):
    """Correspondence-aware structural distance (nearest-image chamfer).

    """
    pf = pf % 1.0
    tf = tf % 1.0
    _, d = min_image_disp(pf, tf, L)
    D_CEIL = 20.0
    d = d.clamp(max=D_CEIL)
    nt = tm.sum(1).clamp(min=1)

    src = torch.ones_like(pm) if cover_all_slots else pm
    d_t2p = d.masked_fill(~src.unsqueeze(2), D_CEIL)
    t2p = d_t2p.min(1).values.masked_fill(~tm, 0.0)

    w = pm.float() if pw is None else pm.float() * pw.clamp(0.0, 1.0)
    d_p2t = d.masked_fill(~tm.unsqueeze(1), D_CEIL)
    p2t = d_p2t.min(2).values * w

    out = t2p.sum(1) / nt + p2t.sum(1) / nt
    ok = tm.sum(1) > 0
    if not bool(ok.any()):
        return pf.sum() * 0.0          # keeps grad_fn; new_zeros(()) does not
    return out[ok].mean()

def type_loss_nn(pred_logits, true_types, pf, tf, L, pm, tm, formability_mask=None,
                 reverse_weight=1.0):
    """Bidirectional nearest-neighbour type cross-entropy.
    """
    pf = pf % 1.0
    tf = tf % 1.0
    _, d = min_image_disp(pf, tf, L)          # (B, P, T)

    D_CEIL = 20.0
    if formability_mask is not None:
        _finite = torch.isfinite(formability_mask)
    else:
        _finite = None


    d_pt = d.masked_fill(~tm.unsqueeze(1), D_CEIL)
    nn_idx = d_pt.argmin(2)
    labels = torch.gather(true_types, 1, nn_idx)
    if _finite is not None:
        _bad_label = ~_finite.gather(0, labels.clamp(min=0).reshape(-1)
                                     ).reshape(labels.shape)
    else:
        _bad_label = torch.zeros_like(labels, dtype=torch.bool)
    ce = F.cross_entropy(pred_logits.transpose(1, 2), labels.clamp(min=0),
                         reduction='none')
    ce = ce.masked_fill(_bad_label, 0.0) * pm.float()
    n = pm.sum(1).clamp(min=1)
    ok = pm.sum(1) > 0
    fwd = ((ce.sum(1) / n)[ok].mean().clamp(0, 50) if ok.any()
           else ce.new_zeros(()))
    if reverse_weight <= 0:
        return fwd

  
    d_tp = d.masked_fill(~pm.unsqueeze(2), D_CEIL)
    nn_p = d_tp.argmin(1)                                  # (B, T) -> pred idx
    C = pred_logits.size(-1)
    logits_for_true = torch.gather(
        pred_logits, 1, nn_p.unsqueeze(-1).expand(-1, -1, C))
    tt = true_types.clamp(min=0)
    if _finite is not None:
        _bad_true = ~_finite.gather(0, tt.reshape(-1)).reshape(tt.shape)
    else:
        _bad_true = torch.zeros_like(tt, dtype=torch.bool)
    ce_r = F.cross_entropy(logits_for_true.transpose(1, 2), tt,
                           reduction='none')
    ce_r = ce_r.masked_fill(_bad_true, 0.0) * tm.float()
    # a sample with no valid predicted atom has no nearest neighbour to supervise
    ce_r = ce_r * (pm.sum(1, keepdim=True) > 0).float()
    n_t = tm.sum(1).clamp(min=1)
    ok_t = tm.sum(1) > 0
    rev = ((ce_r.sum(1) / n_t)[ok_t].mean().clamp(0, 50) if ok_t.any()
           else ce_r.new_zeros(()))
    return 0.5 * (fwd + float(reverse_weight) * rev)

def wyckoff_proj_loss(site_P, site_o, pf, tf, true_P, true_o, L, pm, tm, K,
                      shift=None):
    """Supervise the decoder's Wyckoff mixture against the mined site projectors.
    """
    if true_P is None or true_o is None:
        return site_P.new_zeros(())
    B, NK, _ = pf.shape
    _, d = min_image_disp(pf % 1.0, tf % 1.0, L)
    d = d.masked_fill(~tm.unsqueeze(1), 1.0e4)
    j = d.argmin(2)
    anc = torch.arange(0, NK, K, device=pf.device)
    j0 = j[:, anc]
    tp = torch.gather(true_P.reshape(B, -1, 9), 1,
                      j0.unsqueeze(-1).expand(-1, -1, 9)).reshape(B, -1, 3, 3)
    to = torch.gather(true_o, 1, j0.unsqueeze(-1).expand(-1, -1, 3))
    w = pm[:, anc].float().unsqueeze(-1)
    lp = ((site_P - tp).pow(2).sum(-1) * w).sum() / w.sum().clamp(min=1.0) / 3.0

    o_pred = site_o
    if shift is not None:
        s = shift.detach().to(site_o.dtype).view(B, 1, 3).expand_as(site_o)
        eye3 = torch.eye(3, device=site_o.device, dtype=site_o.dtype)
        o_pred = site_o + torch.einsum(
            'bnij,bnj->bni', eye3.view(1, 1, 3, 3) - site_P, s)
    do = (o_pred - to + 0.5) % 1.0 - 0.5
    lo = (do.pow(2) * w).sum() / w.sum().clamp(min=1.0) / 3.0
    return lp + lo

def wyckoff_class_ce(wk_logits, tgt_wyck, sg, valid_lut):
    """Cross-entropy on the Wyckoff class, with unrepresentable targets ignored.
    """
    if wk_logits is None or tgt_wyck is None:
        return None, 0.0
    W = wk_logits.size(-1)
    ok = valid_lut[sg.view(-1).clamp(1, 230).long() - 1]          # (B, W)
    tgt = tgt_wyck.long()
    in_range = (tgt >= 0) & (tgt < W)
    safe = tgt.clamp(0, W - 1)
    keep = in_range & torch.gather(ok, 1, safe)
    n_lab = int((tgt >= 0).sum())
    n_keep = int(keep.sum())
    if n_keep == 0:
        return None, 0.0
    tgt = torch.where(keep, safe, torch.full_like(safe, -100))
    ce = F.cross_entropy(wk_logits.transpose(1, 2), tgt, ignore_index=-100)
    return ce, (n_keep / max(n_lab, 1))

def hardcore_repulsion(frac, L, mask, types, radii_lut, scale=0.70, n_images=2,
                       other=None, precomputed_dist=None, chunk=27):
    if other is not None:
        fb, mb, tb = other
    else:
        fb, mb, tb = frac, mask, types
    if precomputed_dist is not None:
        dist = precomputed_dist
    else:
        _, dist = min_image_disp(frac, fb, L, n_images=n_images, chunk=chunk)
    A = frac.size(1)
    if other is None:
        eye = torch.eye(A, device=frac.device, dtype=torch.bool).unsqueeze(0)
    else:

        F_ = fb.size(1)
        if F_ % A == 0:
            idx_i = torch.arange(A, device=frac.device).view(1, A, 1)
            j_all = torch.arange(F_, device=frac.device).view(1, 1, F_)
            same_slot = (j_all % A) == idx_i
            identity_block = (j_all // A) == 0
            eye = same_slot & identity_block
        else:

            eye = dist < 1e-6
    valid = (mask.bool().unsqueeze(2) & mb.bool().unsqueeze(1)) & ~eye
    r = radii_lut[types.clamp(min=0, max=radii_lut.size(0) - 1)]
    rb = radii_lut[tb.clamp(min=0, max=radii_lut.size(0) - 1)]
    thr = scale * (r.unsqueeze(2) + rb.unsqueeze(1))

    
    viol = F.relu(1.0 - dist / thr.clamp(min=1e-6)) ** 2

    per_b_atoms = mask.float().sum(-1).clamp(min=1.0)
    per_b_viol = (viol * valid.float()).sum(dim=(1, 2))
    return (per_b_viol / per_b_atoms).mean()

_ENEG_VIOL_CACHE = {}

def _eneg_violation_matrix(eneg_lut):
    """relu(EN_i - EN_j) over the element vocabulary, cached per lut tensor."""
    # Keyed on contents, not id(): a freed tensor's id can be reused by another
    # of the same length, silently returning the wrong matrix.
    key = (tuple(eneg_lut.detach().cpu().reshape(-1).tolist()),
           str(eneg_lut.device), eneg_lut.dtype)
    m = _ENEG_VIOL_CACHE.get(key)
    if m is None:
        en = eneg_lut.view(-1)
        m = F.relu(en.unsqueeze(-1) - en.unsqueeze(0))
        _ENEG_VIOL_CACHE[key] = m
    return m

def charge_neutrality_loss(probs, mask, ox_states_padded, ox_states_mask,
                           iters=10, tau=0.1, damp=0.3, w_feas=1.0, w_neut=0.5,
                           return_components=False,
                           eneg_lut=None, metal_lut=None, w_eneg=1.0,
                           include_alloys=True, exempt_unary=True,
                           eneg_tau=0.5):
    """Differentiable relaxation of the FULL SMACT screen, not just charge.

    """
    if probs.dim() == 3:
        c = (probs * mask.unsqueeze(-1)).sum(1)
        n = mask.sum(-1).clamp(min=1)
    else:
        c = probs
        n = c.sum(-1).clamp(min=1)

    ox = ox_states_padded.unsqueeze(0)
    valid = ox_states_mask.unsqueeze(0) > 0
    big = torch.finfo(c.dtype).max
    ox_min = ox.masked_fill(~valid, big).amin(-1).squeeze(0)
    ox_max = ox.masked_fill(~valid, -big).amax(-1).squeeze(0)

    cf = c / n.unsqueeze(-1).clamp(min=1e-6)

    lo = (cf * ox_min.view(1, -1)).sum(-1)
    hi = (cf * ox_max.view(1, -1)).sum(-1)
    feas = F.relu(lo) + F.relu(-hi)

    a = valid.float()
    a = a / a.sum(-1, keepdim=True).clamp(min=1e-8)
    a = a.expand(c.size(0), -1, -1).contiguous()
    o_bar = (a * ox).sum(-1)
    for _ in range(iters):
        R = (cf * o_bar).sum(-1, keepdim=True)
        R_minus = R - cf * o_bar
        new_res = R_minus.unsqueeze(-1) + cf.unsqueeze(-1) * ox
        logit = (-(new_res ** 2) / tau).masked_fill(~valid, -1e9)
        a = damp * a + (1 - damp) * logit.softmax(-1)
        o_bar = (a * ox).sum(-1)
    var = (a * (ox - o_bar.unsqueeze(-1)) ** 2).sum(-1)
    neut = (cf * o_bar).sum(-1) ** 2 + (cf ** 2 * var).sum(-1)

    if eneg_lut is not None and w_eneg > 0:
        en = eneg_lut.view(1, -1).to(c.dtype)
        cn = cf
        cat = torch.sigmoid(o_bar / float(eneg_tau))          # ~1 when o_bar > 0
        ani = torch.sigmoid(-o_bar / float(eneg_tau))         # ~1 when o_bar < 0
        w_pair = (cn * cat).unsqueeze(-1) * (cn * ani).unsqueeze(1)   # (B, i, j)
        viol = F.relu(en.unsqueeze(-1) - en.unsqueeze(1))             # EN_i - EN_j
        eneg = (w_pair * viol).sum(dim=(1, 2))
    else:
        eneg = torch.zeros_like(feas)

    gate = torch.ones_like(feas)
    present = 1.0 - torch.exp(-c.clamp(min=0.0))              # soft 1[count > 0]
    n_elem = present.sum(-1)
    if exempt_unary:
        gate = gate * (n_elem - 1.0).clamp(0.0, 1.0)
    if include_alloys and metal_lut is not None:
        nonmetal = (c * (1.0 - metal_lut.view(1, -1).to(c.dtype))).sum(-1)
        gate = gate * (1.0 - torch.exp(-nonmetal.clamp(min=0.0)))
   
    gate = gate.detach()
    feas = feas * gate
    neut = neut * gate
    eneg = eneg * gate

    total = (w_feas * feas.mean() + w_neut * neut.mean()
             + float(w_eneg) * eneg.mean())
    if return_components:
        return total, feas, neut, eneg
    return total

_LSA = None

def _linear_sum_assignment(cost_np):
    """Minimum-cost one-to-one assignment; scipy when available, greedy else."""
    global _LSA
    if _LSA is None:
        try:
            from scipy.optimize import linear_sum_assignment as _f
            _LSA = _f
        except Exception:
            _LSA = False
    if _LSA:
        import numpy as _np
        r, c = _LSA(cost_np)
        out = _np.arange(cost_np.shape[0])
        out[r] = c
        return out
    import numpy as _np
    C = cost_np.copy()
    B = C.shape[0]
    out = _np.arange(B)
    for _ in range(B):
        i, j = _np.unravel_index(_np.argmin(C), C.shape)
        if not _np.isfinite(C[i, j]):
            break
        out[i] = j
        C[i, :] = _np.inf
        C[:, j] = _np.inf
    return out

@torch.no_grad()
def composition_set_assignment(pred_counts, true_counts, size_weight=0.25,
                               cond_y=None, cond_sg=None, y_weight=0.5,
                               sg_penalty=10.0):
    """Match each GENERATED crystal to a distinct REAL crystal in the batch.

    """
    pf = pred_counts / pred_counts.sum(-1, keepdim=True).clamp(min=1e-6)
    tf = true_counts / true_counts.sum(-1, keepdim=True).clamp(min=1e-6)
    cost = torch.cdist(pf.unsqueeze(0), tf.unsqueeze(0), p=1).squeeze(0)
    if size_weight:
        pn = torch.log(pred_counts.sum(-1).clamp(min=1.0))
        tn = torch.log(true_counts.sum(-1).clamp(min=1.0))
        cost = cost + float(size_weight) * (pn.view(-1, 1) - tn.view(1, -1)).abs()
    
    if cond_y is not None and float(y_weight) > 0:
        cy = cond_y.view(-1).to(cost.dtype)
        cost = cost + float(y_weight) * (cy.view(-1, 1) - cy.view(1, -1)).abs()
    if cond_sg is not None and float(sg_penalty) > 0:
        cs = cond_sg.view(-1).long()
        cost = cost + float(sg_penalty) * (cs.view(-1, 1) != cs.view(1, -1)).to(cost.dtype)
    idx = _linear_sum_assignment(cost.detach().float().cpu().numpy())
    return torch.as_tensor(idx, dtype=torch.long, device=pred_counts.device)



class Graph:
    def __init__(self, gd):
        try:
            self.nodes = np.array(gd['node_features'], dtype=np.int32)
            self.type_counts = np.array(gd['type_counts'], dtype=np.int32)
            self.neighbor_counts = np.array(gd['neighbor_counts'], dtype=np.int32)
            self.neighbors = np.array(gd['neighbors'], dtype=np.int32)
            self.bond_lengths = np.array(gd['bond_lengths'], dtype=np.float32)
            self.cart_coords = np.array(gd['cart_coords'], dtype=np.float32)
            lm = np.array(gd.get('lattice_matrix', np.eye(3)), dtype=np.float32)
            if lm.ndim == 3:
                lm = lm[0] if lm.shape[0] >= 1 else lm.squeeze(0)
            if lm.ndim == 1:
                lm = np.diag(lm) if lm.size == 3 else (lm.reshape(3, 3) if lm.size == 9 else np.eye(3))
            elif lm.ndim == 2 and lm.shape != (3, 3):
                lm = lm.reshape(3, 3) if lm.size == 9 else np.eye(3)
            self.lattice_matrix = lm
            if 'frac_coords' in gd:
                self.frac_coords = np.array(gd['frac_coords'], dtype=np.float32)
            else:
                try:
                    self.frac_coords = (self.cart_coords @ np.linalg.inv(lm)).astype(np.float32)
                    _data_warn("Graph: no 'frac_coords' in record; derived from "
                               "cart_coords @ inv(lattice). dataupgrade.py DOES store "
                               "frac_coords -- an older mine is being used.")
                except Exception as _e:
                    raise ValueError(f"no frac_coords and lattice not invertible: {_e}")
            self.coordination_numbers = np.array(
                gd.get('coordination_numbers', np.zeros(len(self.nodes), dtype=np.int32)), dtype=np.int32)
            n = len(self.nodes)
            raw_wl = gd.get('wyckoff_letters', None)
            raw_wi = gd.get('wyckoff_indices', None)
            if raw_wl is not None:
                wl = np.array(raw_wl)
                if wl.dtype.kind in ('U', 'S', 'O'):
                    wl = np.array([ord(str(c)[0]) for c in wl], dtype=np.uint8)
                wl = wl.astype(np.uint8)
                if len(wl) != n:
                    if len(wl) == 0:
                        wl = np.zeros(n, dtype=np.uint8)
                    else:
                        idx = np.arange(n) % len(wl)
                        wl = wl[idx]
                self.wyckoff_letters = wl
            else:
                self.wyckoff_letters = np.zeros(n, dtype=np.uint8)
            if raw_wi is not None:
                wi = np.array(raw_wi, dtype=np.int32)
                if len(wi) != n:
                    if len(wi) == 0:
                        wi = np.zeros(n, dtype=np.int32)
                    else:
                        idx = np.arange(n) % len(wi)
                        wi = wi[idx]
                self.wyckoff_indices = wi
            else:
                self.wyckoff_indices = np.zeros(n, dtype=np.int32)

            self.space_group = int(np.asarray(
                gd.get('space_group', 1)).ravel()[0]) if gd.get('space_group') is not None else 1

            sym_rot = gd.get('sym_rot', None)
            self.sym_rot = (np.asarray(sym_rot, dtype=np.float32)
                            if sym_rot is not None else np.eye(3, dtype=np.float32)[None])
            sym_trans = gd.get('sym_trans', None)
            self.sym_trans = (np.asarray(sym_trans, dtype=np.float32)
                              if sym_trans is not None else np.zeros((1, 3), dtype=np.float32))

            wproj = gd.get('wyckoff_proj', None)
            self.wyckoff_proj = (np.asarray(wproj, dtype=np.float32)
                                 if wproj is not None
                                 else np.tile(np.eye(3, dtype=np.float32), (n, 1, 1)))
            woff = gd.get('wyckoff_offset', None)
            self.wyckoff_offset = (np.asarray(woff, dtype=np.float32)
                                   if woff is not None else np.zeros((n, 3), dtype=np.float32))

            wmult = gd.get('wyckoff_multiplicity', None)
            self.wyckoff_multiplicity = (np.asarray(wmult, dtype=np.int32)
                                         if wmult is not None else np.ones(n, dtype=np.int32))
            wfree = gd.get('wyckoff_n_free', None)
            self.wyckoff_n_free = (np.asarray(wfree, dtype=np.int8)
                                   if wfree is not None else np.full(n, 3, dtype=np.int8))

            eq = gd.get('equivalent_atoms', None)
            self.equivalent_atoms = (np.asarray(eq, dtype=np.int32)
                                     if eq is not None else np.arange(n, dtype=np.int32))
            aum = gd.get('asym_unit_mask', None)
            self.has_asym_unit_mask = aum is not None
            self.asym_unit_mask = (np.asarray(aum, dtype=bool)
                                   if aum is not None else np.ones(n, dtype=bool))
            if self.has_asym_unit_mask and len(self.asym_unit_mask) != n:
                self.asym_unit_mask = np.ones(n, dtype=bool)
                self.has_asym_unit_mask = False
            self.n_orbits = int(np.asarray(
                gd.get('n_orbits', len(self.equivalent_atoms))).ravel()[0]) \
                if gd.get('n_orbits') is not None else int(len(np.unique(self.equivalent_atoms)))
            _sok = gd.get('symmetry_ok', None)
            self.symmetry_ok = bool(np.asarray(_sok).ravel()[0]) if _sok is not None else None
            self.n_sym_ops = int(np.asarray(self.sym_rot).reshape(-1, 3, 3).shape[0])
        except KeyError as e:
            raise ValueError(f"Missing field: {e}")
        if len(self.nodes) != len(self.cart_coords):
            raise ValueError("Nodes/coords count mismatch")
        if self.cart_coords.shape[1] != 3:
            raise ValueError("Coords must be 3D")
        if len(self.bond_lengths) != len(self.neighbors):
            raise ValueError("bond_lengths/neighbors count mismatch")
        self._load_edges(gd)

    def _load_edges(self, gd):
        ei = gd.get('edge_index', None)
        ei_arr = np.asarray(ei) if ei is not None else None
        if ei_arr is None or ei_arr.ndim != 2 or ei_arr.shape[0] != 2:
            raise ValueError(
                "Graph record has no valid 'edge_index'. Regenerate the dataset "
                "with the current dataupgrade.py (it saves edge_index / edge_attr / "
                "edge_offsets / edge_types).")
        ei_arr = ei_arr.astype(np.int64)
        bl = np.asarray(gd.get('bond_lengths', self.bond_lengths), dtype=np.float32)
        ea = gd.get('edge_attr', None)
        ea = (np.asarray(ea, dtype=np.float32)
              if ea is not None and np.asarray(ea).ndim == 2 and np.asarray(ea).shape[1] >= 2
              else None)
        et = gd.get('edge_types', None)
        if et is None:
            et = ea[:, 1] if ea is not None else np.zeros(ei_arr.shape[1], dtype=np.float32)
        et = np.asarray(et, dtype=np.float32)
        if ea is None:
            ea = np.column_stack([bl, et]).astype(np.float32)
        off = gd.get('edge_offsets', None)
        if off is not None and np.asarray(off).size:
            off = np.asarray(off, dtype=np.float32).reshape(-1, 3)
        else:
            off = np.zeros((ei_arr.shape[1], 3), dtype=np.float32)
        if ei_arr.shape[1] > 0:

            same = ei_arr[0] == ei_arr[1]
            if len(off) == ei_arr.shape[1]:
                periodic = np.abs(off).sum(axis=1) > 1e-6
            else:
                periodic = np.zeros(ei_arr.shape[1], dtype=bool)
            keep = (~same) | periodic
            n_self_kept = int((same & periodic).sum())
            if n_self_kept:
                _data_warn(f"kept {n_self_kept} periodic self-edge(s) per record "
                           f"(i -> i + t); these carry the lattice vectors.")
            if not bool(keep.all()):
                ei_arr = ei_arr[:, keep]
                ea, bl, et = ea[keep], bl[keep], et[keep]
                off = off[keep] if len(off) == keep.shape[0] else off
        self.edge_index = torch.tensor(ei_arr, dtype=torch.long).contiguous()
        self.edge_attr = torch.tensor(ea, dtype=torch.float32)
        self.edge_offsets = off.astype(np.float32)
        self.bond_lengths = bl
        self.edge_types = et.astype(np.int32)

class CrystalData(Data):
    def __cat_dim__(self, key, value, *args, **kwargs):
        if key in ('wyckoff_indices', 'wyckoff_letters', 'frac_coords',
                   'pos', 'x', 'coordination'):
            return 0
        if key in ('space_group', 'n_orbits', 'sym_rot', 'sym_trans',
                   'n_sym_ops', 'symmetry_ok',
                   'wyckoff_proj', 'wyckoff_offset', 'wyckoff_multiplicity',
                   'wyckoff_n_free', 'equivalent_atoms', 'asym_unit_mask'):
            return 0
        return super().__cat_dim__(key, value, *args, **kwargs)

    def __inc__(self, key, value, *args, **kwargs):
        if key in ('wyckoff_indices', 'wyckoff_letters',
                   'coordination', 'space_group', 'lattice_matrix'):
            return 0
        if key in ('n_orbits', 'sym_rot', 'sym_trans', 'n_sym_ops', 'symmetry_ok'):
            return 0
        if key in ('wyckoff_proj', 'wyckoff_offset', 'wyckoff_multiplicity',
                   'wyckoff_n_free', 'equivalent_atoms', 'asym_unit_mask'):
            return 0

        if key in ('frac_coords', 'pos', 'x', 'edge_offsets', 'edge_attr'):
            return 0
        return super().__inc__(key, value, *args, **kwargs)

class MultiFileGraphDataset(Dataset):
    def __init__(self, base_path, target_name, split, global_atomic_to_idx=None,
                 global_node_vectors=None, n_sites=20,
                 max_atoms: int = None):
        super().__init__()
        self.split = split
        self.path = os.path.join(base_path, split)
        self.target_name = target_name
        self.global_atomic_to_idx = global_atomic_to_idx
        self.global_node_vectors = global_node_vectors
        self._n_sites = int(n_sites)
        self._max_atoms = max_atoms
        self._load_config()
        self._load_graph_data()
        self._apply_max_atoms_filter()
        self._load_targets()
        if len(self.graph_data) != len(self.targets):
            raise ValueError("Graph/target count mismatch")
        self._check_asymmetric_unit()

    def _check_asymmetric_unit(self):
       
        n_asym, n_full, missing = [], [], 0
        for g in self.graph_data:
            if not getattr(g, 'has_asym_unit_mask', False):
                missing += 1
            n_asym.append(int(np.asarray(g.asym_unit_mask).sum()))
            n_full.append(int(len(g.nodes)))
        n_asym = np.asarray(n_asym)
        n_full = np.asarray(n_full)
        sg = np.asarray([int(getattr(g, 'space_group', 1)) for g in self.graph_data])
        self.n_asym_atoms = n_asym
        if missing:
            _data_warn(
                f"[{self.split}] {missing}/{len(n_asym)} records have NO "
                f"'asym_unit_mask'; it silently defaults to all-ones. With "
                f"decode_asymmetric_unit=True the decoder is then trained on "
                f"FULL cells while expand_generated still applies |G| symmetry "
                f"operations at sample time -- every generated cell comes out "
                f"|G| times too dense. Re-mine the dataset or set "
                f"decode_asymmetric_unit=False.")
        sym = sg > 1
        if sym.any():
            ratio = float((n_asym[sym] == n_full[sym]).mean())
            if ratio > 0.9:
                _data_warn(
                    f"[{self.split}] {100 * ratio:.0f}% of non-P1 crystals have "
                    f"asym_unit == full cell. The asymmetric unit is not being "
                    f"reduced; see the note above.")
        try:
            _rat = []
            for g in self.graph_data:
                _mu = np.asarray(getattr(g, 'wyckoff_multiplicity', None),
                                 dtype=np.float64).reshape(-1)
                _am = np.asarray(g.asym_unit_mask, dtype=bool).reshape(-1)
                if _mu.size == _am.size and _am.any():
                    _rat.append(float(_mu[_am].sum()) / max(len(g.nodes), 1))
            if _rat:
                _rat = np.asarray(_rat)
                _med = float(np.median(_rat))
                print(f"  [{self.split}] sum(Wyckoff multiplicity over asym unit)"
                      f" / atoms in cell: median {_med:.2f}  "
                      f"[{float(np.percentile(_rat, 5)):.2f}, "
                      f"{float(np.percentile(_rat, 95)):.2f}]")
                if abs(_med - 1.0) > 0.15:
                    _data_warn(
                        f"[{self.split}] that ratio should be 1.00. At {_med:.2f} "
                        f"the multiplicities and the stored cells use different "
                        f"conventions -- most often conventional-cell "
                        f"multiplicities against primitive cells, where the "
                        f"ratio is the centring factor. Cell VOLUME is built "
                        f"from these multiplicities, so every generated "
                        f"structure comes out ~{_med:.1f}x too dense and the "
                        f"vpa loss cannot fix it.")
        except Exception:
            pass

        cap = int(self._n_sites)
        over = int((n_asym > cap).sum())
        if over:
            _data_warn(
                f"[{self.split}] {over}/{len(n_asym)} crystals have an "
                f"asymmetric unit larger than n_sites = {cap} sites "
                f"(max {int(n_asym.max())}); the decoder physically cannot "
                f"represent them and their reconstruction error is a floor. "
                f"Raise n_sites or use dataset_max_atoms_filter.")
        print(f"  [{self.split}] asym-unit atoms: mean {n_asym.mean():.1f}  "
              f"max {int(n_asym.max())}  (decoder capacity {cap})")
        _mx = int(n_asym.max())
        if over == 0 and cap > _mx + 4:
            print(f"      n_sites={cap} exceeds the largest asymmetric unit in "
                  f"THIS split ({_mx}); {cap - _mx} slots can never hold a real "
                  f"site here. n_sites must cover the maximum across ALL "
                  f"splits, so only lower it once train, val and test have all "
                  f"reported \u2014 anything above the global maximum is spare "
                  f"capacity, anything below it truncates silently.")

    def _load_config(self):
        with open(os.path.join(self.path, f"{self.split}_config.json")) as f:
            cfg = json.load(f)
        if self.global_atomic_to_idx is not None:
            self.atomic_numbers = sorted(self.global_atomic_to_idx.keys())
            self.node_vectors = self.global_node_vectors
            self.atomic_to_idx = self.global_atomic_to_idx
        else:
            self.atomic_numbers = cfg["atomic_numbers"]
            self.node_vectors = np.array(cfg["node_vectors"])
            self.atomic_to_idx = {num: idx for idx, num in enumerate(self.atomic_numbers)}
        n_types = len(self.atomic_numbers)
        onehot = np.eye(n_types, dtype=np.float32)
        self._onehot_vectors = onehot

        if self.global_node_vectors is not None:
            gv = np.asarray(self.global_node_vectors, dtype=np.float32)
            if gv.shape[0] != n_types:
                raise ValueError(
                    f"global_node_vectors has {gv.shape[0]} rows but the global "
                    f"vocabulary has {n_types} elements.")
            self.node_vectors = gv
            return

        elem_props = cfg.get("element_properties", {})
        self.node_vectors = np.concatenate(
            [onehot, build_element_phys_features(self.atomic_numbers, elem_props)], axis=1)

    def _load_graph_data(self):
        with np.load(os.path.join(self.path, f"{self.split}.npz"), allow_pickle=True) as data:
            gd = data['graph_dict'].item()
            self.graph_data, self.graph_names = [], []
            for name, g in gd.items():
                try:
                    if 'cart_coords' not in g:
                        raise ValueError(f"Missing cart_coords in {name}")
                    self.graph_data.append(Graph(g))
                    self.graph_names.append(name)
                except ValueError as e:
                    print(f"Skipping {name}: {e}")
        if not self.graph_data:
            raise RuntimeError("No valid graphs loaded")

    def _apply_max_atoms_filter(self):
        if self._max_atoms is None:
            return
        kept_data, kept_names = [], []
        n_dropped = 0
        for g, name in zip(self.graph_data, self.graph_names):
            n_atoms = len(g.nodes)
            if n_atoms <= self._max_atoms:
                kept_data.append(g)
                kept_names.append(name)
            else:
                n_dropped += 1
        self.graph_data = kept_data
        self.graph_names = kept_names
        if n_dropped > 0:
            print(f"  [{self.split}] max_atoms={self._max_atoms}: dropped "
                  f"{n_dropped} crystals with >{self._max_atoms} atoms "
                  f"({len(self.graph_data)} remain).")
        if not self.graph_data:
            raise RuntimeError(
                f"max_atoms={self._max_atoms} filtered out every crystal in "
                f"the {self.split} split. Increase max_atoms or disable the filter.")

    def _load_targets(self):
        df = pd.read_csv(os.path.join(self.path, f"{self.split}.csv"))
        if self.target_name not in df.columns:
            raise ValueError(f"Target '{self.target_name}' not in CSV")
        if 'material_id' in df.columns:
            df_idx = df.set_index('material_id')
            missing = [n for n in self.graph_names if n not in df_idx.index]
            if missing:
                raise ValueError(f"{len(missing)} graphs missing from CSV. First: {missing[0]}")
            aligned = df_idx.loc[self.graph_names]
            self.targets = aligned[self.target_name].values
            self.material_ids = aligned.index.values
            sg_col = ('space_group' if 'space_group' in aligned.columns else
                      'spacegroup' if 'spacegroup' in aligned.columns else
                      'spacegroup.number' if 'spacegroup.number' in aligned.columns else
                      None)
            csv_sg = aligned[sg_col].values.astype(np.int32) \
                                if sg_col is not None \
                                else np.ones(len(self.targets), dtype=np.int32)
        else:
            self.targets = df[self.target_name].values
            self.material_ids = np.array([f'mat_{i}' for i in range(len(df))])
            csv_sg = np.ones(len(self.targets), dtype=np.int32)

        mined_sg = np.array([int(getattr(g, 'space_group', 1)) for g in self.graph_data],
                            dtype=np.int32)
        sym_ok = np.array(
            [(1 if getattr(g, 'symmetry_ok', None) else 0) for g in self.graph_data],
            dtype=bool)
        has_flag = np.array(
            [getattr(g, 'symmetry_ok', None) is not None for g in self.graph_data],
            dtype=bool)
        use_mined = np.where(has_flag, sym_ok, mined_sg > 1)
        self.space_groups = np.where(use_mined, mined_sg, csv_sg).astype(np.int32)
        n_p1 = int((self.space_groups == 1).sum())
        if n_p1 > 0.5 * len(self.space_groups):
            _data_warn(
                f"[{self.split}] {100.0 * n_p1 / max(len(self.space_groups), 1):.0f}% of "
                f"crystals are P1. Symmetry-aware generation will be weak; check "
                f"symmetry_stats.p1_rate in the split config.")

    def __len__(self): return len(self.graph_data)

    def __getitem__(self, idx):
        g = self.graph_data[idx]
        ai = np.array([self.atomic_to_idx[int(n)] for n in g.nodes])
        return CrystalData(
            x=torch.tensor(self.node_vectors[ai], dtype=torch.float32),
            edge_index=g.edge_index,
            edge_attr=g.edge_attr,
            pos=torch.tensor(g.cart_coords, dtype=torch.float32),
            y=torch.tensor([self.targets[idx]], dtype=torch.float32),
            material_id=self.material_ids[idx],
            lattice_matrix=torch.tensor(g.lattice_matrix, dtype=torch.float32),
            frac_coords=torch.tensor(g.frac_coords, dtype=torch.float32),
            coordination=torch.tensor(g.coordination_numbers, dtype=torch.long),
            space_group=torch.tensor([self.space_groups[idx]], dtype=torch.long),
            wyckoff_letters=torch.tensor(g.wyckoff_letters.astype(np.int32), dtype=torch.long),
            wyckoff_indices=torch.tensor(g.wyckoff_indices.astype(np.int32), dtype=torch.long),
            edge_offsets=torch.tensor(g.edge_offsets, dtype=torch.float32),
            sym_rot=torch.tensor(np.asarray(g.sym_rot).reshape(-1, 3, 3), dtype=torch.float32),
            sym_trans=torch.tensor(np.asarray(g.sym_trans).reshape(-1, 3), dtype=torch.float32),
            n_sym_ops=torch.tensor([int(g.n_sym_ops)], dtype=torch.long),
            symmetry_ok=torch.tensor(
                [0 if getattr(g, 'symmetry_ok', None) is False else 1],
                dtype=torch.long),
            n_orbits=torch.tensor([int(g.n_orbits)], dtype=torch.long),
            wyckoff_proj=torch.tensor(np.asarray(g.wyckoff_proj), dtype=torch.float32),
            wyckoff_offset=torch.tensor(np.asarray(g.wyckoff_offset), dtype=torch.float32),
            wyckoff_multiplicity=torch.tensor(np.asarray(g.wyckoff_multiplicity), dtype=torch.long),
            wyckoff_n_free=torch.tensor(np.asarray(g.wyckoff_n_free).astype(np.int64), dtype=torch.long),
            equivalent_atoms=torch.tensor(np.asarray(g.equivalent_atoms), dtype=torch.long),
            asym_unit_mask=torch.tensor(np.asarray(g.asym_unit_mask), dtype=torch.bool),
        )

def _pyg_collate(batch):
    return Batch.from_data_list(batch)

def _pyg_batch_to_dict(batch, device, num_types=None):
    bvec = batch.batch.to(device)
    B = int(batch.num_graphs)

    if batch.x.dim() == 2 and num_types is not None and batch.x.size(1) >= num_types:
        tidx = batch.x[:, :num_types].argmax(dim=1).long()
    else:
        tidx = (batch.x.argmax(dim=1) if batch.x.dim() == 2 else batch.x).long()
    tidx = tidx.to(device)

    frac_flat = (batch.frac_coords if batch.frac_coords.dim() == 2
                 else batch.frac_coords.reshape(-1, 3)).to(device)
    frac, mask = to_dense_batch(frac_flat, bvec)
    types, _ = to_dense_batch(tidx.unsqueeze(-1), bvec, fill_value=0)
    types = types.squeeze(-1)

    L = batch.lattice_matrix.reshape(B, 3, 3).to(device)
    y = batch.y.reshape(B).to(device)
    sg = batch.space_group.reshape(B).to(device)

    def _densify(attr, fill):
        t = getattr(batch, attr, None)
        if t is None or not torch.is_tensor(t):
            return None
        t = t.to(device)
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        d, _ = to_dense_batch(t, bvec, fill_value=fill)
        return d

    sym_ops = None
    _nops = getattr(batch, 'n_sym_ops', None)
    _srot = getattr(batch, 'sym_rot', None)
    _stra = getattr(batch, 'sym_trans', None)
    if (_nops is not None and _srot is not None and _stra is not None
            and torch.is_tensor(_srot) and _srot.numel()):
        counts = _nops.reshape(-1).tolist()
        rot_all = _srot.reshape(-1, 3, 3).cpu()
        tr_all = _stra.reshape(-1, 3).cpu()
        if int(sum(counts)) == int(rot_all.size(0)) and len(counts) == B:
            sym_ops, off_i = [], 0
            for k in counts:
                k = int(k)
                sym_ops.append((rot_all[off_i:off_i + k], tr_all[off_i:off_i + k]))
                off_i += k

    wp = _densify('wyckoff_proj', 0.0)
    wo = _densify('wyckoff_offset', 0.0)
    wn = _densify('wyckoff_n_free', 3)
    wm = _densify('wyckoff_multiplicity', 1)

    wk = _densify('wyckoff_indices', -100)
    eq = _densify('equivalent_atoms', 0)
    au = _densify('asym_unit_mask', 1)

    ptr = batch.ptr.to(device)
    ei = batch.edge_index.to(device)
    src_g, dst_g = ei[0], ei[1]
    if getattr(batch, 'edge_offsets', None) is not None and batch.edge_offsets.numel():
        off = batch.edge_offsets.to(device).reshape(-1, 3).float()
    else:
        off = torch.zeros(src_g.size(0), 3, device=device)
    be = bvec[src_g]
    src_l = src_g - ptr[be]
    dst_l = dst_g - ptr[be]

    cnt = torch.bincount(be, minlength=B)
    E = max(int(cnt.max().item()) if cnt.numel() else 1, 1)
    eptr = torch.cat([cnt.new_zeros(1), cnt.cumsum(0)])
    rank = torch.arange(be.size(0), device=device) - eptr[be]

    src_pad = torch.zeros(B, E, dtype=torch.long, device=device)
    dst_pad = torch.zeros(B, E, dtype=torch.long, device=device)
    off_pad = torch.zeros(B, E, 3, device=device)
    emask = torch.zeros(B, E, dtype=torch.bool, device=device)
    if be.numel():
        src_pad[be, rank] = src_l
        dst_pad[be, rank] = dst_l
        off_pad[be, rank] = off
        emask[be, rank] = True

    return {'types': types, 'frac': frac, 'lattice': L, 'mask': mask,
            'y': y, 'sg': sg, 'src': src_pad, 'dst': dst_pad,
            'offset': off_pad, 'emask': emask,
            'wyckoff_proj': wp, 'wyckoff_offset': wo, 'wyckoff_n_free': wn,
            'wyckoff_multiplicity': wm, 'wyckoff_index': wk,
            'equivalent_atoms': eq,
            'asym_unit_mask': au,
            'sym_ops': sym_ops}

def to_device(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

def _asym_unit_mask_from_batch(batch, atom_mask):
    """Build a per-atom asymmetric-unit boolean mask, robust to the dataset's
    asym_unit_mask field not actually being one-entry-per-full-cell-atom.

    """
    A = atom_mask.size(1)
    aum = batch.get('asym_unit_mask', None)
    if aum is not None:
        aum = aum.squeeze(-1) if aum.dim() == 3 else aum
    if aum is not None and aum.shape[:2] == atom_mask.shape[:2]:
        return atom_mask & aum.bool()
    eq = batch.get('equivalent_atoms', None)
    if eq is not None:
        eq = eq.squeeze(-1) if eq.dim() == 3 else eq
    if eq is not None and eq.shape[:2] == atom_mask.shape[:2]:
        local_idx = torch.arange(A, device=atom_mask.device).view(1, A)
        return atom_mask & eq.eq(local_idx)

    return atom_mask



def _cosine_lr(opt, base, ep, total, warmup=0, min_frac=0.05):
    if warmup and ep < warmup:
        frac = (ep + 1) / warmup
    else:
        prog = (ep - warmup) / max(1, total - warmup - 1)
        prog = min(1.0, max(0.0, prog))
        frac = min_frac + 0.5 * (1 - min_frac) * (1 + math.cos(math.pi * prog))
   .
    for g in opt.param_groups:
        if '_base_lr' not in g:
            g['_base_lr'] = g['lr']
        g['lr'] = g['_base_lr'] * frac
    return base * frac

def _warmup_cosine_lr(opt, base, step, total_steps, warmup_steps=0, min_frac=0.05):
    """Per-STEP linear warmup then cosine decay, applied to every param group by
    ITS OWN base lr (stage 3's two groups sit at different lrs on purpose).
    """
    warmup_steps = int(max(0, warmup_steps))
    if warmup_steps and step < warmup_steps:
        frac = (step + 1) / warmup_steps
    else:
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps - 1)
        prog = min(1.0, max(0.0, prog))
        frac = min_frac + 0.5 * (1 - min_frac) * (1 + math.cos(math.pi * prog))
    for g in opt.param_groups:
        if '_base_lr' not in g:
            g['_base_lr'] = g['lr']
        g['lr'] = g['_base_lr'] * frac
    return base * frac

def _prog(it, desc=""):
    try:
        from tqdm.auto import tqdm
        return tqdm(it, desc=desc)
    except Exception:
        return it

def _init_metrics_csv(path, fieldnames):
    if not os.path.exists(path):
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

def _append_metrics_csv(path, row):
    with open(path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writerow(row)

def save_checkpoint(path, model, extra=None):
    torch.save({'model': model.state_dict(), 'config': model.cfg.__dict__,
                **(extra or {})}, path)

def load_checkpoint(path, model, map_location='cpu', strict=True):
    ck = torch.load(path, map_location=map_location, weights_only=False)
    try:
        _ckcfg = ck.get('config', None)
        if isinstance(_ckcfg, dict):
            pass
    except Exception:
        pass
    incompatible = model.load_state_dict(ck['model'], strict=strict)
    if not strict and incompatible is not None:
        miss = list(getattr(incompatible, 'missing_keys', []))
        extra = list(getattr(incompatible, 'unexpected_keys', []))
        if miss or extra:
            _data_warn(f"load_checkpoint(strict=False): {len(miss)} missing, "
                       f"{len(extra)} unexpected keys "
                       f"(first missing: {miss[:1]}, first unexpected: {extra[:1]}).",
                       once=False)
    return ck

def pretrain_encoder_losses(model, batch):
    """Autoencoder objective for the GNN encoder, site decoder and structural heads.
    """
    cfg = model.cfg
    frac, L = batch['frac'], batch['lattice']

    r = model.reconstruct(batch, encode_grad=True)
    dec, pm, tm, pf = r['dec'], r['pm'], r['tm'], r['pf']
    z_dec, keep = r['z_dec'], r['keep']

    losses = {}
    losses['chamfer'] = periodic_chamfer(pf, frac, L, pm, tm, pw=dec['occ_soft'])

    losses['type'] = type_loss_nn(dec['type_logits'], batch['types'], pf, frac, L,
                                  pm & ~keep, tm,
                                  formability_mask=model.formability_mask,
                                  reverse_weight=float(getattr(
                                      cfg, 'type_reverse_weight', 0.0)))
    st, sp = site_alignment_losses(dec, r['pf_site'], keep, r['tgt_types'],
                                   r['tgt_frac'], L)
    losses['site_type'], losses['site_pos'] = st, sp

    kf = keep.to(z_dec.dtype)
    losses['count'] = F.smooth_l1_loss(dec['n_sites_soft'],
                                       kf.sum(-1).clamp(min=1.0), beta=1.0)
    losses['mult'] = ((F.smooth_l1_loss(r['mult_pred'], r['tgt_mult'], beta=1.0,
                                        reduction='none') * kf).sum()
                      / kf.sum().clamp(min=1.0))

    y_pred, _ = model.property(z_dec, dec['occ_soft'].detach())
    losses['property'] = F.smooth_l1_loss(
        y_pred.squeeze(-1), model.ynorm(batch['y']), beta=0.5)

    true_params = matrix_to_lattice_params(L.detach())
    pred_params = matrix_to_lattice_params(r['L_pred'])
    losses['lattice'] = (
        F.smooth_l1_loss(torch.log(pred_params[:, :3].clamp(min=1e-2)),
                         torch.log(true_params[:, :3].clamp(min=1e-2)), beta=0.3)
        + F.smooth_l1_loss(pred_params[:, 3:] / 180.0,
                           true_params[:, 3:] / 180.0, beta=0.1)
        + F.smooth_l1_loss(
            torch.log(r['V_pred'].squeeze(-1).clamp(min=1e-2)),
            torch.log(torch.linalg.det(L.detach()).abs().clamp(min=1e-2)), beta=0.3))

    _wl = dec.get('wyckoff_logits')
    _wce, _wcov = wyckoff_class_ce(_wl, r['tgt_wyck'], batch['sg'],
                                   model.decoder.wyckoff_valid)
    if _wce is not None:
        losses['wyckoff_ce'] = _wce

        losses['wyckoff_repr'] = torch.as_tensor(
            _wcov, dtype=torch.float32, device=frac.device)
        with torch.no_grad():
            _m = r['tgt_wyck'] >= 0
            losses['wyckoff_acc'] = ((_wl.argmax(-1) == r['tgt_wyck']) & _m
                                     ).float().sum() / _m.sum().clamp(min=1)

            _kf = r['keep'].to(torch.float32)
            losses['wyckoff_matched'] = (_m.to(torch.float32) * _kf).sum() / \
                _kf.sum().clamp(min=1)

    if cfg.weights.get('wyckoff', 0.0) > 0:
        losses['wyckoff'] = wyckoff_proj_loss(
            dec['site_P'], dec['site_o'], pf, frac,
            batch.get('wyckoff_proj'), batch.get('wyckoff_offset'),
            L, pm, tm, 1, shift=r['shift'])

    losses['sg'] = F.cross_entropy(model.sg_coarse(z_dec),
                                   batch['sg'].view(-1).clamp(1, 230) - 1,
                                   weight=model.sg_class_weights)
    _sg_pos = model.sg_pos_encoder(model._sg_pos_feat(
        dec['frac'], L, dec['mask'].bool(), detach=True))
    losses['sg_fine'] = F.cross_entropy(
        model.sg(z_dec.detach(), pos_feat=_sg_pos),
        batch['sg'].view(-1).clamp(1, 230) - 1,
        weight=model.sg_class_weights)

    with torch.no_grad():

        losses['chamfer_gated'] = periodic_chamfer(
            pf, frac, L, pm, tm, pw=dec['occ_soft'], cover_all_slots=False)
        losses['type_conf'] = ((dec['type_probs'].max(-1).values * dec['mask']).sum()
                               / dec['mask'].sum().clamp(min=1.0))
        losses['n_sites_pred'] = dec['mask'].sum(1).mean()
        losses['n_sites_true'] = kf.sum(-1).mean()

    w = cfg.weights
    total = (w.get('chamfer', 1.0) * losses['chamfer']
             + w.get('type', 1.0) * losses['type']
             + w.get('site_type', 1.0) * losses['site_type']
             + w.get('site_pos', 1.0) * losses['site_pos']
             + w.get('count', 1.0) * losses['count']
             + w.get('mult', 0.5) * losses['mult']
             + w.get('property', 1.0) * losses['property']
             + w.get('lattice', 1.0) * losses['lattice']
             + w.get('sg', 1.0) * losses['sg']
             + w.get('sg_fine', 1.0) * losses.get('sg_fine', 0.0)
             + w.get('wyckoff', 0.5) * losses.get('wyckoff', 0.0)
             + w.get('wyckoff_ce', 1.0) * losses.get('wyckoff_ce', 0.0))
    return total, {k: float(v.detach()) for k, v in losses.items()}

def nonfinite_state(model, limit=12):
   
    bad_p, bad_b = [], []
    with torch.no_grad():
        for n, p in model.named_parameters():
            if not torch.isfinite(p).all():
                bad_p.append(n)
        for n, b in model.named_buffers():
            if b.is_floating_point() and not torch.isfinite(b).all():
                bad_b.append(n)
    return bad_p[:limit], bad_b[:limit]


def _state_report(model):
    bad_p, bad_b = nonfinite_state(model)
    if not bad_p and not bad_b:
        return ("all parameters and buffers are finite, so the loss is "
                "non-finite because of the forward pass on this data, not "
                "because the weights are already dead")
    parts = []
    if bad_p:
        parts.append("non-finite PARAMETERS: " + ", ".join(bad_p))
    if bad_b:
        parts.append("non-finite BUFFERS: " + ", ".join(bad_b))
    return "; ".join(parts)


def run_encoder_pretrain(model, train_loader, val_loader=None, epochs=20,
                         lr=1e-3, clip=1.0, device='cpu', log_every=1,
                         weight_decay=1e-5, patience=5, ckpt=None,
                         monitor=('chamfer_gated', 'site_pos', 'site_type',
                                  'lattice', 'count', 'wyckoff_ce'),
                         metrics_path=None, skip_tripwire=50):
    """Pretrain the autoencoder and the structural heads, then freeze.

    """
    model.to(device)
    model.train()
    params = [p for n, p in model.named_parameters()
              if p.requires_grad and not n.startswith('flow.')
              and not n.startswith('property_frozen.')]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


    if getattr(model.decoder, 'use_wyckoff', False):
        if bool(getattr(model.decoder, '_wyckoff_from_data', torch.tensor(False))):
            print("  Wyckoff codebook: mined from the dataset "
                  "(class indices match the annotations).")
        else:
            _data_warn(
                "The decoder is using the ENUMERATED Wyckoff table. Its class "
                "indices come from this file's own enumeration order, which is "
                "not the crystallographic letter order that the dataset writes "
                "into `wyckoff_indices`, so most class labels will be dropped "
                "by wyckoff_class_ce and the Wyckoff head will barely train. "
                "Call model.decoder.install_wyckoff_codebook([train, val, test]) "
                "first -- main() already does.")


    try:
        _b = next(iter(train_loader))
        if not isinstance(_b, dict):
            _b = _pyg_batch_to_dict(_b.to(device), device, num_types=model.cfg.num_types)
        else:
            _b = to_device(_b, device)
        _tm = (_asym_unit_mask_from_batch(_b, _b['mask'].bool())
               if getattr(model.cfg, 'decode_asymmetric_unit', True) else _b['mask'].bool())
        _A = model.cfg.n_sites
        _pf = torch.rand(_b['frac'].size(0), _A, 3, device=device)
        _pm = (torch.arange(_A, device=device).view(1, -1)
               < _tm.sum(1, keepdim=True).clamp(min=1))
        _rand = float(periodic_chamfer(_pf, _b['frac'], _b['lattice'], _pm, _tm))
        # Same idea for the per-site term. Without this the only way to judge
        # site_pos is to guess the chance level from the cell size, which is
        # how an earlier read of a training log came to call a stalled value
        # "exactly at chance" on an assumed cell edge. Measure it.
        _keep, _tt, _tf, _tmul, _tP, _to, _tw = _canonical_site_targets(
            _b, model.cfg.n_sites,
            decode_asym=bool(getattr(model.cfg, 'decode_asymmetric_unit', True)))
        _kf = _keep.to(_pf.dtype)
        _df = (_pf - _tf + 0.5) % 1.0 - 0.5
        _d = safe_norm(torch.einsum('bij,bjk->bik', _df, _b['lattice']))
        _rand_pos = float((F.smooth_l1_loss(_d, torch.zeros_like(_d), beta=0.5,
                                            reduction='none') * _kf).sum()
                          / _kf.sum().clamp(min=1.0))
        print(f"  Reference: uniformly random coordinates give site_pos "
              f"= {_rand_pos:.4f} on this data.")
        print(f"  Reference: uniformly random coordinates give chamfer "
              f"= {_rand:.2f} \u00c5 on this data. Validation chamfer must fall "
              f"well below it or the coordinate head is not learning.")
    except Exception as _e:
        print(f"  [warn] could not compute the random-chamfer reference: {_e}")

    W = model.cfg.weights
    mon_keys = tuple(monitor) if monitor else None

    def _monitor(comps):
        if not mon_keys:
            return None
        got = [k for k in mon_keys if k in comps]
        return sum(W.get(k, 1.0) * comps[k] for k in got) if got else None

    def _mean(ds):
        if not ds:
            return {}
        ks = set().union(*[d.keys() for d in ds])
        return {k: float(np.mean([d[k] for d in ds if k in d])) for k in ks}

    def _tripwire(mdl, run_skip, ep, limit):
        """Fail as soon as the run is provably dead rather than at the end of
        an epoch of skipped batches."""
        if not limit or run_skip < limit:
            return
        raise RuntimeError(
            f"encoder pretraining epoch {ep+1}: {run_skip} consecutive batches "
            f"produced a non-finite loss or gradient. Diagnosis: "
            f"{_state_report(mdl)}.")

    best, best_state, best_ep, no_improve, ep = float('inf'), None, -1, 0, 0
    hdr_done = False
    for ep in range(epochs):
        model.train()
        tl, tc = [], []
        bar = _prog(train_loader, f"encoder-pretrain {ep+1}/{epochs}")
        n_skip, run_skip = 0, 0
        for batch in bar:
            if isinstance(batch, dict):
                bdict = to_device(batch, device)
            else:
                batch = batch.to(device)
                bdict = _pyg_batch_to_dict(batch, device, num_types=model.cfg.num_types)
            opt.zero_grad()
            loss, comps = pretrain_encoder_losses(model, bdict)
            if not torch.isfinite(loss):
                opt.zero_grad(); n_skip += 1; run_skip += 1
                _tripwire(model, run_skip, ep, skip_tripwire)
                continue
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(params, clip)
            if not torch.isfinite(gn):
                opt.zero_grad(); n_skip += 1; run_skip += 1
                _tripwire(model, run_skip, ep, skip_tripwire)
                continue
            opt.step()
            model.decoder.advance_wyckoff()      # once per step, not per forward
            run_skip = 0
            tl.append(loss.item()); tc.append(comps)
            if hasattr(bar, 'set_postfix'):
                bar.set_postfix(loss=f"{sum(tl)/len(tl):.4f}")

        # An epoch with no usable batch is a dead run, not a zero loss. The
        # old code averaged over max(len,1) and printed 0.0000, which the
        # scoring below then read as the best epoch so far.
        if not tl:
            raise RuntimeError(
                f"encoder pretraining epoch {ep+1} completed {n_skip} batches "
                f"and every one of them produced a non-finite loss or gradient, "
                f"so no optimiser step was taken. Diagnosis: {_state_report(model)}.")
        if n_skip:
            print(f"  [warn] epoch {ep+1}: skipped {n_skip} batch(es) on a "
                  f"non-finite loss or gradient")
        _rej = int(getattr(model.enc_standardizer, '_rejected', 0))
        if _rej:
            print(f"  [warn] the latent standardiser refused {_rej} statistics "
                  f"update(s) as non-finite; the latent is exploding somewhere "
                  f"upstream even though the run is still alive")
            model.enc_standardizer._rejected = 0

        tr, tr_c = float(sum(tl) / len(tl)), _mean(tc)

        vl, vl_c, val_ok = tr, tr_c, (val_loader is None)
        if val_loader is not None:
            model.eval()
            vls, vcs = [], []
            with torch.no_grad():
                for batch in val_loader:
                    if isinstance(batch, dict):
                        bdict = to_device(batch, device)
                    else:
                        batch = batch.to(device)
                        bdict = _pyg_batch_to_dict(batch, device,
                                                   num_types=model.cfg.num_types)
                    l, c = pretrain_encoder_losses(model, bdict)
                    if torch.isfinite(l):
                        vls.append(l.item()); vcs.append(c)
            if vls:
                vl, vl_c, val_ok = float(sum(vls) / len(vls)), _mean(vcs), True
            else:
                # Do NOT silently fall back to the training numbers: that is
                # what made a failed validation look like a perfect train/val
                # match in the log.
                print(f"  [warn] epoch {ep+1}: every validation batch gave a "
                      f"non-finite loss, so the row below is the TRAINING "
                      f"average, and this epoch cannot be selected as best. "
                      f"Diagnosis: {_state_report(model)}")

        mon_vl = _monitor(vl_c) if val_ok else None
        score = mon_vl if mon_vl is not None else vl
        if not val_ok or not np.isfinite(score):
            score = float('inf')

        if (ep + 1) % log_every == 0 or ep == epochs - 1:
            keys = sorted(vl_c.keys())
            if not hdr_done:
                print("    " + "  ".join(f"{k:>8}" for k in
                                         ["epoch", "train", "val", "MONITOR"] + keys))
                hdr_done = True
            row = [f"{ep+1:>8d}", f"{tr:>8.4f}", f"{vl:>8.4f}",
                   f"{(mon_vl if mon_vl is not None else float('nan')):>8.4f}"]
            row += [f"{vl_c.get(k, float('nan')):>8.4f}" for k in keys]
            print("    " + "  ".join(row))
        if metrics_path is not None:
            if ep == 0:
                _init_metrics_csv(metrics_path, ['epoch', 'train', 'val', 'monitor']
                                  + [f"val_{k}" for k in sorted(vl_c)]
                                  + [f"tr_{k}" for k in sorted(tr_c)])
            _append_metrics_csv(metrics_path, dict(
                epoch=ep + 1, train=round(tr, 5), val=round(vl, 5),
                monitor=(round(mon_vl, 5) if mon_vl is not None else ''),
                **{f"val_{k}": round(v, 5) for k, v in vl_c.items()},
                **{f"tr_{k}": round(v, 5) for k, v in tr_c.items()}))

        if score < best - 1e-5:
            best, best_ep, no_improve = score, ep, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            if ckpt:
                save_checkpoint(ckpt, model, {'epoch': ep, 'val': vl, 'monitor': mon_vl})
        else:
            no_improve += 1
            if patience and no_improve >= patience:
                print(f"  early stop at epoch {ep+1}: monitored validation has not "
                      f"improved for {patience} epochs")
                break

    if best_state is not None and best_ep != ep:
        print(f"  restoring epoch {best_ep+1} (best monitored validation "
              f"{best:.4f}) before freezing")
        model.load_state_dict(best_state)

    model.freeze_encoder()
    return model

@torch.no_grad()
def validate(model, loader, stage, device='cpu', encoder=None, rollout_steps=8):
    if loader is None:
        return float('nan')
    model.eval()
    vals = []
    for batch in loader:
        if isinstance(batch, dict):
            bdict = to_device(batch, device)
        else:
            batch = batch.to(device)
            bdict = _pyg_batch_to_dict(batch, device, num_types=model.cfg.num_types)
        if stage == 'flow':
            loss, _ = model.flow_losses(bdict, encoder=encoder)
        else:
            # no_grad: validation must not build the rollout graph, and must not
            # advance model._stage3_step (which drives the gen-weight warmup).
            with torch.no_grad():
                loss, _ = model.stage3_losses(bdict, rollout_steps=rollout_steps,
                                              apply_warmup=False)
        if torch.isfinite(loss):
            vals.append(loss.item())
    if not vals:
        return float('nan')
    return float(sum(vals) / len(vals))

def run_direct_flow(model, train_loader, val_loader=None, epochs=100, lr=1e-4,
                    clip=1.0, ckpt='direct_flow.pt', device='cpu', log_every=1,
                    metrics_path=None, patience=0, encoder=None):
    """Train the direct flow model.

    Args:
        encoder: unused, kept only for call-site backward compatibility.
                 The model's own self.encoder (pretrained via
                 run_encoder_pretrain) supplies flow-matching targets now.
    """
    if encoder is not None:
        print("  [WARN] run_direct_flow(encoder=...) is ignored -- "
              "DirectCrystalFlow uses its own internal, pretrained encoder. "
              "Call run_encoder_pretrain(model, ...) before this function.")
    model.to(device)
    if not bool(model._encoder_pretrained):
        raise RuntimeError(
            "model._encoder_pretrained is False -- run_encoder_pretrain(model, "
            "train_loader, ...) must be called before run_direct_flow(), or "
            "flow_losses/stage3_losses will be fitting the flow against an "
            "untrained encoder's output (silently useless training).")


    trainable = [p for n, p in model.named_parameters()
                 if p.requires_grad and (n.startswith('flow.')
                                         or n.startswith('conditioner.')
                                         or n.startswith('sg_coarse.'))]
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=1e-5)
    best = float('inf')
    no_improve = 0
    history = []
    if metrics_path is not None:
        _init_metrics_csv(metrics_path, ['epoch', 'train_loss', 'val_loss',
                                         'flow', 'repulsion', 'charge', 'property',
                                         'vpa', 'count', 'sg_align', 'lr'])
    for ep in range(epochs):
        cur_lr = _cosine_lr(opt, lr, ep, epochs)
        model.train()

        if bool(model._encoder_pretrained):
            model.encoder.eval()
            model.enc_proj.eval()
            model.enc_standardizer.eval()
        tl, comp_acc = [], {}
        bar = _prog(train_loader, f"DF {ep+1}/{epochs}")
        for batch in bar:
            if isinstance(batch, dict):
                bdict = to_device(batch, device)
            else:
                batch = batch.to(device)
                bdict = _pyg_batch_to_dict(batch, device, num_types=model.cfg.num_types)
            opt.zero_grad()

            # Flow matching loss
            loss, comps = model.flow_losses(bdict, encoder=encoder)

            if not torch.isfinite(loss):
                opt.zero_grad(); continue
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(trainable, clip)
            if not torch.isfinite(gn):
                opt.zero_grad(); continue
            opt.step()
            model.decoder.advance_wyckoff()      # once per step, not per forward
            tl.append(loss.item())
            for k, v in comps.items():
                comp_acc[k] = comp_acc.get(k, 0.0) + v
            if hasattr(bar, 'set_postfix'):
                bar.set_postfix(loss=f"{sum(tl)/len(tl):.4f}", lr=f"{cur_lr:.1e}")
        if not tl:
            raise RuntimeError(
                f"flow-matching epoch {ep+1} took no optimiser step: every "
                f"batch gave a non-finite loss or gradient. Diagnosis: "
                f"{_state_report(model)}.")
        tr = float(sum(tl) / len(tl))
        vl = validate(model, val_loader, 'flow', device, encoder=encoder) if val_loader is not None else tr
        history.append((tr, vl))
        if np.isfinite(vl) and vl < best:
            best = vl
            no_improve = 0
            save_checkpoint(ckpt, model, {'epoch': ep, 'val': vl})
        else:
            no_improve += 1
        if (ep + 1) % log_every == 0 or ep == epochs - 1:
            comp_str = {k: round(v / max(len(tl), 1), 4) for k, v in comp_acc.items()}
            print(f"  [df {ep+1:>3}/{epochs}] train {tr:7.4f}  val {vl:7.4f}  "
                  f"lr {cur_lr:.1e}  best {best:7.4f}  {comp_str}")
        if metrics_path is not None:
            row = {'epoch': ep + 1, 'train_loss': tr, 'val_loss': vl, 'lr': cur_lr}
            for k in ['flow', 'repulsion', 'charge', 'property', 'vpa', 'count', 'sg_align']:
                row[k] = round(comp_acc.get(k, 0.0) / max(len(tl), 1), 6)
            _append_metrics_csv(metrics_path, row)
        if patience and no_improve >= patience:
            print(f"  [df] early stop at epoch {ep+1} "
                  f"(no val improvement for {patience} epochs; best {best:.4f})")
            break
    if val_loader is not None:
        load_checkpoint(ckpt, model)
    return model, history

_S3_CSV_FIELDS = [
    'epoch', 'train_loss', 'val_loss', 'lr',
    'flow', 'repulsion', 'charge', 'property', 'property_struct',
    'property_rank', 'vpa', 'count', 'sg_align', 'composition',
    'recon', 'recon_count', 'mult', 'site_type', 'site_pos', 'wyckoff_ce',
    'lattice', 'wyckoff',
    'n_elem_gen', 'type_conf', 'vpa_gen', 'vpa_ref', 'n_cell_gen',
    'n_sites_gen', 'n_sites_soft', 'token_div', 'mult_gen',
]

def profile_stage3_step(model, batch, rollout_steps=16, repeats=3, device='cpu'):
   
    import time as _time

    def _sync():
        if str(device).startswith('cuda'):
            torch.cuda.synchronize()

    def _t(fn, n=repeats):
        fn(); _sync()
        t0 = _time.perf_counter()
        for _ in range(n):
            fn()
        _sync()
        return (_time.perf_counter() - t0) / n

    cfg = model.cfg
    was_training = model.training
    rows = []

    def step():
        loss, _ = model.stage3_losses(batch, rollout_steps=rollout_steps)
        loss.backward()
        model.zero_grad(set_to_none=True)

    model.train()
    _pk = int(getattr(cfg, 'property_every_k', 4))
    model._stage3_step.zero_()
    rows.append(('full step (fwd+bwd), property on', _t(step)))
    cfg.property_every_k = 10 ** 9          # never fires
    model._stage3_step.fill_(1)
    rows.append(('full step (fwd+bwd), property off', _t(step)))
    cfg.property_every_k = _pk

    model.eval()
    with torch.no_grad():
        yn = model.ynorm(batch['y'])
        B = yn.size(0)
        sgb = batch['sg'].view(-1).clamp(1, 230)
        cond = model.conditioner(yn, B, yn.device, drop_prob=0.0,
                                 training=False, sg=sgb, sg_drop_prob=0.0)
        z0 = torch.randn(B, cfg.n_sites, cfg.sec_dim, device=yn.device)
        ts = torch.linspace(0, 1, rollout_steps + 1, device=yn.device)

        def _vel(q, s_):
            return model.flow(q, s_.view(1, 1).expand(q.size(0), 1), cond,
                              key_padding_mask=model._occupancy_bias(q))

        def rollout():
            z = z0.clone()
            for i in range(rollout_steps):
                z = _rollout_step(model, z, ts[i:i + 1], 1.0 / rollout_steps,
                                  str(getattr(cfg, 'stage3_prefix_solver', 'euler')),
                                  _vel)
            return z
        rows.append((f'  rollout ({rollout_steps} steps, no grad)', _t(rollout)))
        z1 = rollout()
        dec = model.decoder(z1, generation=False,
                            formability_mask=model.formability_mask, sg=sgb)
        rows.append(('  decoder forward', _t(
            lambda: model.decoder(z1, generation=False,
                                  formability_mask=model.formability_mask, sg=sgb))))
        rows.append(('  _build_cell', _t(
            lambda: model._build_cell(z1, dict(dec), sgb,
                                      refine=(model.refiner is not None)))))
        L, _V, _nf, _ws = model._build_cell(z1, dict(dec), sgb, refine=False)
        rows.append(('  property_from_structure', _t(
            lambda: model.property_from_structure(
                dec['frac'], dec['type_probs'], L, dec['mask'], sg=sgb))))
        f_i, m_i, t_i = symmetry_orbit(
            dec['frac'], dec['mask'], dec['sampled_types'], sgb,
            model.sym_rot_lut, model.sym_trans_lut, model.sym_valid_lut,
            n_ops=int(getattr(cfg, 'repulsion_ops', 8)), stochastic=False, L=L,
            dedup=True, select=str(getattr(cfg, 'repulsion_op_select', 'nearest')))
        rows.append((f"  hardcore_repulsion (ops="
                     f"{int(getattr(cfg, 'repulsion_ops', 8))})", _t(
            lambda: hardcore_repulsion(
                dec['frac'], L, dec['mask'], dec['sampled_types'],
                model.radii_lut, other=(f_i, m_i, t_i), n_images=1, chunk=9))))
    rows.append(('  _recon_losses (fwd only)', _t(
        lambda: sum(model._recon_losses(batch, cfg.weights).values()))))

    print("\n  [s3 profile] batch=%d  n_sites=%d  rollout=%d  "
          "property_every_k=%d" % (batch['y'].size(0), cfg.n_sites,
                                   rollout_steps, _pk))
    for name, sec in rows:
        print(f"    {name:<40s} {sec:8.3f} s")
    print("    (indented rows are forward-only components of the step above "
          "them; they do not sum to it)")
    model.train(was_training)
    return dict(rows)


def run_stage3_finetune(model, train_loader, val_loader=None, epochs=60, lr=3e-5,
                        rollout_steps=8, gen_weight=1.0, clip=1.0,
                        ckpt='stage3_finetune.pt', device='cpu', log_every=1,
                        metrics_path=None, patience=0,
                        full_depth_diag_every=5, full_depth_diag_steps=100):
    """Generative fine-tuning via model.stage3_losses -- rollout the flow through
    the decoder and train lattice/sg/sg_coarse/property/decoder(Wyckoff) against
    the resulting structure, not just against a single teacher-forced step.
    """
    model.to(device)
    if not bool(model._encoder_pretrained):
        raise RuntimeError(
            "model._encoder_pretrained is False -- run_encoder_pretrain(model, "
            "train_loader, ...) must be called before run_stage3_finetune().")

    _dec_scale = float(getattr(model.cfg, 'stage3_decoder_lr_scale', 0.25))
    _gen_pref = ('flow.', 'conditioner.')
    gen_p = [p for n, p in model.named_parameters()
             if p.requires_grad and n.startswith(_gen_pref)]
    dec_p = [p for n, p in model.named_parameters()
             if p.requires_grad and not n.startswith(_gen_pref)]
    trainable = gen_p + dec_p
    opt = torch.optim.AdamW(
        [{'params': gen_p, 'lr': lr},
         {'params': dec_p, 'lr': lr * _dec_scale}], weight_decay=1e-5)
    print(f"  [s3] generator lr {lr:.2e} ({len(gen_p)} tensors), "
          f"decoder/heads lr {lr * _dec_scale:.2e} ({len(dec_p)} tensors)")
    best = float('inf')
    no_improve = 0
    history = []
    if metrics_path is not None:
        _init_metrics_csv(metrics_path, _S3_CSV_FIELDS)
    _steps_per_epoch = max(1, len(train_loader))
    _total_steps = _steps_per_epoch * max(1, epochs)
    _warmup_steps = int(getattr(model.cfg, 'stage3_lr_warmup_steps', 0))
    _gstep = 0
    if bool(getattr(model.cfg, 'stage3_profile', False)):
        for _b0 in train_loader:
            _bd0 = (to_device(_b0, device) if isinstance(_b0, dict)
                    else _pyg_batch_to_dict(_b0.to(device), device,
                                            num_types=model.cfg.num_types))
            profile_stage3_step(model, _bd0, rollout_steps=rollout_steps,
                                device=device)
            model._stage3_step.zero_()
            break
    for ep in range(epochs):
        cur_lr = _warmup_cosine_lr(opt, lr, _gstep, _total_steps, _warmup_steps)
        model.train()

        if bool(model._encoder_pretrained):
            model.encoder.eval()
            model.enc_proj.eval()
            model.enc_standardizer.eval()
        tl, comp_acc, n_skip = [], {}, 0
        bar = _prog(train_loader, f"stage3 {ep+1}/{epochs}")
        for batch in bar:
            cur_lr = _warmup_cosine_lr(opt, lr, _gstep, _total_steps, _warmup_steps)
            if isinstance(batch, dict):
                bdict = to_device(batch, device)
            else:
                batch = batch.to(device)
                bdict = _pyg_batch_to_dict(batch, device, num_types=model.cfg.num_types)
            opt.zero_grad()

            loss, comps = model.stage3_losses(
                bdict, rollout_steps=rollout_steps, gen_weight=gen_weight)

            if not torch.isfinite(loss):
                n_skip += 1
                opt.zero_grad(); continue
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(trainable, clip)
            if not torch.isfinite(gn):
                n_skip += 1
                opt.zero_grad(); continue
            opt.step()
            _gstep += 1
            model.decoder.advance_wyckoff()      # once per step, not per forward
            tl.append(loss.item())
            for k, v in comps.items():
                comp_acc[k] = comp_acc.get(k, 0.0) + v
            if hasattr(bar, 'set_postfix'):
                bar.set_postfix(loss=f"{sum(tl)/len(tl):.4f}", lr=f"{cur_lr:.1e}")
        
        n_done = len(tl)
        if n_skip:
            _data_warn(
                f"[s3 epoch {ep+1}] {n_skip}/{n_skip + n_done} steps skipped on "
                f"a non-finite loss or gradient "
                f"({100 * n_skip / max(n_skip + n_done, 1):.1f}%).")
        if n_done == 0:
            raise RuntimeError(
                f"stage 3 epoch {ep+1}: every one of {n_skip} steps produced a "
                f"non-finite loss or gradient, so no parameter was updated and "
                f"training cannot recover. Lower stage3_lr or "
                f"stage3_grad_steps; a degenerate predicted cell is the usual "
                f"source.")
        tr = float(sum(tl) / n_done)
        vl = (validate(model, val_loader, 'stage3', device, rollout_steps=rollout_steps)
              if val_loader is not None else tr)
        history.append((tr, vl))
        if vl < best:
            best = vl
            no_improve = 0
            save_checkpoint(ckpt, model, {'epoch': ep, 'val': vl})
        else:
            no_improve += 1
        if (ep + 1) % log_every == 0 or ep == epochs - 1:
            comp_str = {k: round(v / n_done, 4) for k, v in comp_acc.items()}
            print(f"  [s3 {ep+1:>3}/{epochs}] train {tr:7.4f}  val {vl:7.4f}  "
                  f"lr {cur_lr:.1e}  best {best:7.4f}  skip {n_skip}  {comp_str}")
        if (full_depth_diag_every and val_loader is not None
                and (ep + 1) % full_depth_diag_every == 0):
            try:
                _diag_batch = next(iter(val_loader))
                if not isinstance(_diag_batch, dict):
                    _diag_batch = _pyg_batch_to_dict(
                        _diag_batch.to(device), device, num_types=model.cfg.num_types)
                else:
                    _diag_batch = to_device(_diag_batch, device)
                print(f"  [s3 {ep+1:>3}/{epochs}] full-depth check "
                      f"({full_depth_diag_steps} steps, train uses {rollout_steps}):")
                model.diagnostics(_diag_batch, rollout_steps=full_depth_diag_steps,
                                  verbose=True)
            except Exception as _e:
                _data_warn(f"[s3 epoch {ep+1}] full-depth diagnostic failed: {_e}")
        if metrics_path is not None:
            row = {'epoch': ep + 1, 'train_loss': tr, 'val_loss': vl, 'lr': cur_lr}
            for k in _S3_CSV_FIELDS:
                if k in ('epoch', 'train_loss', 'val_loss', 'lr'):
                    continue
                row[k] = round(comp_acc.get(k, 0.0) / n_done, 6)
            _append_metrics_csv(metrics_path, row)
        if patience and no_improve >= patience:
            print(f"  [s3] early stop at epoch {ep+1} "
                  f"(no val improvement for {patience} epochs; best {best:.4f})")
            break
    if val_loader is not None:
        load_checkpoint(ckpt, model)
    return model, history


def _site_priority(nfree, conf, idx):
    """Sort key deciding which of two conflicting sites keeps its place.
    """
    return (int(nfree), -float(conf), int(idx))


@torch.no_grad()
def resolve_site_conflicts(dec, sg, L, radii_lut, min_dist=0.75,
                           overlap_scale=0.5, n_relocate=32, snap_tol=SYM_SNAP_TOL,
                           max_atoms=None, seed=0):
    """Make the decoded asymmetric unit expand to a conflict-free crystal.
    """
    frac = dec['frac']
    mask = dec['mask']
    dev = frac.device
    B, M, _ = frac.shape
    f_np = frac.detach().cpu().numpy().astype(np.float64) % 1.0
    m_np = mask.detach().cpu().numpy() > 0.5
    t_np = dec['sampled_types'].detach().cpu().numpy().astype(int)
    L_np = L.detach().cpu().numpy().astype(np.float64)
    sg_np = torch.as_tensor(sg).reshape(-1).cpu().numpy().astype(int)
    radii = radii_lut.detach().cpu().numpy().astype(np.float64)

    P_np = (dec['site_P'].detach().cpu().numpy().astype(np.float64)
            if dec.get('site_P') is not None else None)
    o_np = (dec['site_o'].detach().cpu().numpy().astype(np.float64)
            if dec.get('site_o') is not None else None)
    nf_np = (dec['wyckoff_nfree'].detach().cpu().numpy().astype(int)
             if dec.get('wyckoff_nfree') is not None else None)

    conf = dec['type_probs'].detach().amax(-1)
    if dec.get('wyckoff_probs') is not None:
        conf = conf * dec['wyckoff_probs'].detach().amax(-1)
    conf = (conf * mask.detach()).cpu().numpy().astype(np.float64)

    rng = np.random.default_rng(int(seed))
    n_reloc = n_drop = n_conf = 0
    all_orbits = []
   
    mult_out = np.zeros((B, M), dtype=np.float32)

    for b in range(B):
        orbits_b = [None] * M
        idxs = [i for i in range(M) if m_np[b, i]]
        all_orbits.append(orbits_b)
        if not idxs:
            continue
        sgb = int(sg_np[b]) if sg_np.size > 1 else int(sg_np[0])
        ops_arr = _sg_ops_arrays(sgb) if 1 <= sgb <= 230 else (
            np.eye(3)[None], np.zeros((1, 3)))
        if ops_arr[0].shape[0] == 0:
            ops_arr = (np.eye(3)[None], np.zeros((1, 3)))
        L_use = L_np[b]
        try:
            invL = np.linalg.inv(L_use)
        except np.linalg.LinAlgError:
            continue

        order = sorted(idxs, key=lambda i: _site_priority(
            nf_np[b, i] if nf_np is not None else 3, conf[b, i], i))

        acc_pts = np.zeros((0, 3))
        acc_r = np.zeros((0,))
        n_acc_atoms = 0

        for i in order:
            r_i = float(radii[int(t_np[b, i]) % len(radii)])
            if nf_np is None or P_np is None:
                # no Wyckoff machinery: fall back to the escalating snap
                f_s, orb, _tol, _d, _esc = _physical_site_snap(
                    f_np[b, i], ops_arr, L_use, invL, base_tol=snap_tol,
                    min_dist=min_dist)
                cands = [(f_s, orb)]
                nfree_i = 3
            else:
                nfree_i = int(nf_np[b, i])
                cands = [(f_np[b, i], _orbit_of(f_np[b, i], ops_arr, L_use, invL,
                                                tol_sq=1e-4))]

            def _score(orbit):
                """Smallest contact this orbit would create, in Angstrom."""
                d_int = _orbit_min_contact(orbit, L_use, invL)
                if acc_pts.shape[0]:
                    d2 = _min_image_sq_matrix(orbit, acc_pts, L_use, invL)
                    thr = np.maximum(overlap_scale * (r_i + acc_r), min_dist)
                    # margin: how far every pair is above its own threshold
                    marg = np.sqrt(np.maximum(d2, 0.0)) - thr[None, :]
                    d_ext = float(marg.min())
                else:
                    d_ext = float('inf')
                thr_self = max(overlap_scale * 2.0 * r_i, min_dist)
                return min(d_int - thr_self, d_ext)

            best_f, best_orb = cands[0]
            best_s = _score(best_orb)

            if best_s < 0.0 and nfree_i > 0 and P_np is not None:
                # RELOCATE inside the Wyckoff subspace: f = P u + o keeps the
                # site symmetry exactly, so the class, its multiplicity and the
                # space group all survive the move.
                n_conf += 1
                P_i, o_i = P_np[b, i], o_np[b, i]
                u = rng.random((int(n_relocate), 3))
                cf = (u @ P_i.T + o_i[None]) % 1.0
                for c in range(cf.shape[0]):
                    orb_c = _orbit_of(cf[c], ops_arr, L_use, invL, tol_sq=1e-4)
                    sc = _score(orb_c)
                    if sc > best_s:
                        best_s, best_f, best_orb = sc, cf[c], orb_c
                    if best_s >= 0.0:
                        break
                if best_s >= 0.0:
                    n_reloc += 1
            elif best_s < 0.0:
                n_conf += 1

            _first = (n_acc_atoms == 0)
            if best_s < 0.0 and not _first:
                m_np[b, i] = False
                n_drop += 1
                continue
            if (max_atoms is not None and not _first
                    and n_acc_atoms + len(best_orb) > int(max_atoms)):
                m_np[b, i] = False
                n_drop += 1
                continue

            f_np[b, i] = best_f % 1.0
            orbits_b[i] = best_orb
            mult_out[b, i] = float(len(best_orb))
            acc_pts = np.concatenate([acc_pts, best_orb], 0)
            acc_r = np.concatenate([acc_r, np.full(len(best_orb), r_i)], 0)
            n_acc_atoms += len(best_orb)

    dec['frac'] = torch.as_tensor(f_np, dtype=frac.dtype, device=dev)
    dec['mask'] = torch.as_tensor(m_np.astype(np.float32),
                                  dtype=mask.dtype, device=dev)
    return {'orbits': all_orbits, 'n_relocated': n_reloc, 'n_dropped': n_drop,
            'n_conflicts': n_conf,
            'mult': torch.as_tensor(mult_out, dtype=frac.dtype, device=dev)}


def reconcile_symmetry_equivalent_types(out, sg_pred, snap_tol=SYM_SNAP_TOL,
                                        clash_tol_sq=1e-4):
   
    frac, mask, types = out['frac'], out['mask'], out['sampled_types']
    raw, probs = out.get('sampled_types_raw'), out.get('type_probs')
    total_changed = 0
    for b in range(frac.size(0)):
        mk = mask[b].bool()
        if int(mk.sum()) < 2:
            continue
        sg = int(sg_pred[b]) if torch.is_tensor(sg_pred) else int(sg_pred[b])
        if not (1 <= sg <= 230):
            continue
        ops_arr = _sg_ops_arrays(sg)
        if ops_arr[0].shape[0] == 0:
            continue
        L_use = (out['lattice'][b].detach().cpu().numpy()
                 if 'lattice' in out else np.eye(3))
        if not np.isfinite(np.linalg.cond(L_use)):
            continue                       # degenerate cell, nothing to compare
        fb = frac[b][mk].detach().cpu().numpy() % 1.0
        tb = types[b][mk].detach().cpu().numpy().astype(int)
        snapped = _site_symmetry_snap(fb, ops_arr, tol=snap_tol)
        n, K = len(snapped), ops_arr[0].shape[0]

        img = (np.einsum('kij,nj->nki', ops_arr[0], snapped)
               + ops_arr[1][None]) % 1.0                       # (n, K, 3)
        d = img.reshape(n * K, 3)[:, None, :] - snapped[None]   # (nK, n, 3)
        d -= np.round(d)
        c = d @ L_use
        coin = ((c * c).sum(-1) < clash_tol_sq).reshape(n, K, n).any(1)
        coin &= tb[:, None] != tb[None, :]
        if not coin.any():
            continue

        parent = np.arange(n)

        def _find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return int(x)

        for i, j in zip(*np.nonzero(np.triu(coin, 1))):
            ri, rj = _find(int(i)), _find(int(j))
            if ri != rj:
                parent[ri] = rj

        roots = np.array([_find(i) for i in range(n)])
        pb = probs[b][mk].detach().cpu().numpy() if probs is not None else None
        rb = raw[b][mk].detach().cpu().numpy().astype(int) if raw is not None else None
        nb_changed = 0
        for r in np.unique(roots):
            members = np.nonzero(roots == r)[0]
            if members.size < 2:
                continue
            if pb is not None:
                chosen = int(pb[members].sum(0).argmax())
            else:
                vals, cnts = np.unique(tb[members], return_counts=True)
                chosen = int(vals[int(cnts.argmax())])
            nb_changed += int((tb[members] != chosen).sum())
            tb[members] = chosen
            if rb is not None:
                rb[members] = chosen
            if pb is not None:
                pb[members] = 0.0
                pb[members, chosen] = 1.0
        if nb_changed:
            idx = torch.where(mk)[0].to(types.device)
            types[b].index_copy_(0, idx, torch.as_tensor(
                tb, dtype=types.dtype, device=types.device))
            if rb is not None:
                raw[b].index_copy_(0, idx, torch.as_tensor(
                    rb, dtype=raw.dtype, device=raw.device))
            if pb is not None:
                probs[b].index_copy_(0, idx, torch.as_tensor(
                    pb, dtype=probs.dtype, device=probs.device))
            total_changed += nb_changed
    return total_changed

def expand_generated(out, sg_pred, num_types, max_atoms=80, symprec=0.1,
                     mined_sym_ops=None, snap_tol=SYM_SNAP_TOL, reconcile_types=True,
                     target_vpa=None, rescale_volume=True, merge_dist=0.75,
                     resolved_orbits=None, priority=None):
    """Replicate the asymmetric unit into the full cell.
    """
    import numpy as np
    if reconcile_types:
        n_rec = reconcile_symmetry_equivalent_types(out, sg_pred, snap_tol=snap_tol)
    else:
        n_rec = 0

    B = out['frac'].size(0)
    dev = out['frac'].device
    frac = out['frac'].cpu().numpy()
    mask = out['mask'].cpu().numpy()
    types = out['sampled_types'].cpu().numpy()
    types_raw = (out['sampled_types_raw'].cpu().numpy()
                 if 'sampled_types_raw' in out else types)
    sgn = sg_pred.cpu().numpy().astype(int)
    L_np = out['lattice'].detach().cpu().numpy() if 'lattice' in out else None

    if mined_sym_ops is not None:
        batch_ops = list(mined_sym_ops)
    else:
        batch_ops = [None] * B

    ef, et, et_raw, esrc, A = [], [], [], [], 1
    conv_mats = []
    n_failed, n_conflicts, n_trunc, n_escalated_total = 0, 0, 0, 0

    for b in range(B):
        mk = mask[b] > 0.5
        fb = frac[b][mk] % 1.0
        tb = [int(t) for t in types[b][mk].tolist()]
        tb_raw = [int(t) for t in types_raw[b][mk].tolist()]
        sg = int(sgn[b])
        src_idx = np.where(mk)[0]
        if sg < 1 or sg > 230 or len(fb) == 0:
            ef.append(fb); et.append(tb); et_raw.append(tb_raw)
            esrc.append(list(src_idx)); A = max(A, len(tb))
            conv_mats.append(L_np[b] if L_np is not None else np.eye(3)); continue

        if batch_ops[b] is not None:
            L_use = L_np[b] if L_np is not None else np.eye(3)
            ops_arr = _sg_ops_arrays(sg, mined_ops=batch_ops[b])
        else:
            try:
                from pymatgen.core.lattice import Lattice
                params = matrix_to_lattice_params(
                    torch.as_tensor(L_np[b], dtype=torch.float32)).numpy()
                a, b_, c, al, be, ga = (float(params[0]), float(params[1]),
                                        float(params[2]), float(params[3]),
                                        float(params[4]), float(params[5]))
                conv_lat = Lattice.from_parameters(a, b_, c, al, be, ga)
                L_use = conv_lat.matrix
            except Exception:
                L_use = L_np[b] if L_np is not None else np.eye(3)
            ops_arr = _sg_ops_arrays(sg)

        if ops_arr[0].shape[0] == 0:
            ef.append(fb); et.append(tb); et_raw.append(tb_raw)
            esrc.append(list(src_idx)); A = max(A, len(tb))
            conv_mats.append(L_use); continue

        invL = np.linalg.inv(L_use)

        pre = (resolved_orbits[b] if resolved_orbits is not None else None)
        if len(fb) and pre is not None:
            orbits = [pre[int(j)] for j in src_idx]
            if any(o_ is None for o_ in orbits):
                orbits = [o_ if o_ is not None else
                          _orbit_of(fb[k], ops_arr, L_use, invL, tol_sq=1e-4)
                          for k, o_ in enumerate(orbits)]
            n_escalated = 0
        elif len(fb):
            _snap_out = [_physical_site_snap(fb[i], ops_arr, L_use, invL,
                                             base_tol=snap_tol, min_dist=merge_dist)
                        for i in range(len(fb))]
            orbits = [so[1] for so in _snap_out]
            n_escalated = int(sum(1 for so in _snap_out if so[4]))
            n_escalated_total += n_escalated
        else:
            orbits = []
            n_escalated = 0
        # Lowest priority truncates first. Without a priority vector the
        # fallback is the old smallest-orbit-first rule.
        if priority is not None:
            _pr = np.asarray(priority[b]).reshape(-1)[src_idx]
            _order = list(np.argsort(-_pr, kind='stable'))
        else:
            _order = sorted(range(len(orbits)), key=lambda i: len(orbits[i]))

        of, ot, ot_raw, os_ = [], [], [], []
        for _i in _order:
            orbit, t, t_raw = orbits[_i], tb[_i], tb_raw[_i]
            if len(of) + len(orbit) > max_atoms:
                n_trunc += 1
                continue                       # whole site skipped, never split
            if of:
                d2 = _min_image_sq_matrix(orbit, np.asarray(of), L_use, invL)
                if bool((d2 < (merge_dist ** 2)).any()):
                    # Should be unreachable once resolve_site_conflicts has run.
                    # Counted, and the whole site is dropped so that whatever is
                    # emitted is still closed under the group.
                    n_conflicts += 1
                    continue
            for nf in orbit:
                of.append(nf); ot.append(t); ot_raw.append(t_raw)
                os_.append(int(src_idx[_i]))

        if len(of) == 0:
            n_failed += 1
            of, ot, ot_raw, os_ = list(fb), list(tb), list(tb_raw), list(src_idx)

        ef.append(np.asarray(of)); et.append(ot); et_raw.append(ot_raw)
        esrc.append(os_)
        A = max(A, len(ot))
        conv_mats.append(L_use)

    A = max(1, min(A, max_atoms))
    pf = torch.zeros(B, A, 3, device=dev)
    pt = torch.zeros(B, A, dtype=torch.long, device=dev)
    pr = torch.zeros(B, A, dtype=torch.long, device=dev)
    pm = torch.zeros(B, A, device=dev)
    probs_in = out.get('type_probs')
    pp = (torch.zeros(B, A, probs_in.size(-1), device=dev, dtype=probs_in.dtype)
          if probs_in is not None else None)
    for b in range(B):
        k = min(len(et[b]), A)
        if k:
            pf[b, :k] = torch.as_tensor(ef[b][:k], dtype=pf.dtype, device=dev)
            pt[b, :k] = torch.as_tensor(et[b][:k], dtype=torch.long, device=dev)
            pr[b, :k] = torch.as_tensor(et_raw[b][:k], dtype=torch.long, device=dev)
            pm[b, :k] = 1.0
            if pp is not None and len(esrc[b]) >= k:
                si = torch.as_tensor(esrc[b][:k], dtype=torch.long, device=dev)
                pp[b, :k] = probs_in[b].index_select(0, si)
    out = dict(out)
    out['frac'], out['sampled_types'], out['mask'] = pf, pt, pm
    out['sampled_types_raw'] = pr
    if pp is not None:
        out['type_probs'] = pp
    conv_stack = (np.stack([m for m in conv_mats], 0) if conv_mats
                  else np.tile(np.eye(3), (B, 1, 1)))
    conv_mat = torch.as_tensor(conv_stack, dtype=pf.dtype, device=dev)

    if rescale_volume and target_vpa is not None:
        tv = torch.as_tensor(target_vpa, dtype=pf.dtype, device=dev).reshape(-1)
        if tv.numel() == 1:
            tv = tv.expand(B)
        n_now = pm.sum(1).clamp(min=1.0)
        V_want = (tv * n_now).clamp(min=1e-3)
        V_now = torch.linalg.det(conv_mat).abs().clamp(min=1e-6)
        scale = (V_want / V_now).clamp(min=1e-6).pow(1.0 / 3.0)
        conv_mat = conv_mat * scale.view(-1, 1, 1)

    out['coords'] = torch.einsum('bak,bkl->bal', pf, conv_mat)
    out['lattice_conv'] = conv_mat

    out['lattice'] = conv_mat
    out['n_expansion_failed'] = n_failed
    out['n_site_conflicts'] = n_conflicts
    out['n_orbit_truncated'] = n_trunc
    out['n_snap_escalated'] = n_escalated_total
    out['n_types_reconciled'] = n_rec
    return out

@torch.no_grad()
def enforce_unique_types(types, probs, mask, max_unique):
    if not max_unique or max_unique <= 0:
        return types, probs
    types, probs = types.clone(), probs.clone()
    B, A = types.shape
    for b in range(B):
        mb = mask[b].bool()
        t_b = types[b][mb]
        if t_b.numel() == 0:
            continue
        uniq, counts = t_b.unique(return_counts=True)
        if uniq.numel() <= max_unique:
            continue
        keep_list = uniq[counts.argsort(descending=True)[:max_unique]].tolist()
        keep_set = set(keep_list)
        for j in range(A):
            if not bool(mb[j]) or int(types[b, j]) in keep_set:
                continue
            new_t = keep_list[int(probs[b, j, keep_list].argmax())]
            types[b, j] = new_t
            probs[b, j].zero_()
            probs[b, j, new_t] = 1.0
    return types, probs

@torch.no_grad()
def smact_balance_types(types_v, probs, mask, ox_states_padded, ox_states_mask,
                        max_edits=8, atomic_numbers=None, site_weights=None):
    """Greedy charge-neutrality edit pass.
    """
    import itertools
    from collections import Counter
    ox = ox_states_padded.detach().cpu().numpy() if torch.is_tensor(ox_states_padded) else np.asarray(ox_states_padded)
    om = ox_states_mask.detach().cpu().numpy() if torch.is_tensor(ox_states_mask) else np.asarray(ox_states_mask)
    tv = types_v.detach().cpu().numpy().copy() if torch.is_tensor(types_v) else np.asarray(types_v)
    pr = probs.detach().cpu().numpy().copy() if torch.is_tensor(probs) else np.asarray(probs)
    mb = mask.detach().cpu().numpy() > 0.5 if torch.is_tensor(mask) else (np.asarray(mask) > 0.5)
    if site_weights is None:
        sw = np.ones(tv.shape, dtype=np.int64)
    else:
        sw = (site_weights.detach().cpu().numpy() if torch.is_tensor(site_weights)
              else np.asarray(site_weights))
        sw = np.maximum(np.rint(sw), 1).astype(np.int64)
    B, N, T = pr.shape

    def _states(e):
        s = ox[e][om[e] > 0]
        return s if s.size else np.array([0.0])

    def _en(e):

        if atomic_numbers is None or e >= len(atomic_numbers):
            return 1.5
        return _PAULING_EN.get(int(atomic_numbers[e]), 1.5)

    def _best_combo(tlist, wlist=None):
        """Returns (min_abs_net, winning_combo_or_None, elems)."""
        c = Counter()
        if wlist is None:
            c.update(tlist)
        else:
            for e, m in zip(tlist, wlist):
                c[e] += int(m)
        elems = list(c.keys())
        if not elems:
            return 0.0, None, elems
        states = [_states(e) for e in elems]
        if len(states) == 1:
            s = states[0]
            j = int(np.argmin(np.abs(s)))
            return float(abs(s[j])), (s[j],), elems
        best, best_combo = 1e9, None
        for combo in itertools.product(*states):
            net = sum(c[e] * o for e, o in zip(elems, combo))
            a = abs(net)
            if a < best:
                best, best_combo = a, combo
            if best == 0:
                break
        return best, best_combo, elems

    def _min_abs_net(tlist, wlist=None):
        return _best_combo(tlist, wlist)[0]

    def _pauling_ok(tlist, wlist=None):
        """Cations (positive ox state) must not be MORE electronegative
        than anions (negative ox state) in the best-fit assignment."""
        net, combo, elems = _best_combo(tlist, wlist)
        if combo is None or len(elems) < 2:
            return True
        cations = [(elems[i], combo[i]) for i in range(len(elems)) if combo[i] > 0]
        anions = [(elems[i], combo[i]) for i in range(len(elems)) if combo[i] < 0]
        for ce, _ in cations:
            for ae, _ in anions:
                if _en(ce) > _en(ae):
                    return False
        return True

    for b in range(B):
        idx = np.where(mb[b])[0]
        if idx.size == 0:
            continue
        edited = tv[b, idx].tolist()
        wts = sw[b, idx].tolist()
        if _min_abs_net(edited, wts) == 0.0 and _pauling_ok(edited, wts):
            continue
        conf = pr[b, idx].max(-1)
        order = np.argsort(conf)
        for _ in range(max_edits):
            best_before = _min_abs_net(edited, wts)
            pauling_before = _pauling_ok(edited, wts)
            best_trial, best_score = None, None
            for j_local in order:
                j = int(idx[j_local])
                orig = int(tv[b, j])
                for k in range(T):
                    if k == orig:
                        continue
                    if om[k].sum() == 0:
                        continue
                    if atomic_numbers is not None and k < len(atomic_numbers) \
                            and int(atomic_numbers[k]) in NON_FORMABLE_Z:
                        continue
                    trial = list(edited)
                    trial[j_local] = k
                    net = _min_abs_net(trial, wts)
                    if net >= best_before - 1e-6:
                        continue
                    p_ok = _pauling_ok(trial, wts)
                    if (not p_ok) and pauling_before:

                        continue
                    score = (net, 0 if p_ok else 1)
                    if best_score is None or score < best_score:
                        best_score, best_trial = score, (j_local, j, k, net)
            if best_trial is None:
                break
            j_local, j, k, net = best_trial
            edited[j_local] = k
            tv[b, j] = k
            pr[b, j] = 0.0
            pr[b, j, k] = 1.0
            if net == 0.0 and _pauling_ok(edited, wts):
                break

    out_t = torch.as_tensor(tv, dtype=types_v.dtype if torch.is_tensor(types_v) else torch.long)
    out_p = torch.as_tensor(pr, dtype=probs.dtype if torch.is_tensor(probs) else torch.float32)
    if torch.is_tensor(types_v):
        out_t = out_t.to(types_v.device)
        out_p = out_p.to(probs.device)
    return out_t, out_p

def apply_formability_mask(probs, atomic_numbers):
    pr = probs.detach().clone() if torch.is_tensor(probs) else torch.as_tensor(probs).clone()
    T = pr.shape[-1]
    for vi, z in enumerate(atomic_numbers):
        if vi >= T:
            break
        if int(z) in NON_FORMABLE_Z:
            pr[..., vi] = 0.0
    pr = pr.clamp(min=0.0)
    s = pr.sum(dim=-1, keepdim=True)
    s = s.clamp(min=1e-8)
    return pr / s

@torch.no_grad()
def structural_validity(frac, L, mask, types, radii_lut,
                        vpa_lo=3.0, vpa_hi=60.0, conn_cut=4.0, scale=0.5,
                        min_abs=0.75, return_components=False):
    """Geometric plausibility of a decoded cell."""
    B, A, _ = frac.shape
    dvec, dist = min_image_disp(frac, frac, L, n_images=1, chunk=9)
    eye = torch.eye(A, device=frac.device, dtype=torch.bool).unsqueeze(0)
    valid_pair = mask.bool().unsqueeze(2) & mask.bool().unsqueeze(1) & ~eye
    r = radii_lut[types.clamp(min=0)]
    thr = (scale * (r.unsqueeze(2) + r.unsqueeze(1))).clamp(min=float(min_abs))
    overlap = (valid_pair & (dist < thr)).any(2).any(1)
    V = torch.linalg.det(L).abs()
    n = mask.sum(1).clamp(min=1)
    vpa = V / n
    vpa_ok = (vpa >= vpa_lo) & (vpa <= vpa_hi)
    latt_min = torch.linalg.norm(L, dim=-1).min(dim=-1).values
    self_nbr = (latt_min < conn_cut).view(-1, 1)
    nbr = valid_pair & (dist < conn_cut) & (dist > 1e-6)
    has_nbr = (nbr.any(dim=2) | self_nbr | ~mask.bool())
    connected = has_nbr.all(dim=1)
    ok = (~overlap) & vpa_ok & connected
    if return_components:
        return ok, {'no_overlap': ~overlap, 'vpa_ok': vpa_ok,
                    'connected': connected, 'vpa': vpa}
    return ok

def _smact_valid_composition(zs, counts, ox_table, en_table, metal_set,
                             use_pauling_test=True, include_alloys=True,
                             max_combos=400000):
    """ Checked against the installed `smact` package on 600 random compositions:
    """
    import itertools
    zs = [int(z) for z in zs]
    counts = [int(c) for c in counts]
    if not zs:
        return False
    uniq = sorted(set(zs))
    if len(uniq) == 1:
        return True
    if include_alloys and all(z in metal_set for z in uniq):
        return True

    agg = {}
    for z, c in zip(zs, counts):
        agg[z] = agg.get(z, 0) + c
    elems = sorted(agg)
    cnt = [agg[e] for e in elems]
    g = 0
    for v in cnt:
        g = math.gcd(g, int(v))
    g = max(g, 1)
    cnt = [c // g for c in cnt]

    states = [list(ox_table.get(e, [])) for e in elems]
    if any(len(s) == 0 for s in states):
        return False
    n_comb = 1
    for s in states:
        n_comb *= len(s)
        if n_comb > max_combos:
            break
    if n_comb > max_combos:
        # too many assignments to enumerate: fall back to interval feasibility,
        # which is a strict over-approximation (never rejects a valid cell)
        lo = sum(c * min(s) for c, s in zip(cnt, states))
        hi = sum(c * max(s) for c, s in zip(cnt, states))
        return bool(lo <= 0 <= hi)

    ens = [en_table.get(e, None) for e in elems]
    for combo in itertools.product(*states):
        if sum(c * o for c, o in zip(cnt, combo)) != 0:
            continue
        if not use_pauling_test:
            return True
        ok = True
        for i in range(len(combo)):
            for j in range(i + 1, len(combo)):
                e1, e2 = ens[i], ens[j]
                o1, o2 = combo[i], combo[j]
                if e1 is None or e2 is None:
                    ok = False
                elif (o1 > 0 and o2 < 0 and e1 >= e2) or (o1 < 0 and o2 > 0 and e1 <= e2):
                    ok = False
                if not ok:
                    break
            if not ok:
                break
        if ok:
            return True
    return False

@torch.no_grad()
def composition_validity(types, mask, ox_states_padded=None, ox_states_mask=None,
                         atomic_numbers=None, use_pauling_test=True,
                         include_alloys=True, use_model_table=False,
                         eneg_lut=None, metal_lut=None):
    """SMACT composition validity, matching the published CDVAE gate.
    """
    tv = types.detach().cpu().numpy()
    mb = mask.detach().cpu().numpy() > 0.5
    B = tv.shape[0]

    if use_model_table and ox_states_padded is not None:
        op = ox_states_padded.detach().cpu().numpy()
        om = ox_states_mask.detach().cpu().numpy()
        ox_table = {}
        for vi in range(op.shape[0]):
            z = int(atomic_numbers[vi]) if atomic_numbers is not None else vi
            s = op[vi][om[vi] > 0]
            if s.size:
                ox_table[z] = [int(round(float(v))) for v in s]
        en_table, metal_set = {}, set()
        use_pauling_test = False
        include_alloys = False
    else:
        ox_table = {z: list(v) for z, v in SMACT_OXIDATION_STATES.items()}
        en_table = {z: float(v) for z, v in SMACT_PAULING_EN.items()}
        metal_set = set(SMACT_METAL_Z)

    if atomic_numbers is None and ox_states_padded is not None:
        _data_warn(
            "composition_validity was given a model oxidation table but no "
            "`atomic_numbers`, so the type indices are being read as atomic "
            "numbers. Pass atomic_numbers=list(cfg.atomic_numbers) -- otherwise "
            "vocabulary index 3 is scored as lithium.")

    # One cache per call. Scoring 1000 samples typically touches far fewer
    # distinct reduced formulas, and the enumeration behind each is the
    # expensive part.
    cache = {}
    out = np.zeros(B, dtype=bool)
    for b in range(B):
        tl = tv[b][mb[b]]
        if tl.size == 0:
            continue
        if atomic_numbers is not None:
            lut = np.asarray(atomic_numbers, dtype=np.int64)
            zl = lut[np.clip(tl.astype(np.int64), 0, len(lut) - 1)]
        else:
            zl = tl.astype(np.int64)
        elems, counts = np.unique(zl, return_counts=True)
        counts = counts // np.gcd.reduce(counts)
        key = (tuple(elems.tolist()), tuple(counts.tolist()))
        hit = cache.get(key)
        if hit is None:
            hit = _smact_valid_composition(
                elems.tolist(), counts.tolist(), ox_table, en_table, metal_set,
                use_pauling_test=use_pauling_test, include_alloys=include_alloys)
            cache[key] = hit
        out[b] = hit
    return torch.as_tensor(out, device=types.device)

def _reduced_formula_key(types, mask, atomic_numbers=None):
    """(sorted element, integer-reduced count) tuple -- a composition fingerprint."""
    from collections import Counter
    import math as _m
    tl = [int(t) for t, m in zip(types.tolist(), mask.tolist()) if m > 0.5]
    if not tl:
        return ()
    c = Counter(tl)
    g = 0
    for v in c.values():
        g = _m.gcd(g, int(v))
    g = max(g, 1)
    if atomic_numbers is not None:
        return tuple(sorted((int(atomic_numbers[e]) if e < len(atomic_numbers) else int(e),
                             int(v) // g) for e, v in c.items()))
    return tuple(sorted((int(e), int(v) // g) for e, v in c.items()))

def _training_composition_keys(train_dataset, atomic_to_idx=None):
    import math as _m
    from collections import Counter
    keys = set()
    if train_dataset is None:
        return keys
    try:
        for g in train_dataset.graph_data:
            zs = [int(z) for z in np.asarray(g.nodes).tolist()]
            c = Counter(zs)
            gg = 0
            for v in c.values():
                gg = _m.gcd(gg, int(v))
            gg = max(gg, 1)
            keys.add(tuple(sorted((int(e), int(v) // gg) for e, v in c.items())))
    except Exception:
        pass
    return keys

def _spearman(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 3:
        return float('nan')
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean(); rb -= rb.mean()
    den = (np.sqrt((ra ** 2).sum()) * np.sqrt((rb ** 2).sum()))
    return float((ra * rb).sum() / den) if den > 0 else float('nan')

@torch.no_grad()
def generate_and_evaluate(model, train_targets, device='cpu', quantiles=None,
                          n_per=32, steps=100, temperature=0.0, guidance=None,
                          results_path=None, train_dataset=None):
    """Sample the trained model and score it.
    """
    cfg = model.cfg
    model.eval()
    qs = quantiles or [0.05, 0.25, 0.50, 0.75, 0.95]
    ys = np.asarray(train_targets, dtype=np.float64)
    rows = []
    train_keys = _training_composition_keys(train_dataset)
    zs_vocab = list(cfg.atomic_numbers) if cfg.atomic_numbers is not None else None
    all_targets, all_struct_y = [], []
    print("\nGeneration + evaluation")
    for q in qs:
        ty = float(np.quantile(ys, q))
        out = model.sample(n_per, target_y=ty, steps=steps, guidance=guidance,
                           device=device, temperature=temperature,
                           expand_orbits=True)
        frac, L = out['frac'], out['lattice']
        mask, types = out['mask'], out['sampled_types']
        ok_s, comp = structural_validity(
            frac, L, mask, types, model.radii_lut,
            scale=float(getattr(cfg, 'validity_overlap_scale', 0.5)),
            min_abs=float(getattr(cfg, 'validity_min_dist', 0.75)),
            return_components=True)
        ok_c = composition_validity(
            types, mask, model.ox_states_padded, model.ox_states_mask,
            atomic_numbers=list(cfg.atomic_numbers) if cfg.atomic_numbers else None,
            use_pauling_test=True, include_alloys=True, use_model_table=False)
        ok_c_strict = composition_validity(
            types, mask, model.ox_states_padded, model.ox_states_mask,
            atomic_numbers=list(cfg.atomic_numbers) if cfg.atomic_numbers else None,
            use_model_table=True)

        sg_agree = (out['sg_pred'].view(-1) == out['sg_fine'].view(-1)).float().mean()
        keys = set()
        comp_keys = []
        for b in range(frac.size(0)):
            m = mask[b].bool()
            keys.add((int(out['sg_pred'][b]),
                      tuple(sorted(types[b][m].tolist()))))
            comp_keys.append(_reduced_formula_key(types[b], mask[b], zs_vocab))
        novel = ([k for k in comp_keys if k and k not in train_keys]
                 if train_keys else [])

        _pe = out.get('pre_expansion') or out
        y_struct = model.ydenorm(model.property_from_structure(
            _pe['frac'], _pe['type_probs'], _pe['lattice'], _pe['mask'],
            sg=_pe.get('sg'))[0])
        all_targets.extend([ty] * int(frac.size(0)))
        all_struct_y.extend([float(v) for v in y_struct.reshape(-1).tolist()])
        row = dict(
            quantile=q, target_y=round(ty, 5),
            struct_valid=round(float(ok_s.float().mean()), 4),
            comp_valid=round(float(ok_c.float().mean()), 4),
            comp_valid_modeltable=round(float(ok_c_strict.float().mean()), 4),
            both_valid=round(float((ok_s & ok_c).float().mean()), 4),
            no_overlap=round(float(comp['no_overlap'].float().mean()), 4),
            connected=round(float(comp['connected'].float().mean()), 4),
            vpa_mean=round(float(comp['vpa'].mean()), 3),
            n_atoms_mean=round(float(mask.sum(1).float().mean()), 2),
            sg_selfagree=round(float(sg_agree), 4),
            uniq_frac=round(len(keys) / max(frac.size(0), 1), 4),
            novel_frac=(round(len(novel) / max(len(comp_keys), 1), 4)
                        if train_keys else ''),
            y_head_mean=round(float(out['property'].mean()), 5),
            y_struct_mean=round(float(y_struct.mean()), 5),
        )
        rows.append(row)
        print("  " + "  ".join(f"{k}={v}" for k, v in row.items()))

    rho = _spearman(all_targets, all_struct_y)
    q_t = [r['target_y'] for r in rows]
    q_s = [r['y_struct_mean'] for r in rows]
    slope = float(np.polyfit(q_t, q_s, 1)[0]) if len(q_t) > 1 else float('nan')
    print(f"\n  CONDITIONING CHECK  spearman(target_y, y_struct) = {rho:+.3f}   "
          f"slope of quantile means = {slope:+.3f}")
    print("  A slope near 0 means the conditioning variable is not moving the "
          "generated structures at all, however good the validity numbers look.")
    print("  NOTE: y_head_mean reads the model's own latent and proves nothing. "
          "y_struct_mean re-encodes the generated structure through the frozen "
          "encoder + frozen property head -- still a surrogate, but it at least "
          "depends on the atoms. novel_frac is composition-level only. Neither "
          "replaces a DFT check or a StructureMatcher match-rate.")
    for r in rows:
        r['cond_spearman'] = round(rho, 4)
        r['cond_slope'] = round(slope, 4)
    if results_path:
        _init_metrics_csv(results_path, list(rows[0].keys()))
        for r in rows:
            _append_metrics_csv(results_path, r)
        print(f"  wrote {results_path}")
    return rows



@dataclass
class DirectFlowConfig:
    num_types: int = 89
    atomic_numbers: Optional[list] = None
    hidden: int = 256
    n_sites: int = 20
    sec_dim: int = 64
    cutoff: float = 6.0
    num_rbf: int = 16
    enc_hidden: int = 128
    enc_layers: int = 4
    enc_attn_layers: int = 2
    dec_attn_layers: int = 2
    conv_type: str = 'equivariant'
    pos_freqs: int = 6
    use_pos_features: bool = True
    count_sharpness: float = 4.0
    elem_feat_dim: int = 0
    latent_noise_std: float = 0.15
    cond_dim: int = 128
    flow_hidden: int = 192
    flow_layers: int = 4
    t_dim: int = 64
    decoder_dropout: float = 0.1
    use_refiner: bool = True
    refiner_steps: int = 2
    refiner_scale: float = 0.3
    cfg_dropout: float = 0.15
    cfg_sg_dropout: float = 0.15
    guidance: float = 1.5
    sg_guidance: float = 1.0
    sample_solver: str = 'rk4'
    sample_sg_from_prior: bool = True
    # OFF: the cell is fixed before conflict resolution now, so rescaling it
    # afterwards would shrink distances that were just certified clash-free.
    rescale_volume_after_expand: bool = False
    surrogate_max_deg: int = 12
    surrogate_max_self: int = 6
    max_asym_atoms: Optional[int] = None
    elem_feat_table: Optional[list] = None
    repulsion_scale: float = 0.70
    # TRAINING density band. 5.0 A^3/atom is below any real crystal (diamond
    # is 5.7, the densest metals ~10) and the lattice head, out of
    # distribution on rolled-out latents, pinned V to it -- which guarantees
    # severe overlap and made `hardcore_repulsion` the dominant gradient in
    # stage 3 (measured: ~80,712 of the decoder's gradient against ~15,878
    # for the reconstruction anchor, ~94:1 decoder-to-flow overall).
    vpa_floor: float = 10.0
    vpa_ceiling: float = 60.0
    # SAMPLING density band, deliberately wider than the window
    # structural_validity accepts, so that vpa_ok is a measurement rather
    # than a restatement of the training clamp.
    sample_vpa_floor: float = 3.0
    sample_vpa_ceiling: float = 120.0
    # Hard-sphere floor used by orbit_multiplicity's merge test, matching the
    # scale of the expansion guard's own coincidence check.
    min_contact_abs: float = 0.75
    stage3_gen_warmup: int = 2500
    use_wyckoff: bool = True
    # One rank-0 (fixed-point) Wyckoff class per crystal at generation time.
    # Two tokens on the same fixed point ARE two atoms at one coordinate, and
    # no downstream pass can separate them -- see assign_unique_wyckoff.
    unique_wyckoff: bool = True
    # Strength (logit units) of the empirical p(class | space group) used only
    # to break ties in that assignment.
    wyckoff_prior_weight: float = 0.5
    # Size the cell from the exact Wyckoff multiplicity table rather than from
    # SiteMultiplicityHead's regression of it.
    use_wyckoff_multiplicity: bool = True
    # Candidate positions tried inside a site's own Wyckoff subspace before it
    # is dropped for overlapping something already placed.
    site_relocate_tries: int = 32
    # How many times the resolver may re-try a crystal at a lower density
    # before it starts giving up sites, and by how much it lowers it.
    resolve_passes: int = 3
    resolve_inflate: float = 1.6
    # Stage-3 cost controls. property_every_k amortises the structure surrogate
    # (the dominant term); stage3_prefix_solver is the integrator used on the
    # part of the rollout that carries no gradient.
    property_every_k: int = 4
    stage3_prefix_solver: str = 'euler'
    # Print a per-component timing table for one batch before stage 3 starts.
    stage3_profile: bool = False
    wyckoff_warmup: int = 3000
    wyckoff_teacher: float = 1.0
    gen_anchor_jitter: float = 0.0
    decode_asymmetric_unit: bool = True
    expand_max_atoms: int = 400
    expand_max_ops: int = 8
    # Repulsion: which symmetry operations the term is allowed to see.
    # 'nearest' ranks operations by the displacement they induce on the site,
    # so the operation that creates a short contact is always included; a
    # uniform sample found it a small minority of the time in groups with
    # many operations.
    # 8, not 16: with 'nearest' selection the operation that creates a short
    # contact is ranked first or second, so the extra eight contribute almost
    # nothing while doubling the (B, A, n_ops*A, 27) distance tensor that
    # `hardcore_repulsion` builds -- profiled at 0.49 s of a 2.7 s forward.
    repulsion_ops: int = 8
    repulsion_op_select: str = 'nearest'
    # Post-hoc composition repair, OFF by default: both rewrite elements after
    # the model has produced them, so any validity number measured with them on
    # describes the repair pass, not the model.
    smact_balance_on_sample: bool = False
    unique_type_cap: int = 0
    reconcile_types_on_sample: bool = True
    sym_snap_tol: float = SYM_SNAP_TOL
    # SMACT screen configuration -- must match `composition_validity`
    charge_eneg_weight: float = 1.0
    smact_include_alloys: bool = True
    smact_exempt_unary: bool = True
    min_elements: float = 2.0
    # Property conditioning
    # Orbit expansion inside property_from_structure. This is by far the most
    # expensive thing in stage 3 -- it runs the encoder on n_ops * n_sites atoms
    # with gradient. Measured
    # s/step at B=8: ops=8 -> 3.74, ops=4 -> 1.73, ops=3 -> 1.35, ops=2 -> 1.20,
    # expansion off -> 0.86. Four images already restore most of a site's first
    # coordination shell, which is what the surrogate needs.
    surrogate_expand_orbits: bool = True
    surrogate_expand_ops: int = 4
    # Flow
    flow_occ_weight: float = 3.0
    # Attention key-bias in LatentSetFlow suppressing keys likely to be
    # unoccupied slots (hard ground-truth mask in teacher-forced training,
    # soft self-referential mask during rollout/sampling). See
    # DirectCrystalFlow._occupancy_bias.
    flow_occ_mask: bool = True
    # Origin gauge on the per-site position loss. 'auto' = off when the encoder
    # sees absolute coordinates (use_pos_features), since the latent then
    # carries the origin and the gauge only adds noise.
    site_align_gauge: str = 'auto'
    stage3_solver: str = 'rk2'
    stage3_guidance: Optional[float] = None
    # Detach the rolled-out latent before the structural decode so the many
    # validity penalties train the decoder + heads only, and the generator's
    # distribution is set by flow matching (stable at any sampling depth)
    # rather than by back-prop of penalties that all bottom out on a collapsed
    # unary cell. This is the single change that removes the stage-3 collapse.
    stage3_decode_detach: bool = True
    # Steps of the (now decoder-detached) rollout that still carry a flow
    # gradient, used ONLY by the frozen-critic property term. A short window
    # is plenty and keeps back-prop-through-ODE cheap and
    # well conditioned; the old default of 8 (== full depth) was a needless
    # source of high-variance generator gradient.
    stage3_grad_steps: int = 2
    # Linear LR warmup (optimiser steps) at the start of stage 3, on top of the
    # cosine decay. Generative fine-tuning starts from heads that have only ever
    # seen encoder latents; ramping the LR stops the first few hundred steps
    # from kicking them out of distribution before the schedule settles.
    stage3_lr_warmup_steps: int = 300
    stage3_property_scale: float = 1.5
    recon_every_k: int = 1
    set_assignment: bool = True
    assign_size_weight: float = 0.25
    # Keep the Hungarian matching in composition_set_assignment from
    # contradicting the per-sample (y, sg) conditioning used elsewhere in
    # stage3_losses.
    assign_respect_conditioning: bool = True
    assign_y_weight: float = 0.5
    assign_sg_penalty: float = 10.0
    # Stage 3 optimises the flow and the decoder together. Measured gradient
    # mass into the decoder is ~94x the flow's, so one shared lr is set by
    # whatever keeps the decoder stable and starves the generator.
    stage3_decoder_lr_scale: float = 0.25
    # property_rank: margin (in normalised y units) a pair must clear before
    # its ranking loss goes to zero.
    rank_margin: float = 0.15
    type_reverse_weight: float = 0.0
    stage3_epochs: int = 60
    stage3_lr: float = 3e-5
    stage3_patience: int = 15
    stage3_steps: int = 8
    translation_invariant_recon: bool = True
    align_iters: int = 3
    refiner_project_sites: bool = True
    validity_overlap_scale: float = 0.5
    validity_min_dist: float = 0.75
    generation_quantiles: Optional[list] = None
    generation_n_per: int = 32
    generation_temperature: float = 0.0
    generation_steps: int = 100
    results_path: Optional[str] = None
    aug_coord_noise: float = 0.02
    sg_pos_grad: bool = True
    weights: Dict[str, float] = field(default_factory=lambda: dict(
        chamfer=0.3, type=0.1,
        site_type=2.0, site_pos=1.0,
        count=1.0,
        mult=1.0,
        property=1.5, property_struct=3.0, lattice=1.0,
        wyckoff=1.5, wyckoff_ce=1.0,
        sg=0.1, sg_fine=0.1, recon=1.0,
        repulsion=1.5, charge=2.0, vpa=1.0, sg_align=1.0,
        composition=2.0, property_rank=2.0))


    encoder_pretrain_monitor: tuple = (
        'chamfer_gated', 'site_pos', 'site_type', 'lattice', 'count',
        'wyckoff_ce', 'property')

def config_from_dict(d, **overrides):
    import dataclasses
    names = {f.name for f in dataclasses.fields(DirectFlowConfig)}
    kw = {}
    for k, v in (d or {}).items():
        if k in names and v is not None:
            kw[k] = v
    for k, v in overrides.items():
        if v is not None:
            kw[k] = v
    weights = kw.pop('weights', None)
    cfg = DirectFlowConfig(**kw)
    if isinstance(weights, dict):
        merged = dict(DirectFlowConfig().weights)
        merged.update(weights)
        cfg.weights = merged
    return cfg

# Where the mined MP-20 dataset lives (train/ val/ test/ subfolders). 

DEFAULT_DATA_ROOT = os.environ.get("MP20_ROOT", os.path.join("data", "MP20"))

DEFAULT_CKPT_DIR = os.environ.get("EMF_CKPT_DIR", "checkpoints")


def default_config():
    return {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'dataset_path': DEFAULT_DATA_ROOT,
        'target_name': 'formation_energy_per_atom',
        'results_path': 'test_results.csv',
        'num_workers': 0,
        'batch_size': 32,

        'flow_epochs': 80,
        'flow_lr': 2e-4,
        'flow_ckpt': 'direct_flow.pt',
        'flow_patience': 10,

        'stage3_epochs': 50,
        'stage3_lr': 1e-4,
        'stage3_ckpt': 'stage3_finetune.pt',
        'stage3_metrics_path': None,

        'n_sites': 20,
        'pos_freqs': 6,
        'use_pos_features': True,
        'encoder_pretrain_monitor': ('chamfer_gated', 'site_pos', 'site_type',
                                     'lattice', 'count', 'wyckoff_ce',
                                     'property'),
        'dataset_max_atoms_filter': None,
        'expand_orbits_on_generation': True,
        'decode_asymmetric_unit': True,
        'expand_max_ops': 8,
        # TRAINING density band. 10.0, matching DirectFlowConfig: a 5.0 floor
        # sits below every real crystal (diamond is 5.7), and the lattice head,
        # out of distribution on rolled-out latents, pins the cell volume to
        # whatever the floor is -- which then makes hardcore_repulsion the
        # dominant gradient in stage 3 and emits cells several times too dense.
        'vpa_floor': 10.0,
        'vpa_ceiling': 60.0,
        'stage3_gen_warmup': 2500,
        'generation_quantiles': [0.05, 0.25, 0.50, 0.75, 0.95],
        'generation_n_per': 32,
        'generation_temperature': 0.0,
        'use_wyckoff': True,
        'translation_invariant_recon': True,
        'align_iters': 3,
        'refiner_project_sites': True,
        'validity_overlap_scale': 0.5,
        'validity_min_dist': 0.75,
        'repulsion_scale': 0.70,
        'expand_max_atoms': 400,
        'stage3_guidance': None,
        'rescale_volume_after_expand': False,
        'resolve_passes': 3,
        'resolve_inflate': 1.6,
        'property_every_k': 4,
        'stage3_prefix_solver': 'euler',
        'stage3_profile': False,
        'smact_balance_on_sample': False,
        'unique_type_cap': 0,
        'reconcile_types_on_sample': True,
        'sym_snap_tol': SYM_SNAP_TOL,
        'flow_occ_weight': 3.0,
        'flow_occ_mask': True,
        'site_align_gauge': 'auto',
        'charge_eneg_weight': 1.0,
        'surrogate_expand_orbits': True,
        'surrogate_expand_ops': 4,
        'sample_vpa_floor': 3.0,
        'sample_vpa_ceiling': 120.0,
        'min_contact_abs': 0.75,
        'repulsion_ops': 8,
        'repulsion_op_select': 'nearest',
        'assign_respect_conditioning': True,
        'assign_y_weight': 0.5,
        'assign_sg_penalty': 10.0,
        'stage3_decoder_lr_scale': 0.25,
        'rank_margin': 0.15,
        'unique_wyckoff': True,
        'wyckoff_prior_weight': 0.5,
        'use_wyckoff_multiplicity': True,
        'site_relocate_tries': 32,
        'gen_anchor_jitter': 0.0,
        'sg_pos_grad': True,
        'decoder_dropout': 0.1,
        'use_refiner': True,
        'refiner_steps': 2,
        'refiner_scale': 0.3,
        'aug_coord_noise': 0.02,
        'enc_layers': 4,
        'enc_attn_layers': 2,
        'dec_attn_layers': 2,
        'count_sharpness': 4.0,
        'stage3_patience': 10,
        'stage3_rollout_steps': 16,
        'stage3_grad_steps': 2,  # flow-gradient window (property term only)
        'stage3_decode_detach': True,   # structural penalties train decoder, not flow
        'stage3_lr_warmup_steps': 300,
        'stage3_property_scale': 1.5,
        'recon_every_k': 1,
        'set_assignment': True,
        'assign_size_weight': 0.25,
        'type_reverse_weight': 0.0,
        'run_stage3_finetune': True,
        'node_features': None,
        'edge_features': None,
        'num_atom_types': None,
        'atomic_numbers': None,
    }

def main(config: dict, encoder_path=None):
    """Train a direct flow matching model for crystal generation.

    Args:
        config: Configuration dictionary
        encoder_path: Optional path to a DirectCrystalFlow checkpoint whose
                      encoder (+ enc_proj/enc_standardizer) is already
                      pretrained (model._encoder_pretrained == True). If
                      absent or not pretrained, the encoder is pretrained
                      from scratch here via run_encoder_pretrain before any
                      flow training happens.
    """
    config.setdefault('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(config['device'])

    print("Loading datasets...")
    all_atomic_numbers = set()
    for split in ['train', 'val', 'test']:
        cfg_path = os.path.join(config['dataset_path'], split, f"{split}_config.json")
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                cfg_split = json.load(f)
            all_atomic_numbers.update(cfg_split["atomic_numbers"])
    global_atomic_numbers = sorted(all_atomic_numbers)
    global_atomic_to_idx = {num: idx for idx, num in enumerate(global_atomic_numbers)}
    num_global_atom_types = len(global_atomic_numbers)

    onehot_g = np.eye(num_global_atom_types, dtype=np.float32)
    merged_props, split_cfgs = {}, {}
    for split in ['train', 'val', 'test']:
        cfg_path = os.path.join(config['dataset_path'], split, f"{split}_config.json")
        if not os.path.exists(cfg_path):
            continue
        try:
            with open(cfg_path) as _f:
                _c = json.load(_f)
            split_cfgs[split] = _c
            for _k, _v in (_c.get('element_properties', {}) or {}).items():
                merged_props.setdefault(str(_k), _v)
        except Exception:
            continue
    phys_g = build_element_phys_features(global_atomic_numbers, merged_props)
    global_node_vectors = np.concatenate([onehot_g, phys_g], axis=1)
    print(f"  Global atom types: {num_global_atom_types}")
    print(f"  Node feature dim:  {global_node_vectors.shape[1]}")

    _ds_max_atoms = config.get('dataset_max_atoms_filter', None)
    if isinstance(_ds_max_atoms, (int, float)) and _ds_max_atoms <= 0:
        _ds_max_atoms = None
    if _ds_max_atoms is None:
        config['dataset_max_atoms_filter'] = None


    _n_sites = int(config.get('n_sites') or 0)
    if _n_sites <= 0:
        _n_sites = int(config.get('secondary_caps', 8)) * \
            int(config.get('max_atoms_per_motif', 4))
    config['n_sites'] = _n_sites
    tr_ds = MultiFileGraphDataset(
        config['dataset_path'], config['target_name'], 'train',
        global_atomic_to_idx, global_node_vectors,
        n_sites=_n_sites, max_atoms=_ds_max_atoms)
    val_ds = MultiFileGraphDataset(
        config['dataset_path'], config['target_name'], 'val',
        global_atomic_to_idx, global_node_vectors,
        n_sites=_n_sites, max_atoms=_ds_max_atoms)
    te_ds = MultiFileGraphDataset(
        config['dataset_path'], config['target_name'], 'test',
        global_atomic_to_idx, global_node_vectors,
        n_sites=_n_sites, max_atoms=_ds_max_atoms)

    s0 = tr_ds[0]
    config['node_features'] = int(global_node_vectors.shape[1])
    config['edge_features'] = int(s0.edge_attr.size(1))
    config['num_atom_types'] = num_global_atom_types
    config['atomic_numbers'] = global_atomic_numbers
    print(f"  Train: {len(tr_ds)}  Val: {len(val_ds)}  Test: {len(te_ds)}")

    nw = int(config.get('num_workers', 0))
    bs = int(config.get('batch_size', 64))
    tr_l = DataLoader(tr_ds, bs, shuffle=True, collate_fn=_pyg_collate, num_workers=nw, pin_memory=True)
    v_l = DataLoader(val_ds, bs, shuffle=False, collate_fn=_pyg_collate, num_workers=nw, pin_memory=True)
    te_l = DataLoader(te_ds, bs, shuffle=False, collate_fn=_pyg_collate, num_workers=nw, pin_memory=True)

    print("\nBuilding DirectCrystalFlow model...")
    print("  Architecture: noise + (y, sg) -> LatentSetFlow -> site tokens -> "
          "SiteDecoder -> asymmetric unit -> cell")
    print(f"  Latent: {_n_sites} site tokens x {config.get('sec_dim', 64)} dims "
          f"(plain GNN encoder, no capsules / routing / gate)")
    print("  The encoder is frozen after stage 1; it supplies the flow-matching "
          "targets and the grounded property readout.")
    cfg = config_from_dict(
        config,
        num_types=num_global_atom_types,
        atomic_numbers=global_atomic_numbers,
        n_sites=_n_sites,
        # phys_g is folded into the node vectors AND handed to the encoder;
        # without elem_feat_table the encoder would only see the type index.
        elem_feat_dim=int(phys_g.shape[1]),
        elem_feat_table=phys_g.astype(np.float32),
    )

    model = DirectCrystalFlow(cfg).to(device)
    total = sum(p.numel() for p in model.parameters())
    # Train split only: val/test would leak held-out geometry. Classes seen
    # only in val/test are dropped by wyckoff_class_ce, which reports coverage.
    model.decoder.install_wyckoff_codebook([tr_ds])
    print(f"  Total parameters: {total:,}")

    model.fit_property(torch.as_tensor(tr_ds.targets, dtype=torch.float32))
    _sgc = np.bincount(np.asarray(tr_ds.space_groups, dtype=np.int64).clip(1, 230) - 1,
                       minlength=230)
    model.set_sg_class_weights(_sgc)
    print(f"  Space-group prior: {int((_sgc > 0).sum())}/230 groups present, "
          f"P1 share {_sgc[0] / max(_sgc.sum(), 1):.1%} "
          f"(class weights applied to the sg / sg_align losses)")
    print(f"  Property normalization: mean={float(model.y_mean):.4f}  "
          f"std={float(model.y_std):.4f}  (from train split '{config['target_name']}')")


    loaded_pretrained = False
    if encoder_path and os.path.exists(encoder_path):
        print(f"  Loading checkpoint from {encoder_path}")
        load_checkpoint(encoder_path, model, map_location=device, strict=False)
        if bool(model._encoder_pretrained):
            loaded_pretrained = True
            # load_state_dict does NOT restore requires_grad, so the freeze has
            # to be reapplied here or stage 3 will train the encoder.
            model.freeze_encoder()
            print("  Loaded encoder is pretrained; skipping pretrain_encoder "
                  "(encoder/property re-frozen after load).")
        else:
            print("  [WARN] Loaded checkpoint's encoder is NOT marked pretrained "
                  "-- pretraining from scratch.")

    if not loaded_pretrained:
        print("\nPretraining encoder (deterministic, no KL) + decoder + "
              "property (real z, true y)...")
        model = run_encoder_pretrain(
            model, tr_l, v_l,
            epochs=config.get('encoder_pretrain_epochs', 50),
            lr=config.get('encoder_pretrain_lr', 1e-4),
            weight_decay=config.get('encoder_pretrain_wd', 1e-5),
            patience=config.get('encoder_pretrain_patience', 20),
            monitor=config.get('encoder_pretrain_monitor',
                               tuple(cfg.encoder_pretrain_monitor)),
            ckpt=config.get('encoder_ckpt', 'encoder_pretrained.pt'),
            metrics_path=config.get('encoder_metrics', 'encoder_pretrain_metrics.csv'),
            device=device,
        )
        save_checkpoint(config.get('encoder_ckpt', 'encoder_pretrained.pt'), model)


    print("\nPhase 1/2: flow-matching pretraining (model.flow_losses)...")
    model, _flow_hist = run_direct_flow(
        model, tr_l, v_l,
        epochs=config.get('flow_epochs', config.get('stage3_epochs', 80)),
        lr=config.get('flow_lr', config.get('stage3_lr', 2e-4)),
        device=device,
        ckpt=config.get('flow_ckpt', 'direct_flow.pt'),
        patience=config.get('flow_patience', 0),
    )

    
    try:
        _db0 = next(iter(v_l if v_l is not None else tr_l))
        if not isinstance(_db0, dict):
            _db0 = _pyg_batch_to_dict(_db0.to(device), device,
                                      num_types=model.cfg.num_types)
        else:
            _db0 = to_device(_db0, device)
        print("\n  Post-DF flow diagnostics (pre-stage-3, 100-step rollout):")
        model.diagnostics(_db0, rollout_steps=100, verbose=True)
    except Exception as _e0:
        print(f"  [WARN] post-DF diagnostics failed: {type(_e0).__name__}: {_e0}")


    if bool(config.get('run_stage3_finetune', True)):
        print("\nPhase 2/2: generative fine-tuning (model.stage3_losses, "
              "rollout through the decoder)...")
        model, _stage3_hist = run_stage3_finetune(
            model, tr_l, v_l,
            epochs=config.get('stage3_epochs', 30),
            lr=config.get('stage3_lr', 3e-5),
            rollout_steps=config.get('stage3_rollout_steps', 8),
            device=device,
            ckpt=config.get('stage3_ckpt', 'stage3_finetune.pt'),
            patience=config.get('stage3_patience', 15),
            metrics_path=config.get('stage3_metrics_path', None),
            full_depth_diag_every=config.get('full_depth_diag_every', 5),
            full_depth_diag_steps=config.get('full_depth_diag_steps', 100),
        )
    else:
        print("\nconfig['run_stage3_finetune'] is False -- skipping generative "
              "fine-tuning. The decoder, lattice, sg and property heads stay "
              "where pretrain_encoder_losses left them; all of them have real "
              "supervision there, but none of them has ever seen a latent that "
              "came out of the flow rather than out of the encoder.")

    print("\nDirect flow model training complete.")


    try:
        _db = next(iter(v_l if v_l is not None else tr_l))
        if not isinstance(_db, dict):
            _db = _pyg_batch_to_dict(_db.to(device), device,
                                     num_types=model.cfg.num_types)
        else:
            _db = to_device(_db, device)
        model.diagnostics(_db, verbose=True)
    except Exception as e:
        print(f"  [WARN] diagnostics failed: {type(e).__name__}: {e}")


    try:
        model.eval()
        te_vals = []
        with torch.no_grad():
            for batch in te_l:
                if isinstance(batch, dict):
                    bd = to_device(batch, device)
                else:
                    batch = batch.to(device)
                    bd = _pyg_batch_to_dict(batch, device,
                                            num_types=model.cfg.num_types)
                l, c = pretrain_encoder_losses(model, bd)
                if torch.isfinite(l):
                    te_vals.append(c)
        if te_vals:
            keys = sorted(set().union(*[d.keys() for d in te_vals]))
            agg = {k: float(np.mean([d[k] for d in te_vals if k in d])) for k in keys}
            print("  Held-out TEST autoencoder metrics:")
            print("    " + "  ".join(f"{k}={agg[k]:.4f}" for k in keys))
    except Exception as e:
        print(f"  [WARN] test-split evaluation failed: {type(e).__name__}: {e}")

    try:
        generate_and_evaluate(
            model, tr_ds.targets, device=device,
            quantiles=config.get('generation_quantiles'),
            n_per=int(config.get('generation_n_per', 32)),
            steps=int(config.get('generation_steps', 100)),
            temperature=float(config.get('generation_temperature', 0.0)),
            results_path=config.get('results_path'),
            train_dataset=tr_ds)
    except Exception as e:
        print(f"  [WARN] generation/evaluation failed: {type(e).__name__}: {e}")

    return model

def run(data_root=None, mount=False, ckpt_dir=None,
         s1_epochs=None, s2_epochs=None, s3_epochs=None, s3_lr=None,
         flow_epochs=None, flow_lr=None,
         batch_size=None, n_sites=None, n_caps=None, K=None,
         target=None, device=None,
         encoder_path=None,
         **cfg_overrides):
    """
    s3_epochs/s3_lr now drive the generative fine-tune phase specifically
    (model.stage3_losses via run_stage3_finetune, previously dead code).
    flow_epochs/flow_lr drive the flow-matching pretraining phase (model.
    flow_losses via run_direct_flow) that s3_epochs/s3_lr used to (mis)drive.
    """
    if mount:
        try:
            from google.colab import drive
            drive.mount('/content/drive')
        except Exception as e:
            print(f"(not in Colab or mount failed: {e})")
    data_root = data_root or DEFAULT_DATA_ROOT
    if ckpt_dir is None:
        ckpt_dir = DEFAULT_CKPT_DIR
    os.makedirs(ckpt_dir, exist_ok=True)
    config = default_config()
    explicit = dict(dataset_path=data_root, ckpt_dir=ckpt_dir)
    for key, val in (('target_name', target), ('device', device),
                     ('stage3_epochs', s3_epochs), ('stage3_lr', s3_lr),
                     ('flow_epochs', flow_epochs), ('flow_lr', flow_lr),
                     ('batch_size', batch_size), ('n_sites', n_sites),
                     ('secondary_caps', n_caps),
                     ('max_atoms_per_motif', K)):
        if val is not None:
            explicit[key] = val
    if encoder_path is not None:
        explicit['encoder_path'] = encoder_path
    config.update(explicit)
    for _k, _f in (('encoder_ckpt', 'encoder_pretrained.pt'),
                   ('flow_ckpt', 'direct_flow.pt'),
                   ('stage3_ckpt', 'stage3_finetune.pt')):
        config[_k] = os.path.join(ckpt_dir, _f)
    config.update(cfg_overrides)
    print(f"device: {config['device']} | checkpoints -> {ckpt_dir}\n")
    return main(config, encoder_path=encoder_path)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(
        description="Train DirectCrystalFlow on the mined MP-20 dataset.")
    ap.add_argument("--data-root", default=DEFAULT_DATA_ROOT,
                    help="dir with train/ (val/ test/) subfolders")
    ap.add_argument("--encoder-path", default=None,
                    help="Optional checkpoint with a pretrained encoder. If omitted, the encoder "
                         "is pretrained from scratch first (Stage 1).")
    ap.add_argument("--target", default='formation_energy_per_atom')
    ap.add_argument("--flow-epochs", type=int, default=80,
                    help="phase 1: flow-matching pretraining epochs")
    ap.add_argument("--flow-lr", type=float, default=2e-4)
    ap.add_argument("--s3-epochs", type=int, default=30,
                    help="phase 2: generative fine-tuning epochs "
                         "(model.stage3_losses, previously dead code)")
    ap.add_argument("--s3-lr", type=float, default=1e-4)
    ap.add_argument("--s3-steps", type=int, default=8)
    ap.add_argument("--no-stage3", action="store_true",
                    help="skip phase 2 (generative fine-tuning) entirely")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--n-sites", type=int, default=None,
                    help="latent tokens = max sites in the asymmetric unit")
    ap.add_argument("--device", default=None)
    args, _ = ap.parse_known_args()

    config = default_config()
    os.makedirs(DEFAULT_CKPT_DIR, exist_ok=True)
    for _k, _f in (('encoder_ckpt', 'encoder_pretrained.pt'),
                   ('flow_ckpt', 'direct_flow.pt'),
                   ('stage3_ckpt', 'stage3_finetune.pt')):
        config[_k] = os.path.join(DEFAULT_CKPT_DIR, _f)
    config['dataset_path'] = args.data_root
    config['target_name'] = args.target
    config['device'] = args.device or config['device']
    config['batch_size'] = args.batch_size
    config['flow_epochs'] = args.flow_epochs
    config['flow_lr'] = args.flow_lr
    config['stage3_epochs'] = args.s3_epochs
    config['stage3_lr'] = args.s3_lr
    config['stage3_rollout_steps'] = args.s3_steps
    config['run_stage3_finetune'] = not args.no_stage3
    if args.n_sites is not None:
        config['n_sites'] = args.n_sites
    main(config, encoder_path=args.encoder_path)
