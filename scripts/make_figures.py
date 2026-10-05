"""Figure panels for the manuscript, generated directly from the evaluation log.

    python make_figures.py [emf_eval_results.json]

Every plotted number is read from the JSON (emf_eval, 2026-09-29, stage3_finetune (43)).
The only constant not in the log is the test-set median VPA (TEST_VPA_MEDIAN), a property of
the MP-20 test split carried over from the earlier evaluation, and the MP-20 compositional
validity reference line (from the literature)."""
import json, sys, os
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from math import sqrt

JSON = sys.argv[1] if len(sys.argv) > 1 else "emf_eval_results.json"
R = json.load(open(JSON))
os.makedirs("figures", exist_ok=True)

TEST_VPA_MEDIAN = 18.2      # A^3/atom, MP-20 test split (not in the log)
MP20_COMP_VALIDITY = 90.65  # % literature reference line (not in the log)

plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans", "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.bbox": "tight", "savefig.dpi": 300})
C1, C2, C3, CG = "#1f5aa6", "#d1603d", "#3a9a5b", "#8c8c8c"

def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z*z/n; c = (p + z*z/(2*n)) / d; h = z*sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return 100*p, 100*(p-(c-h)), 100*((c+h)-p)
def tag(ax, s): ax.text(-0.18, 1.04, s, transform=ax.transAxes, fontweight="bold", fontsize=10)

N_GEN = R["n_generated"]
COMP = 100 * R["n_comp_valid"] / N_GEN
STRUCT = 100 * R["n_struct_valid"] / N_GEN

# ---------------- Figure 2: unconditional generation ----------------
cov = R["coverage"]
fig, axs = plt.subplots(1, 4, figsize=(7.4, 2.2), gridspec_kw=dict(wspace=0.6))
ax = axs[0]
lab = ["Struct.", "Comp.", "Training\ncriterion"]
val = [STRUCT, COMP, 100 * R["model_gate_validity"]]
ax.bar(range(3), val, color=[C1, C2, C3], width=0.7)
ax.axhline(MP20_COMP_VALIDITY, ls="--", lw=0.7, color="k")
ax.text(-0.45, 103, "MP-20 comp. validity", fontsize=5.5, ha="left")
ax.set_xticks(range(3)); ax.set_xticklabels(lab, fontsize=6.5, rotation=35); ax.set_ylim(0, 110)
ax.set_ylabel("Rate (%)")
for i, v in enumerate(val): ax.text(i, v/2, f"{v:.1f}", ha="center", fontsize=6, color="w", rotation=90)
tag(ax, "a")

ax = axs[1]
st = R["stability"]
vals = [st["vpa_before_median"], st["vpa_after_median"], TEST_VPA_MEDIAN]
ax.bar(range(3), vals, color=[C2, C2, C1], width=0.7)
ax.patches[1].set_hatch("////"); ax.patches[1].set_edgecolor("w")
ax.set_xticks(range(3)); ax.set_xticklabels(["Gen.", "Gen.,\nrelaxed", "MP-20\ntest"], fontsize=6.5)
ax.set_ylabel("Median volume per atom (Å$^3$)"); ax.set_ylim(0, 33)
for i, v in enumerate(vals): ax.text(i, v + 0.6, f"{v:.1f}", ha="center", fontsize=6)
tag(ax, "b")

ax = axs[2]
keys = ["ds0.2", "ds0.4", "ds0.6"]
thr = ["0.2/5", "0.4/10", "0.6/15"]
ax.plot(thr, [100*cov[k]["cov_r"] for k in keys], "o-", color=C1, ms=3, label="COV-R")
ax.plot(thr, [100*cov[k]["cov_p"] for k in keys], "s-", color=C2, ms=3, label="COV-P")
ax.axvline(1, color=CG, lw=0.6, ls=":"); ax.set_ylim(0, 105)
ax.set_xlabel(r"$\delta_{\rm struct}/\delta_{\rm comp}$"); ax.set_ylabel("Coverage (%)")
ax.legend(frameon=False, fontsize=6, loc="lower right"); tag(ax, "c")

ax = axs[3]
so = R["site_occupancy"]
sp = [100*so["generated"]["frac_atoms_special"], 100*so["reference"]["frac_atoms_special"]]
ax.bar([0, 1], sp, color=[C2, C1], width=0.6)
ax.set_xticks([0, 1]); ax.set_xticklabels(["Gen.", "MP-20"]); ax.set_ylim(0, 100)
ax.set_ylabel("On special positions (%)")
for i, v in enumerate(sp): ax.text(i, v + 2, f"{v:.1f}", ha="center", fontsize=6)
tag(ax, "d")
fig.savefig("figures/fig2_unconditional.pdf")

# ---------------- Figure 3: property-conditioned generation ----------------
BG = R["conditional_fidelity"]["by_guidance"]
WS = [("1.0", C1, "o", "$w_y=1.0$"), ("1.5", C2, "s", "$w_y=1.5$")]
rows = {w: BG[w]["rows"] for w, *_ in WS}
q = np.array([r["quantile"] for r in rows["1.0"]])
tgt = np.array([r["target"] for r in rows["1.0"]])
NPT = rows["1.0"][0]["n"]
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.2), gridspec_kw=dict(wspace=0.5))
ax = axs[0]
lim = [-5.4, 0.8]
ax.plot(lim, lim, "--", color=CG, lw=0.7)
for (w, c, m, lb), dx in zip(WS, (-0.06, 0.06)):
    bias = np.array([r["latent_bias"] for r in rows[w]]); mae = np.array([r["latent_mae"] for r in rows[w]])
    ax.errorbar(tgt + dx, tgt + bias, yerr=mae, fmt=m, color=c, ms=3.5, capsize=2, lw=0.8, label=lb)
ax.legend(frameon=False, fontsize=6, loc="lower right")
ax.set_xlim(-4.4, 0.4); ax.set_ylim(lim)
ax.set_xlabel(r"Target $E_{\rm f}$ (eV atom$^{-1}$)"); ax.set_ylabel(r"Surrogate $\hat E_{\rm f}$ (eV atom$^{-1}$)")
ax.text(-4.3, 0.6, "latent surrogate\n(not independent)", fontsize=5.5, va="top", color=CG)
tag(ax, "a")

ax = axs[1]
for w, c, m, lb in WS:
    ax.plot(q, [r["oracle_mae"] for r in rows[w]], m + "-", color=c, ms=3.5, lw=0.8, label=lb)
ax.axhline(R["conditional_fidelity"]["oracle_holdout_mae"], ls=":", color=CG, lw=0.7)
ax.text(0.97, R["conditional_fidelity"]["oracle_holdout_mae"] + 0.05, "oracle hold-out MAE", fontsize=5.5,
        ha="right", va="bottom", color=CG)
ax.set_xlabel("Target quantile"); ax.set_ylabel(r"Independent-oracle MAE (eV atom$^{-1}$)")
ax.set_ylim(0, 2.4); ax.legend(frameon=False, fontsize=6, loc="upper right"); tag(ax, "b")

ax = axs[2]
for (w, c, m, lb), dx in zip(WS, (-0.008, 0.008)):
    ks = [round(r["valid_frac"] * r["n"]) for r in rows[w]]
    ww = [wilson(k, r["n"]) for k, r in zip(ks, rows[w])]
    ax.errorbar(q + dx, [a[0] for a in ww], yerr=[[a[1] for a in ww], [a[2] for a in ww]], fmt=m + "-", color=c,
                ms=3.5, capsize=2, lw=0.8, label=lb)
ax.axhline(COMP, ls=":", color=CG, lw=0.7); ax.text(0.5, COMP - 2, "unconditional", fontsize=6, ha="center", va="top", color=CG)
ax.set_xlabel("Target quantile"); ax.set_ylabel("Comp. validity (%)"); ax.set_ylim(0, 105)
ax.legend(frameon=False, fontsize=6, loc="lower right"); tag(ax, "c")
fig.savefig("figures/fig3_property.pdf")

# ---------------- Figure 4: space-group control ----------------
NAMES = {2: "$P\\bar{1}$", 14: "$P2_1/c$", 62: "$Pnma$", 148: "$R\\bar{3}$", 194: "$P6_3/mmc$", 225: "$Fm\\bar{3}m$"}
sg = R["sg_controllability"]["rows"]
g = [f"{NAMES[r['requested']]}\n({r['requested']})" for r in sg]
ex = [100 * r["exact"] for r in sg]; exc = [100 * r["control_exact"] for r in sg]
vk = [round(r["valid_frac"] * r["n_analyzed"]) for r in sg]
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.2), gridspec_kw=dict(wspace=0.3))
x = np.arange(len(sg)); w = 0.38
ax = axs[0]
ax.bar(x - w/2, ex, w, color=C1, label="flow conditioned on $g$")
ax.bar(x + w/2, exc, w, color=CG, label="flow conditioned on random $g'\\neq g$")
for i, r in enumerate(sg):
    if r["control_exact"] < 1:
        ax.text(i + w/2, exc[i] + 1, f"{round(r['control_exact']*r['n_analyzed'])}/{r['n_analyzed']}", ha="center", fontsize=5.5)
ax.set_xticks(x); ax.set_xticklabels(g, fontsize=6.5); ax.set_ylabel("spglib group $= g$ (%)"); ax.set_ylim(0, 132); ax.set_yticks([0, 20, 40, 60, 80, 100])
ax.legend(frameon=False, fontsize=6, loc="upper center", ncol=2); tag(ax, "a")
ax = axs[1]
a = [wilson(k, r["n_analyzed"]) for k, r in zip(vk, sg)]
ax.errorbar(x, [t[0] for t in a], yerr=[[t[1] for t in a], [t[2] for t in a]], fmt="o", color=C1, ms=3.5, capsize=2, lw=0.8,
            label="conditioned")
ax.axhline(COMP, ls=":", color=CG, lw=0.7); ax.text(1.5, COMP + 1.5, "unconditional", fontsize=6, ha="center", va="bottom", color=CG)
ax.set_xticks(x); ax.set_xticklabels(g, fontsize=6.5); ax.set_ylabel("Comp. validity (%)"); ax.set_ylim(0, 105)
tag(ax, "b")
fig.savefig("figures/fig4_spacegroup.pdf")
print("ok")
