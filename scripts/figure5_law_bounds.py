#!/usr/bin/env python3
"""Figures 4 and 5 of the JMGM manuscript, drawn only from deposited result tables.
python3 figure5_law_bounds.py --results ../results --outdir ../../manuscript/figures-jmgm
Writes PNG (600 dpi) and vector PDF for each figure."""
import argparse, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
RC = {"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"], "font.size": 7,
      "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "axes.linewidth": 0.6,
      "pdf.fonttype": 42}
C = {"blue": "#0072B2", "verm": "#D55E00", "green": "#009E73", "pink": "#CC79A7", "orange": "#E69F00", "sky": "#56B4E9", "grey": "#666666"}

def fig4(R, out):
    v3 = pd.read_csv(f"{R}/phase6_theory_v3.csv"); v5 = pd.read_csv(f"{R}/phase7b_multi_restraint_v5.csv"); p9 = pd.read_csv(f"{R}/phase9_pdb_population.csv")
    fig, ax = plt.subplots(1, 3, figsize=(7, 2.5))
    a = ax[0]
    a.plot([0.08, 0.3], [0.08, 0.3], color=C["grey"], lw=0.6, ls="--")
    a.scatter(v3.f_pred, v3.f_meas, marker="o", s=14, color=C["blue"], label="One restraint, six sizes")
    a.scatter(v5.f_T5, v5.f_meas, marker="s", s=14, color=C["verm"], label="1 to 4 restraints, v5")
    a.scatter(v5.f_T4, v5.f_meas, marker="^", s=14, facecolors="none", edgecolors=C["grey"], label="1 to 4 restraints, v4")
    a.set_xlabel("Predicted closure share"); a.set_ylabel("Measured closure share"); a.legend(frameon=False, loc="upper left")
    a.text(-0.2, 1.04, "a", transform=a.transAxes, fontweight="bold", fontsize=9)
    b = ax[1]
    b.plot([0.09, 0.22], [0.09, 0.22], color=C["grey"], lw=0.6, ls="--")
    b.scatter(p9.f_T5, p9.f_bondonly, marker="o", s=10, color=C["green"])
    b.set_xlabel("Predicted closure share (v5)"); b.set_ylabel("Measured closure share, bond terms")
    b.text(-0.2, 1.04, "b", transform=b.transAxes, fontweight="bold", fontsize=9)
    c = ax[2]
    big = p9.n >= 10
    c.scatter(p9.n[big], p9.f_allterms[big], marker="o", s=10, color=C["blue"], label="10 or more residues")
    c.scatter(p9.n[~big], p9.f_allterms[~big], marker="D", s=12, color=C["orange"], label="8 or 9 residues")
    c.axhline(0.2, color=C["verm"], lw=0.7, ls="--"); c.axhline(0.5, color=C["grey"], lw=0.7, ls=":")
    c.set_ylim(0, 0.55); c.set_xlabel("Ring size (residues)"); c.set_ylabel("Closure share, all terms"); c.legend(frameon=False, loc="center right")
    c.text(-0.2, 1.04, "c", transform=c.transAxes, fontweight="bold", fontsize=9)
    fig.tight_layout(); fig.savefig(out + ".png", dpi=600); fig.savefig(out + ".pdf"); plt.close(fig)

def fig5(R, out):
    b = pd.read_csv(f"{R}/phase8_bounds.csv"); names = list(b["filter"].unique())
    cols = [C["blue"], C["verm"], C["green"], C["pink"]]
    fig, ax = plt.subplots(1, 4, figsize=(7, 2.3), sharey=True)
    for a, nm, col, lab in zip(ax, names, cols, "abcd"):
        g = b[b["filter"] == nm]
        a.fill_between(g.pi, g.auroc_min, g.auroc_max, color=col, alpha=0.25, lw=0)
        a.plot(g.pi, g.auroc_min, color=col, lw=0.8); a.plot(g.pi, g.auroc_max, color=col, lw=0.8)
        a.plot(g.pi, g.auroc_random_mean, color=col, lw=0.9, ls=":", marker="o", ms=2.5)
        a.axhline(0.5, color=C["grey"], lw=0.6, ls="--"); a.axhline(g.observed_tested_only.iloc[0], color="black", lw=0.6)
        a.set_title(nm.split(" (")[0], fontsize=7); a.set_ylim(0.1, 0.9)
        a.text(-0.05, 1.12, lab, transform=a.transAxes, fontweight="bold", fontsize=9)
    ax[0].set_ylabel("Pooled within-target AUROC"); fig.supxlabel("Binder prevalence among the 38 untested designs", fontsize=7)
    fig.tight_layout(); fig.savefig(out + ".png", dpi=600); fig.savefig(out + ".pdf"); plt.close(fig)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--results", required=True); ap.add_argument("--outdir", required=True); o = ap.parse_args()
    plt.rcParams.update(RC); os.makedirs(o.outdir, exist_ok=True)
    fig4(o.results, f"{o.outdir}/figure4_law"); fig5(o.results, f"{o.outdir}/figure5_bounds")
