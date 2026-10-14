#!/usr/bin/env python3
"""Figure 4: retrospective discrimination of the published in-silico filters.

    python figure4_retrospective.py --result results/experiment_c_retrospective.json \
        --designs data/tier1_rfpeptides_per_design.csv --out results/figure4_retrospective.png
"""
from __future__ import annotations

import argparse
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RNG = np.random.default_rng(20261013)   # jitter only; fixed so the figure is reproducible

ORDER = ["contact_molecular_surface_A2", "rosetta_ddg_kcal_mol", "sap", "ipae"]
SHORT = {"contact_molecular_surface_A2": "Contact\nsurface",
         "rosetta_ddg_kcal_mol": "Rosetta\nddG", "sap": "SAP", "ipae": "AF2\niPAE"}
SIGN = {"contact_molecular_surface_A2": +1, "rosetta_ddg_kcal_mol": -1,
        "sap": -1, "ipae": -1}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", required=True)
    ap.add_argument("--designs", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    res = json.load(open(a.result))
    rows = list(csv.DictReader(open(a.designs)))
    tested = [r for r in rows if r["synthesised"] == "True"]

    # ACS Appendix 2: double-column graphics must be 300-504 pt (4.167-7 in) wide,
    # max depth 660 pt (9.167 in) incl. caption, lettering >= 4.5 pt at final size.
    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
                         "font.size": 7, "axes.labelsize": 7,
                         "xtick.labelsize": 6, "ytick.labelsize": 6,
                         "axes.linewidth": 0.6, "lines.linewidth": 1.0,
                         "xtick.major.width": 0.6, "ytick.major.width": 0.6})
    fig, axgrid = plt.subplots(2, 2, figsize=(7.0, 5.4))
    axes = axgrid.ravel()

    # (a) pooled AUROC with bootstrap interval
    ax = axes[0]
    y = np.arange(len(ORDER))[::-1]
    for i, m in enumerate(ORDER):
        d = res["primary"]["metrics"][m]
        v, (lo, hi) = d["auroc_pooled_within_target"], d["ci95"]
        ax.plot([lo, hi], [y[i]] * 2, color="0.35", lw=2, solid_capstyle="round")
        ax.plot(v, y[i], "o", ms=5, color="#1f4e79", zorder=3)
        ax.text(1.03, y[i], f"{v:.2f}", va="center", fontsize=6)
    ax.axvline(0.5, color="crimson", ls="--", lw=1.2)
    ax.set_yticks(y, [SHORT[m].replace("\n", " ") for m in ORDER], fontsize=6)
    ax.set_xlim(0, 1.18)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("AUROC, binder vs non-binder\n(pooled within target)")
    ax.set_title("(a) No filter separates binders", fontsize=7, loc="left")

    # (b) per-target AUROC
    ax = axes[1]
    tgts = sorted(res["primary"]["metrics"][ORDER[0]]["per_target"])
    M = np.array([[res["primary"]["metrics"][m]["per_target"][t]["auroc"]
                   for t in tgts] for m in ORDER])
    im = ax.imshow(M, cmap="RdBu_r", vmin=0, vmax=1, aspect="auto")
    for i in range(len(ORDER)):
        for j, t in enumerate(tgts):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6,
                    color="white" if abs(M[i, j] - 0.5) > 0.3 else "black")
    n = res["primary"]["metrics"][ORDER[0]]["per_target"]
    ax.set_xticks(range(len(tgts)),
                  [f"{t}\n{n[t]['n_binders']}/{n[t]['n']}" for t in tgts], fontsize=6)
    ax.set_yticks(range(len(ORDER)), [SHORT[m] for m in ORDER], fontsize=6)
    ax.set_title("(b) Per target (binders/tested)", fontsize=7, loc="left")
    fig.colorbar(im, ax=ax, fraction=0.046, label="AUROC")

    # (c) the best-performing metric, design by design
    ax = axes[2]
    m = "rosetta_ddg_kcal_mol"
    for j, t in enumerate(tgts):
        g = [r for r in tested if r["target"] == t]
        for r in g:
            b = r["binder_screen"] == "True"
            ax.plot(j + (0.12 if b else -0.12) + RNG.uniform(-0.05, 0.05),
                    float(r[m]), "o" if b else "x", ms=6,
                    color="#c0392b" if b else "0.55",
                    mew=1.4, zorder=3 if b else 2)
    ax.set_xticks(range(len(tgts)), tgts, fontsize=6, rotation=20)
    ax.set_ylabel("Rosetta ddG (kcal/mol)")
    # Deliberately neutral: binders sit high on MCL1 and low on MDM2, which is the point.
    ax.set_title("(c) Best of the four, by target;\n     binders do not cluster",
                 fontsize=7, loc="left")
    ax.plot([], [], "o", color="#c0392b", label="binder")
    ax.plot([], [], "x", color="0.55", label="no binding")
    ax.legend(fontsize=6, frameon=False)

    # (d) censoring: were the tested designs a biased slice?
    ax = axes[3]
    c = res["censoring"]["metrics"]
    x = np.arange(len(ORDER))
    syn = [c[m]["auroc_synthesised_vs_censored"] for m in ORDER]
    ax.bar(x, syn, color="#4a7ba7", width=0.55)
    ax.axhline(0.5, color="crimson", ls="--", lw=1.2)
    for i, m in enumerate(ORDER):
        ax.text(i, syn[i] + 0.02, f"p={c[m]['p_permutation']:.2f}",
                ha="center", fontsize=6)
    ax.set_xticks(x, [SHORT[m] for m in ORDER], fontsize=6)
    ax.set_ylim(0, 0.8)
    ax.set_ylabel("AUROC, synthesised vs censored", fontsize=7)
    ax.set_title(f"(d) Censored designs ({res['censoring']['n_censored']}) score\n"
                 "     like tested ones", fontsize=7, loc="left")

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(pad=0.6, w_pad=1.2, h_pad=1.4)
    fig.savefig(a.out, dpi=300)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
