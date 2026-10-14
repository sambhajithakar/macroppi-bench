#!/usr/bin/env python3
"""Redraw manuscript Figures 1 and 2 from the deposited result tables.

COMMAND LINE AS RUN (Sections 4.1 and 4.2 of the manuscript)

    python3 make_figures.py \
        --gradients ../results/experiment_a_gradient_decomposition.csv \
        --chemistries ../results/experiment_b1_chemistries.csv \
        --outdir ../../manuscript/figures

Both figures are drawn only from the CSVs written by phase1_closure.py and
phase1_chemistries.py; nothing is hand-placed, so a rerun of those analyses
propagates to the figures.

Sizing follows ACS Appendix 2 (Preparing Graphics): double-column graphics are
between 300 and 504 points wide (4.167-7 in), the maximum depth is 660 points
(9.167 in) including the caption, lettering is at least 4.5 points at the final
published size, and lines are at least 0.5 point. Both figures are drawn at
exactly 7 in wide and saved at 300 dpi, the ACS minimum for colour art, so they
are placed at 1:1 with no rescaling. Marker shape and line style, not colour
alone, distinguish the series.
"""
import argparse
import csv
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ACS: >= 4.5 pt lettering at final size. These are the sizes at 7 in wide.
RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 7.5,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.0,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
}

# Okabe-Ito, colour-blind safe; paired with distinct markers so colour is never
# the only carrier of identity.
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
MARKERS = ["o", "s", "^", "D", "v", "P"]

# Pre-registered decision rule for the closure share (Section 3.4).
REFUTED_BAND = 0.2
PROCEED_BAND = 0.5


def read(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def by_key(rows, key):
    out = defaultdict(list)
    for r in rows:
        out[r[key]].append(r)
    for k in out:
        out[k].sort(key=lambda r: float(r["t"]))
    return out


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)


def figure1(rows, out):
    groups = by_key(rows, "structure")
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.55))

    ax = axes[0]
    ax.axhspan(0, REFUTED_BAND, color="0.90", zorder=0)
    ax.axhline(REFUTED_BAND, color="0.45", ls=":", lw=0.8)
    ax.axhline(PROCEED_BAND, color="crimson", ls="--", lw=0.9)
    ax.text(0.40, 0.035, "pre-registered: refuted", fontsize=5.5, color="0.30")
    ax.text(0.02, PROCEED_BAND + 0.02, "pre-registered: proceed", fontsize=5.5,
            color="crimson")
    for i, (name, rs) in enumerate(sorted(groups.items())):
        t = [float(r["t"]) for r in rs]
        f = [float(r["f_closure_mean"]) for r in rs]
        ax.plot(t, f, marker=MARKERS[i % len(MARKERS)], ms=2.6,
                color=PALETTE[i % len(PALETTE)], label=name.split("_")[0])
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("diffusion time $t$")
    ax.set_ylabel("closure share of gradient norm")
    ax.set_title("(a) Closure share stays in the\n     refuted band", loc="left")
    ax.legend(frameon=False, ncol=2, handlelength=1.4, columnspacing=0.8)

    ax = axes[1]
    ax.axhline(1.0, color="0.45", ls=":", lw=0.8)
    for i, (name, rs) in enumerate(sorted(groups.items())):
        t = [float(r["t"]) for r in rs]
        e = [float(r["closure_per_bond_enrichment"]) for r in rs]
        ax.plot(t, e, marker=MARKERS[i % len(MARKERS)], ms=2.6,
                color=PALETTE[i % len(PALETTE)])
    ax.set_yscale("log")
    ax.set_xlabel("diffusion time $t$")
    ax.set_ylabel("enrichment over an ordinary bond")
    ax.set_title("(b) Enrichment is real but\n     bounded", loc="left")

    ax = axes[2]
    for i, (name, rs) in enumerate(sorted(groups.items())):
        t = [float(r["t"]) for r in rs]
        d = [float(r["closure_dist_mean_A"]) for r in rs]
        ax.plot(t, d, marker=MARKERS[i % len(MARKERS)], ms=2.6,
                color=PALETTE[i % len(PALETTE)])
    ax.set_xlabel("diffusion time $t$")
    ax.set_ylabel(r"mean closure distance ($\mathrm{\AA}$)")
    ax.set_title("(c) The ring does open, as the\n     hypothesis assumes", loc="left")

    for a in axes:
        style(a)
    fig.tight_layout(pad=0.5, w_pad=1.3)
    fig.savefig(out, dpi=300)
    print(f"wrote {out}")


def figure2(rows, out):
    groups = by_key(rows, "topology")
    order = [t for t in ("head_to_tail", "disulfide", "bicycle") if t in groups]
    label = {"head_to_tail": "head-to-tail", "disulfide": "disulfide",
             "bicycle": "bicycle"}

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))

    ax = axes[0]
    ax.axhspan(0, REFUTED_BAND, color="0.90", zorder=0)
    ax.axhline(REFUTED_BAND, color="0.45", ls=":", lw=0.8)
    ax.axhline(PROCEED_BAND, color="crimson", ls="--", lw=0.9)
    ax.text(0.02, PROCEED_BAND + 0.02, "pre-registered: proceed", fontsize=5.5,
            color="crimson")
    for i, topo in enumerate(order):
        # average the structures sharing a topology at each time point
        acc = defaultdict(list)
        for r in groups[topo]:
            acc[float(r["t"])].append(float(r["f_closure_mean"]))
        t = sorted(acc)
        f = [sum(acc[k]) / len(acc[k]) for k in t]
        ax.plot(t, f, marker=MARKERS[i], ms=3.0, color=PALETTE[i],
                label=label[topo])
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("diffusion time $t$")
    ax.set_ylabel("closure share of gradient norm")
    ax.set_title("(a) Only the bicycle group crosses into the\n     partial band", loc="left")
    ax.legend(frameon=False, handlelength=1.4)

    ax = axes[1]
    width = 0.36
    xs = range(len(order))
    # Averaged over the pre-registered window t in [0.3, 0.9]. The per-restraint
    # figure is the mean of each structure's own group/n ratio, not the ratio of
    # the means: structures within a topology carry different restraint counts
    # (one disulfide in SFTI-1, three in kalata B1), so the two differ. This is
    # the statistic reported in Table 2.
    grp, per = [], []
    for topo in order:
        win = [r for r in groups[topo] if 0.3 <= float(r["t"]) <= 0.9]
        gs = [float(r["closure_group_enrichment"]) for r in win]
        ps = [float(r["closure_group_enrichment"]) / float(r["n_closure_restraints"])
              for r in win if float(r["n_closure_restraints"])]
        grp.append(sum(gs) / len(gs))
        per.append(sum(ps) / len(ps) if ps else 0.0)
    ax.bar([x - width / 2 for x in xs], grp, width, color=PALETTE[0],
           edgecolor="0.25", linewidth=0.5, label="whole closure group")
    ax.bar([x + width / 2 for x in xs], per, width, color="white",
           edgecolor=PALETTE[1], linewidth=0.9, hatch="///",
           label="per restraint")
    for x, (g, p) in enumerate(zip(grp, per)):
        ax.text(x - width / 2, g, f"{g:.2f}", ha="center", va="bottom", fontsize=5.5)
        ax.text(x + width / 2, p, f"{p:.2f}", ha="center", va="bottom", fontsize=5.5)
    ax.axhline(1.0, color="0.45", ls=":", lw=0.8)
    ax.set_xticks(list(xs), [label[t] for t in order])
    ax.set_ylabel("enrichment over an ordinary bond")
    ax.set_title("(b) Adding restraints raises the group,\n"
                 "     not the restraint", loc="left")
    ax.legend(frameon=False)

    for a in axes:
        style(a)
    fig.tight_layout(pad=0.5, w_pad=1.3)
    fig.savefig(out, dpi=300)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gradients", required=True)
    ap.add_argument("--chemistries", required=True)
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()

    plt.rcParams.update(RC)
    figure1(read(a.gradients), f"{a.outdir}/figure1_closure_diagnostic.png")
    figure2(read(a.chemistries), f"{a.outdir}/figure2_chemistries.png")


if __name__ == "__main__":
    main()
