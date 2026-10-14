#!/usr/bin/env python3
"""
Experiment C (manuscript Section 4.3): do the published in-silico filters separate
macrocyclic binders from non-binders?

COMMAND LINE AS RUN
-------------------
    python3 phase2_retrospective.py --designs data/tier1_rfpeptides_per_design.csv \
        --outdir results --n-boot 10000 --n-perm 10000 --seed 20261013

QUESTION
--------
RFpeptides selected designs for synthesis using four quantities, all computed before any
experiment: contact molecular surface, Rosetta ddG, SAP (surface aggregation propensity)
and AlphaFold interface PAE. Those quantities are published per design, and so is which
designs were made and which ones bound. That makes it possible to ask, after the fact, how
much each filter was actually worth -- a question the design papers do not ask of
themselves, because they report only the designs that succeeded.

DESIGN OF THE TEST
------------------
Population: the 39 designs that were synthesised and assayed. A design that was never
synthesised has no binding label, and is excluded from the discrimination analysis rather
than counted as a failure (see CENSORING below).

Primary statistic: AUROC, computed WITHIN each target and pooled over within-target pairs
only. Pooling designs across targets would let a target with an easy interface and many
binders masquerade as predictive signal; restricting to pairs that share a target removes
that confound. With 14 binders over 4 targets this is a small-sample estimate and is
reported with a bootstrap interval, not as a point value.

Null: labels permuted within target. The reported p is the fraction of permutations whose
pooled AUROC is at least as extreme, two-sided about 0.5.

Direction: each metric is oriented so that the value the authors preferred counts as the
higher score (contact surface up; ddG, SAP, iPAE down). Orientation is fixed from the
papers' stated selection direction, before looking at any outcome.

CENSORING
---------
38 of 77 designs were scored but never tested, in large part because of poor cyclization
yield. They are NOT non-binders. Their exclusion is informative rather than random, so this
script separately tests whether the four metrics differ between synthesised and censored
designs: if they do, the 39 tested designs are a biased slice of the 77, and any AUROC
measured on them understates what the filter would do on the full designed pool. That test
is reported alongside the main result instead of being left as a caveat in prose.

TWO LABEL DEFINITIONS
---------------------
`binder_screen` (primary) is detectable binding at or below 100 uM in the SPR screen, the
criterion applied uniformly to all four targets: 14 of 39. `binder_characterised` is the
stricter subset carried forward to a measured affinity: 6 of 39. Both are reported. The
stricter label is too small to support a per-target estimate and is given pooled only.

    python phase2_retrospective.py --designs data/tier1_rfpeptides_per_design.csv \
        --outdir results --n-boot 10000 --n-perm 10000 --seed 20261013
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np

# Sign by which each metric was used as a filter: +1 means the authors preferred larger.
# Fixed from the papers' stated selection direction, independent of any outcome.
METRICS = {
    "contact_molecular_surface_A2": (+1, "Contact molecular surface (A^2)"),
    "rosetta_ddg_kcal_mol": (-1, "Rosetta ddG (kcal/mol)"),
    "sap": (-1, "SAP"),
    "ipae": (-1, "AF2 interface PAE"),
}


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Concordance of two samples; ties count a half. NaN when either side is empty."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    d = pos[:, None] - neg[None, :]
    return float(((d > 0).sum() + 0.5 * (d == 0).sum()) / d.size)


def pooled_auroc(groups) -> float:
    """AUROC over within-target pairs only: concordant pairs / comparable pairs.

    Equivalent to a sample-size-weighted mean of the per-target AUROCs, which is what keeps
    a target with many binders from standing in for predictive signal.
    """
    conc = tot = 0.0
    for pos, neg in groups:
        if len(pos) == 0 or len(neg) == 0:
            continue
        d = pos[:, None] - neg[None, :]
        conc += (d > 0).sum() + 0.5 * (d == 0).sum()
        tot += d.size
    return conc / tot if tot else float("nan")


def mannwhitney_u_p(a: np.ndarray, b: np.ndarray, rng, n_perm: int) -> tuple:
    """Two-sided permutation test on the rank-sum statistic (AUROC form)."""
    obs = auroc(a, b)
    if not np.isfinite(obs):
        return obs, float("nan")
    pool = np.concatenate([a, b])
    n = len(a)
    hits = 0
    for _ in range(n_perm):
        p = rng.permutation(pool)
        if abs(auroc(p[:n], p[n:]) - 0.5) >= abs(obs - 0.5) - 1e-12:
            hits += 1
    return obs, (hits + 1) / (n_perm + 1)


def topk_enrichment(scores: np.ndarray, labels: np.ndarray, k: int) -> float:
    """Binder rate in the top-k by score, divided by the overall binder rate."""
    base = labels.mean()
    if base == 0 or k > len(scores):
        return float("nan")
    order = np.argsort(-scores, kind="stable")
    return float(labels[order[:k]].mean() / base)


def load(path: str):
    rows = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            for m in METRICS:
                r[m] = float(r[m])
            r["synthesised"] = r["synthesised"] == "True"
            r["censored"] = r["censored"] == "True"
            for lab in ("binder_screen", "binder_characterised"):
                r[lab] = (r[lab] == "True") if r[lab] != "" else None
            rows.append(r)
    return rows


def analyse(rows, label: str, rng, n_boot: int, n_perm: int):
    tested = [r for r in rows if r["synthesised"]]
    by_target = defaultdict(list)
    for r in tested:
        by_target[r["target"]].append(r)

    out = {"label": label, "n_tested": len(tested),
           "n_binders": sum(r[label] for r in tested), "metrics": {}}

    for metric, (sign, pretty) in METRICS.items():
        groups, per_target = [], {}
        for tgt, rs in sorted(by_target.items()):
            s = np.array([sign * r[metric] for r in rs])
            y = np.array([r[label] for r in rs], dtype=bool)
            groups.append((s[y], s[~y]))
            per_target[tgt] = {
                "n": len(rs), "n_binders": int(y.sum()),
                "auroc": auroc(s[y], s[~y]),
            }

        obs = pooled_auroc(groups)

        # Bootstrap: resample designs within each target, preserving the stratification.
        boot = []
        for _ in range(n_boot):
            gb = []
            for tgt, rs in sorted(by_target.items()):
                idx = rng.integers(0, len(rs), len(rs))
                s = np.array([sign * rs[i][metric] for i in idx])
                y = np.array([rs[i][label] for i in idx], dtype=bool)
                gb.append((s[y], s[~y]))
            v = pooled_auroc(gb)
            if np.isfinite(v):
                boot.append(v)
        lo, hi = (np.percentile(boot, [2.5, 97.5]) if boot else (float("nan"),) * 2)

        # Null: permute labels within target.
        hits = 0
        for _ in range(n_perm):
            gp = []
            for tgt, rs in sorted(by_target.items()):
                s = np.array([sign * r[metric] for r in rs])
                y = rng.permutation(np.array([r[label] for r in rs], dtype=bool))
                gp.append((s[y], s[~y]))
            if abs(pooled_auroc(gp) - 0.5) >= abs(obs - 0.5) - 1e-12:
                hits += 1
        p = (hits + 1) / (n_perm + 1)

        all_s = np.array([sign * r[metric] for r in tested])
        all_y = np.array([r[label] for r in tested], dtype=bool)

        out["metrics"][metric] = {
            "pretty": pretty, "preferred_direction": "higher" if sign > 0 else "lower",
            "auroc_pooled_within_target": obs,
            "ci95": [float(lo), float(hi)], "p_permutation": p,
            "n_bootstrap_valid": len(boot),
            "per_target": per_target,
            "top10_enrichment_unstratified": topk_enrichment(all_s, all_y, 10),
            "top20_enrichment_unstratified": topk_enrichment(all_s, all_y, 20),
        }
    return out


def censoring_test(rows, rng, n_perm: int):
    """Do the metrics differ between designs that were synthesised and ones that were not?

    A difference means the tested 39 are a biased slice of the designed 77, which bounds how
    far any AUROC measured on them can be generalised to the full pool.
    """
    syn = [r for r in rows if r["synthesised"]]
    cen = [r for r in rows if r["censored"]]
    res = {"n_synthesised": len(syn), "n_censored": len(cen), "metrics": {}}
    for metric, (sign, pretty) in METRICS.items():
        a = np.array([sign * r[metric] for r in syn])
        b = np.array([sign * r[metric] for r in cen])
        u, p = mannwhitney_u_p(a, b, rng, n_perm)
        res["metrics"][metric] = {
            "pretty": pretty,
            "auroc_synthesised_vs_censored": u, "p_permutation": p,
            "mean_synthesised": float(np.mean([r[metric] for r in syn])),
            "mean_censored": float(np.mean([r[metric] for r in cen])),
        }
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--designs", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--n-perm", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20261013)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    rows = load(args.designs)
    rng = np.random.default_rng(args.seed)

    result = {
        "experiment": "C / manuscript 6.4 - retrospective value of published in-silico filters",
        "source": "Rettie et al., Nat Chem Biol 21:1948-1956 (2025), Supplementary Tables 1b-4b, 7-10",
        "n_designs_scored": len(rows),
        "seed": args.seed, "n_boot": args.n_boot, "n_perm": args.n_perm,
        "primary": analyse(rows, "binder_screen", rng, args.n_boot, args.n_perm),
        "secondary_characterised": analyse(rows, "binder_characterised", rng,
                                           args.n_boot, args.n_perm),
        "censoring": censoring_test(rows, rng, args.n_perm),
    }

    with open(os.path.join(args.outdir, "experiment_c_retrospective.json"), "w") as fh:
        json.dump(result, fh, indent=2)

    with open(os.path.join(args.outdir, "experiment_c_auroc.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["label", "metric", "preferred", "auroc", "ci95_lo", "ci95_hi",
                    "p_perm", "top10_enrich", "top20_enrich"])
        for key in ("primary", "secondary_characterised"):
            blk = result[key]
            for m, d in blk["metrics"].items():
                w.writerow([blk["label"], m, d["preferred_direction"],
                            f"{d['auroc_pooled_within_target']:.4f}",
                            f"{d['ci95'][0]:.4f}", f"{d['ci95'][1]:.4f}",
                            f"{d['p_permutation']:.4f}",
                            f"{d['top10_enrichment_unstratified']:.3f}",
                            f"{d['top20_enrichment_unstratified']:.3f}"])

    p = result["primary"]
    print(f"\nPRIMARY  binder = detectable at <=100 uM   "
          f"{p['n_binders']}/{p['n_tested']} synthesised designs")
    print(f"{'metric':<34}{'AUROC':>7}  {'95% CI':>16}  {'p':>7}  {'top10x':>7}")
    for m, d in p["metrics"].items():
        print(f"{d['pretty']:<34}{d['auroc_pooled_within_target']:>7.3f}  "
              f"[{d['ci95'][0]:.3f}, {d['ci95'][1]:.3f}]  {d['p_permutation']:>7.3f}  "
              f"{d['top10_enrichment_unstratified']:>7.2f}")
        for tgt, t in d["per_target"].items():
            print(f"    {tgt:<28}{t['auroc']:>7.3f}   (n={t['n']}, {t['n_binders']} binders)")

    s = result["secondary_characterised"]
    print(f"\nSECONDARY  binder = carried to a measured affinity   "
          f"{s['n_binders']}/{s['n_tested']}")
    for m, d in s["metrics"].items():
        print(f"{d['pretty']:<34}{d['auroc_pooled_within_target']:>7.3f}  "
              f"[{d['ci95'][0]:.3f}, {d['ci95'][1]:.3f}]  {d['p_permutation']:>7.3f}")

    c = result["censoring"]
    print(f"\nCENSORING  synthesised (n={c['n_synthesised']}) vs never tested "
          f"(n={c['n_censored']})")
    for m, d in c["metrics"].items():
        print(f"{d['pretty']:<34}{d['auroc_synthesised_vs_censored']:>7.3f}  "
              f"p={d['p_permutation']:.4f}   mean {d['mean_synthesised']:.2f} vs "
              f"{d['mean_censored']:.2f}")
    print(f"\nwrote {args.outdir}/experiment_c_retrospective.json and experiment_c_auroc.csv")


if __name__ == "__main__":
    main()
