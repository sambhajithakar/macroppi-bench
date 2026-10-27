"""Power of the retrospective AUROC test (pre-specified 27 Oct 2026, before first run).

QUESTION. How large must the true AUROC of a score be for the RFpeptides stratum
(25 non-binders, 14 binders) to detect it, and how many designs would a campaign need?

SPECIFICATION (fixed before running, not tuned afterwards):
  * Scores are simulated as equal-variance normals with mean shift d between binders and
    non-binders, where true AUROC = Phi(d / sqrt(2)).
  * Test: one-sided permutation test of AUROC > 0.5, 2000 permutations, alpha = 0.05.
  * 2000 simulated datasets per cell; fixed seed 20261027.
  * Grid A: true AUROC in {0.60, 0.65, 0.70, 0.75, 0.80, 0.90}, n = (25 nb, 14 b).
  * Grid B: true AUROC 0.70, total designs N in {39, 60, 100, 150, 200, 300}
    keeping the observed binder fraction 14/39.
  * Reported: power = fraction of datasets with p < 0.05. No other outputs.
Run: python scripts/phase3_power.py
"""
import numpy as np, csv, pathlib
from scipy.stats import norm, rankdata

rng = np.random.default_rng(20261027)
OUT = pathlib.Path(__file__).resolve().parent.parent / "results" / "phase3_power.csv"

def auc(x, y):  # y=1 binder
    r = rankdata(x)
    n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

def power(true_auc, n0, n1, sims=2000, perms=2000):
    d = np.sqrt(2) * norm.ppf(true_auc)
    y = np.r_[np.zeros(n0), np.ones(n1)].astype(int)
    hit = 0
    for _ in range(sims):
        x = rng.normal(size=n0 + n1) + d * y
        a = auc(x, y)
        # vectorised permutation: ranks fixed, permute labels
        r = rankdata(x)
        idx = np.argsort(rng.random((perms, n0 + n1)), axis=1)[:, :n1]
        s = r[idx].sum(axis=1)
        nulls = (s - n1 * (n1 + 1) / 2) / (n1 * n0)
        hit += (nulls >= a).mean() < 0.05
    return hit / sims

rows = []
for a in (0.60, 0.65, 0.70, 0.75, 0.80, 0.90):
    rows.append(("A", a, 25, 14, power(a, 25, 14)))
for N in (39, 60, 100, 150, 200, 300):
    n1 = round(N * 14 / 39); rows.append(("B", 0.70, N - n1, n1, power(0.70, N - n1, n1)))
OUT.parent.mkdir(exist_ok=True)
with open(OUT, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["grid", "true_auroc", "n_nonbinders", "n_binders", "power"]); w.writerows(rows)
for r in rows: print(r)
