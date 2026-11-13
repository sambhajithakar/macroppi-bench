#!/usr/bin/env python3
"""Identification bounds and decision-level utility for the RFpeptides filter evaluation.
Pre-registered in PREREGISTRATION_theory_12Nov2026.md (Amendment 5). Seed 20261206.
python3 phase8_bounds.py --designs ../data/tier1_rfpeptides_per_design.csv"""
import argparse, itertools, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase2_retrospective import METRICS
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
PIS = (0.0, 0.1, 0.2, 0.36, 0.5)

def conc(pos, neg):
    if len(pos) == 0 or len(neg) == 0: return 0.0, 0
    d = pos[:, None] - neg[None, :]; return float((d > 0).sum() + 0.5 * (d == 0).sum()), d.size

def pooled(blocks):
    c = t = 0.0
    for pos, neg in blocks: a, b = conc(pos, neg); c += a; t += b
    return c / t if t else float("nan")

def target_blocks(lab_s, lab_y, un_s, un_y):
    s = np.concatenate([lab_s, un_s]); y = np.concatenate([lab_y, un_y]).astype(bool); return s[y], s[~y]

def extreme(lab_s, lab_y, un_s, k, which):
    order = np.argsort(-un_s, kind="stable"); y = np.zeros(len(un_s), bool)
    idx = order[:k] if which == "max" else order[len(un_s) - k:]; y[idx] = True
    return target_blocks(lab_s, lab_y, un_s, y)

def solver_check(rng, n=200):
    ok = 0
    for _ in range(n):
        tg = []
        for _t in range(2):
            ls = rng.normal(size=4); ly = np.array([1, 1, 0, 0]); rng.shuffle(ly)
            us = rng.normal(size=5); k = int(rng.integers(1, 3)); tg.append((ls, ly, us, k))
        for which, fn in (("max", max), ("min", min)):
            fast = pooled([extreme(ls, ly, us, k, which) for ls, ly, us, k in tg])
            best = []
            for c1 in itertools.combinations(range(5), tg[0][3]):
                for c2 in itertools.combinations(range(5), tg[1][3]):
                    bl = []
                    for (ls, ly, us, k), c in zip(tg, (c1, c2)):
                        y = np.zeros(5, bool); y[list(c)] = True; bl.append(target_blocks(ls, ly, us, y))
                    best.append(pooled(bl))
            ok += abs(fn(best) - fast) < 1e-12
    return ok, 2 * n

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--designs", required=True); o = ap.parse_args()
    rng = np.random.default_rng(20261206)
    ok, tot = solver_check(rng); print(f"RB1 solver check: closed form equals exhaustive in {ok} of {tot} (200 problems, max and min)")
    d = pd.read_csv(o.designs); d["lab"] = d.synthesised.astype(str) == "True"
    d["y"] = d.binder_screen.map({True: 1, False: 0, "True": 1, "False": 0})
    rows = []; util = []
    for m, (sg, pretty) in METRICS.items():
        d["s"] = sg * d[m]
        T = {t: g for t, g in d.groupby("target")}
        obs = pooled([(g[g.lab & (g.y == 1)].s.values, g[g.lab & (g.y == 0)].s.values) for g in T.values()])
        for pi in PIS:
            mx, mn, rnd = [], [], []
            for t, g in T.items():
                L = g[g.lab]; U = g[~g.lab]; k = int(round(pi * len(U)))
                mx.append(extreme(L.s.values, L.y.values, U.s.values, k, "max")); mn.append(extreme(L.s.values, L.y.values, U.s.values, k, "min"))
            draws = []
            for _ in range(2000):
                bl = []
                for t, g in T.items():
                    L = g[g.lab]; U = g[~g.lab]; k = int(round(pi * len(U))); y = np.zeros(len(U), bool); y[rng.choice(len(U), k, replace=False)] = True
                    bl.append(target_blocks(L.s.values, L.y.values, U.s.values, y))
                draws.append(pooled(bl))
            rows.append(dict(filter=pretty, pi=pi, observed_tested_only=obs, auroc_min=pooled(mn), auroc_max=pooled(mx), auroc_random_mean=float(np.mean(draws))))
        # decision-level utility on the tested designs (stratified bootstrap)
        for q in (0.25, 0.5):
            def enr(groups):
                hit = ns = 0.0; exp = 0.0
                for L in groups:
                    n = len(L); ksel = int(np.ceil(q * n)); o_ = np.argsort(-L.s.values, kind="stable")[:ksel]
                    hit += L.y.values[o_].sum(); ns += ksel; exp += ksel * L.y.mean()
                return hit / exp if exp > 0 else np.nan, hit, exp
            groups = [g[g.lab].reset_index(drop=True) for g in T.values()]
            e0, h0, x0 = enr(groups); bs = []
            for _ in range(10000):
                bs.append(enr([g.iloc[rng.integers(0, len(g), len(g))].reset_index(drop=True) for g in groups])[0])
            lo, hi = np.nanpercentile(bs, [2.5, 97.5])
            util.append(dict(filter=pretty, top_fraction=q, binders_in_selection=h0, expected_by_chance=round(x0, 2), enrichment=e0, ci_lo=lo, ci_hi=hi))
    R = pd.DataFrame(rows); U = pd.DataFrame(util)
    R.round(3).to_csv(os.path.join(OUT, "phase8_bounds.csv"), index=False); U.round(3).to_csv(os.path.join(OUT, "phase8_utility.csv"), index=False)
    pd.set_option("display.width", 220); print(R.round(3).to_string(index=False)); print(U.round(2).to_string(index=False))
    for f, g in R.groupby("filter"):
        cov = g[(g.auroc_min <= .5) & (g.auroc_max >= .5)].pi.tolist(); b70 = g[g.auroc_max >= .70].pi.tolist()
        print(f"RB2 {f}: pi values where interval contains 0.5: {cov}; smallest pi with best case >= 0.70: {min(b70) if b70 else 'none in grid'}")

if __name__ == "__main__":
    main()
