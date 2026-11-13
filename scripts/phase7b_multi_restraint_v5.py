#!/usr/bin/env python3
"""Theory v5 (native-length correction). Pre-registered in PREREGISTRATION_theory_12Nov2026.md (Amendment 6). SIMULATION. Seed 20261207."""
import os, sys
import numpy as np, pandas as pd
from scipy.optimize import brentq
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase1_closure import EnergyParams, synthetic_ring
from phase7_multi_restraint import pairs_for, measure, f4
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
rg_ = np.random.default_rng(13); CHI = np.linalg.norm(rg_.normal(size=(200000, 4, 3)), axis=2)  # draws x up to 4 restraints

def f5(n, rg, b, lam=1.0):
    s = np.sqrt(2) * rg; m = len(b)
    a = np.sqrt(2) * lam * np.sqrt(((s * CHI[:, :m] - np.array(b)[None, :]) ** 2).sum(1))
    Bn = np.sqrt((3 * n - 4) * np.mean((s * CHI[:, 0] - 3.8) ** 2))
    return float(np.mean(a / (a + Bn)))

def main():
    rng = np.random.default_rng(20261207); p = EnergyParams(k_angle=0.0, k_steric=0.0); rows = []
    for n in (22, 40):
        pep = synthetic_ring(n, rng); rg = float(np.sqrt(((pep - pep.mean(0)) ** 2).sum(1).mean()))
        for m in (1, 2, 3, 4):
            pairs = pairs_for(n, m, rng); b = [float(np.linalg.norm(pep[i] - pep[j])) for i, j in pairs]
            f_m = measure(pep, pairs, 1.0, 400, rng, p); f_5 = f5(n, rg, b); f_4 = f4(n, m)
            lo, hi = 0.1, 100.0
            for _ in range(14):
                mid = np.sqrt(lo * hi); fm = measure(pep, pairs, mid, 400, rng, p); lo, hi = (mid, hi) if fm < 0.5 else (lo, mid)
            l_m = float(np.sqrt(lo * hi)); l_5 = brentq(lambda l: f5(n, rg, b, l) - 0.5, 0.05, 500)
            rows.append(dict(n=n, m=m, f_meas=f_m, f_T5=f_5, f_T4=f_4, err_T5=f_5 / f_m - 1 if False else f_m / f_5 - 1, err_T4=f_m / f_4 - 1, lam_meas=l_m, lam_T5=l_5, err_lam=l_m / l_5 - 1))
    df = pd.DataFrame(rows); df.round(4).to_csv(os.path.join(OUT, "phase7b_multi_restraint_v5.csv"), index=False)
    pd.set_option("display.width", 200); print(df.round(3).to_string(index=False))
    print("RT13 f within 5%:", int((df.err_T5.abs() <= .05).sum()), "of 8;  RT14 lambda* within 10%:", int((df.err_lam.abs() <= .10).sum()), "of 8;  RT15 MAE T5 %.3f vs T4 %.3f" % (df.err_T5.abs().mean(), df.err_T4.abs().mean()))

if __name__ == "__main__":
    main()
