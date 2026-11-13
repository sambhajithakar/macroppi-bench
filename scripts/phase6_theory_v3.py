#!/usr/bin/env python3
"""Theory v3 (Jensen correction). Pre-registered in PREREGISTRATION_theory_12Nov2026.md (Amendment 3). SIMULATION. Seed 20261204."""
import os, sys
import numpy as np, pandas as pd
from scipy.optimize import brentq
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase1_closure import EnergyParams, synthetic_ring
from phase4_theory import measure
from phase5_theory_v2_schedule import lam_star
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
rng_i = np.random.default_rng(7); CHI = np.linalg.norm(rng_i.normal(size=(4_000_000, 3)), axis=1)

def f3(n, lam):
    a = np.sqrt(2) * lam * CHI; return float(np.mean(a / (a + np.sqrt(3 * (3 * n - 4)))))

def main():
    rng = np.random.default_rng(20261204); p = EnergyParams(k_angle=0.0, k_steric=0.0); rows = []
    for n in (9, 13, 18, 26, 36, 56):
        pep = synthetic_ring(n, rng)
        f_m, e_m = measure(pep, 1.0, 400, rng, p=p); l_m = lam_star(pep, 400, rng, p)
        f_p = f3(n, 1.0); e_p = np.sqrt(2) * (n - 1) * CHI.mean() / np.sqrt(3 * (3 * n - 4)); l_p = brentq(lambda l: f3(n, l) - 0.5, 0.1, 100)
        rows.append(dict(n=n, f_pred=f_p, f_meas=f_m, enr_pred=e_p, enr_meas=e_m, lam_pred=l_p, lam_meas=l_m, err_f=f_m / f_p - 1, err_enr=e_m / e_p - 1, err_lam=l_m / l_p - 1))
    df = pd.DataFrame(rows); df.round(4).to_csv(os.path.join(OUT, "phase6_theory_v3.csv"), index=False)
    pd.set_option("display.width", 200); print(df.round(3).to_string(index=False))
    print("RT7 f within 5%:", int((df.err_f.abs() <= .05).sum()), "of 6;  RT8 lambda* within 10%:", int((df.err_lam.abs() <= .10).sum()), "of 6;  RT9 enrichment within 5%:", int((df.err_enr.abs() <= .05).sum()), "of 6")
if __name__ == "__main__":
    main()
