#!/usr/bin/env python3
"""Theory test for the closure share (pre-registered in PREREGISTRATION_theory_12Nov2026.md). SIMULATION. Seed 20261130.
python3 phase4_theory.py [--draws 400]"""
import argparse, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase1_closure import Topology, EnergyParams, decomposed_gradient, noise, synthetic_ring

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
SEED = 20261130

def pred(n, lam=1.0):
    a = lam * np.sqrt(2); return a / (a + np.sqrt(3 * n - 4))

def enrich_pred(n): return np.sqrt(2) * (n - 1) / np.sqrt(3 * n - 4)

def measure(pep, t, draws, rng, lam=1.0, p=None, target=None):
    top = Topology(len(pep)); rg = float(np.sqrt(((pep - pep.mean(0)) ** 2).sum(1).mean()))
    f, e = [], []
    for _ in range(draws):
        xt = noise(pep, t, rng, scale=rg)
        terms = decomposed_gradient(xt, top, target, p, lam_closure=lam)
        nm = {k: float(np.linalg.norm(g)) for k, (_, g) in terms.items()}
        tot = sum(nm.values()); f.append(nm["closure"] / tot)
        e.append(nm["closure"] / (nm["bond"] / (len(pep) - 1)) / lam)  # enrichment of the unscaled bond
    return float(np.mean(f)), float(np.mean(e))

def lam_star(pep, t, draws, rng, p):
    lo, hi = 0.1, 100.0
    for _ in range(14):
        mid = np.sqrt(lo * hi)
        f, _ = measure(pep, t, draws, rng, lam=mid, p=p)
        lo, hi = (mid, hi) if f < 0.5 else (lo, mid)
    return float(np.sqrt(lo * hi))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--draws", type=int, default=400); o = ap.parse_args()
    rng = np.random.default_rng(SEED); p = EnergyParams(k_angle=0.0, k_steric=0.0)
    rows = []
    for n in (8, 12, 16, 24, 32, 48):
        pep = synthetic_ring(n, rng)
        r = dict(n=n)
        for t in (1.0, 0.5):
            f, e = measure(pep, t, o.draws, rng, p=p)
            r[f"f_meas_t{str(t).replace(".","p")}"] = f; r[f"enr_meas_t{str(t).replace(".","p")}"] = e
        r["f_pred"] = pred(n); r["enr_pred"] = enrich_pred(n)
        r["lam_star_meas"] = lam_star(pep, 1.0, o.draws, rng, p); r["lam_star_pred"] = float(np.sqrt((3 * n - 4) / 2))
        rows.append(r)
    df = pd.DataFrame(rows)
    df["err_f"] = df.f_meas_t1p0 / df.f_pred - 1; df["err_enr"] = df.enr_meas_t1p0 / df.enr_pred - 1
    df["err_lam"] = df.lam_star_meas / df.lam_star_pred - 1
    os.makedirs(OUT, exist_ok=True); df.round(4).to_csv(os.path.join(OUT, "phase4_theory.csv"), index=False)
    pd.set_option("display.width", 200); print(df.round(3).to_string(index=False))
    print("RT1 (f within 10%):", int((df.err_f.abs() <= 0.10).sum()), "of 6")
    print("RT2 (lambda* within 15%):", int((df.err_lam.abs() <= 0.15).sum()), "of 6")
    print("RT3 (enrichment within 10%):", int((df.err_enr.abs() <= 0.10).sum()), "of 6")

if __name__ == "__main__":
    main()
