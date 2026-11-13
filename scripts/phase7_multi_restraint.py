#!/usr/bin/env python3
"""Theory v4 (m closure restraints). Pre-registered in PREREGISTRATION_theory_12Nov2026.md (Amendment 4). SIMULATION. Seed 20261205."""
import os, sys
import numpy as np, pandas as pd
from scipy.optimize import brentq
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase1_closure import EnergyParams, synthetic_ring, noise, _bond_energy_grad
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
rng_i = np.random.default_rng(11)
CH = {m: np.linalg.norm(rng_i.normal(size=(1_000_000, 3 * m)), axis=1) for m in (1, 2, 3, 4, 40)}

def f4(n, m, lam=1.0, ch=None):
    chm = CH[m] if ch is None else ch
    a = np.sqrt(2) * lam * chm; return float(np.mean(a / (a + np.sqrt(3 * (3 * n - 4)))))

def pairs_for(n, m, rng):
    used = {0, n - 1}; pairs = [(n - 1, 0)]
    while len(pairs) < m:
        i, j = sorted(rng.choice(n, size=2, replace=False))
        if i in used or j in used or abs(i - j) < 3 or (i == 0 and j == n - 1): continue
        used |= {i, j}; pairs.append((i, j))
    return pairs

def measure(pep, pairs, lam, draws, rng, p):
    n = len(pep); rg = float(np.sqrt(((pep - pep.mean(0)) ** 2).sum(1).mean()))
    sb = np.array([[i, i + 1] for i in range(n - 1)]); b0 = [float(np.linalg.norm(pep[i] - pep[j])) for i, j in pairs]
    f = []
    for _ in range(draws):
        xt = noise(pep, 1.0, rng, scale=rg)
        _, gb = _bond_energy_grad(xt, sb[:, 0], sb[:, 1], p.k_bond, p.b0)
        gc = np.zeros_like(xt)
        for (i, j), b in zip(pairs, b0):
            _, g = _bond_energy_grad(xt, np.array([i]), np.array([j]), p.k_bond, b); gc += lam * g
        a, c = np.linalg.norm(gc), np.linalg.norm(gb); f.append(a / (a + c))
    return float(np.mean(f))

def main():
    rng = np.random.default_rng(20261205); p = EnergyParams(k_angle=0.0, k_steric=0.0); rows = []
    for n in (18, 30):
        pep = synthetic_ring(n, rng)
        for m in (1, 2, 3, 4):
            pairs = pairs_for(n, m, rng)
            f_m = measure(pep, pairs, 1.0, 400, rng, p); f_p = f4(n, m)
            lo, hi = 0.1, 100.0
            for _ in range(14):
                mid = np.sqrt(lo * hi); fm = measure(pep, pairs, mid, 400, rng, p); lo, hi = (mid, hi) if fm < 0.5 else (lo, mid)
            l_m = float(np.sqrt(lo * hi)); l_p = brentq(lambda l: f4(n, m, l) - 0.5, 0.1, 500)
            rows.append(dict(n=n, m=m, f_pred=f_p, f_meas=f_m, err_f=f_m / f_p - 1, lam_pred=l_p, lam_meas=l_m, err_lam=l_m / l_p - 1))
    df = pd.DataFrame(rows); df.round(4).to_csv(os.path.join(OUT, "phase7_multi_restraint.csv"), index=False)
    pd.set_option("display.width", 200); print(df.round(3).to_string(index=False))
    print("RT10 f within 5%:", int((df.err_f.abs() <= .05).sum()), "of 8;  RT11 lambda* within 10%:", int((df.err_lam.abs() <= .10).sum()), "of 8")
    print("RT12 f increasing in m:", {n: bool(np.all(np.diff(df[df.n == n].f_meas.values) > 0)) for n in (18, 30)})
    # reported: m at which prediction crosses 0.5 for n=14, 29, using chi with 3m dof, m large (normal approx via sampling)
    for n in (14, 29):
        for m in range(1, 200):
            ch = np.linalg.norm(rng_i.normal(size=(20000, 3 * m)), axis=1)
            if f4(n, m, 1.0, ch) >= 0.5: print(f"predicted m* (f=0.5) for n={n}: {m}"); break
    # reported: descriptive T4 predictions for real chemistries
    b1 = pd.read_csv(os.path.join(OUT, "experiment_b1_chemistries.csv")); b1 = b1[(b1.t >= .3) & (b1.t <= .9)]
    g = b1.groupby(["structure", "n_residues", "topology", "n_closure_restraints"]).f_closure_mean.mean().reset_index()
    g["f_T4_bond_only"] = [f4(r.n_residues, int(r.n_closure_restraints)) for r in g.itertuples()]
    g.round(3).to_csv(os.path.join(OUT, "phase7_real_descriptive.csv"), index=False); print(g.round(3).to_string(index=False))

if __name__ == "__main__":
    main()
