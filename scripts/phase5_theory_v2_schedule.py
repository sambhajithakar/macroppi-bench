#!/usr/bin/env python3
"""Theory v2 + open-chain share + share-capped schedule. Pre-registered in PREREGISTRATION_theory_12Nov2026.md (Amendment 2).
SIMULATION. python3 phase5_theory_v2_schedule.py [--procs 4]"""
import argparse, os, sys
import numpy as np, pandas as pd
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase1_closure import Topology, EnergyParams, decomposed_gradient, noise, synthetic_ring, lam_closure_aware
from phase4_theory import measure

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
B0 = 3.8

def rho_of(sigma, m=4_000_000, seed=1):
    rng = np.random.default_rng(seed)
    e = rng.normal(size=(m, 3, 3)) * sigma
    da, db = e[:, 1] - e[:, 0], e[:, 2] - e[:, 1]
    ca = 1 - B0 / np.linalg.norm(da, axis=1); cb = 1 - B0 / np.linalg.norm(db, axis=1)
    A = np.mean(ca ** 2 * (da ** 2).sum(1)); B = np.mean(ca * cb * (da * db).sum(1))
    return B / A

def pred2(n, rho, lam=1.0):
    Q = 2 * (n - 2) * (1 - rho) + 2
    f = lam * np.sqrt(2) / (lam * np.sqrt(2) + np.sqrt(Q))
    return f, np.sqrt(2) * (n - 1) / np.sqrt(Q), np.sqrt((n - 2) * (1 - rho) + 1)

def lam_star(pep, draws, rng, p):
    lo, hi = 0.1, 100.0
    for _ in range(14):
        mid = np.sqrt(lo * hi); f, _ = measure(pep, 1.0, draws, rng, lam=mid, p=p)
        lo, hi = (mid, hi) if f < 0.5 else (lo, mid)
    return float(np.sqrt(lo * hi))

def theory_v2(draws, rng):
    p = EnergyParams(k_angle=0.0, k_steric=0.0); rows = []
    for n in (10, 14, 20, 28, 40, 64):
        pep = synthetic_ring(n, rng); rg = float(np.sqrt(((pep - pep.mean(0)) ** 2).sum(1).mean()))
        rho = rho_of(rg); f_p, e_p, l_p = pred2(n, rho)
        f_m, e_m = measure(pep, 1.0, draws, rng, p=p); l_m = lam_star(pep, draws, rng, p)
        rows.append(dict(n=n, rg=rg, rho=rho, f_pred=f_p, f_meas=f_m, enr_pred=e_p, enr_meas=e_m, lam_pred=l_p, lam_meas=l_m,
                         err_f=f_m / f_p - 1, err_enr=e_m / e_p - 1, err_lam=l_m / l_p - 1))
    return pd.DataFrame(rows)

def open_chain(n, rng):
    d = rng.normal(size=(n - 1, 3)); d = B0 * d / np.linalg.norm(d, axis=1, keepdims=True)
    return np.vstack([np.zeros(3), np.cumsum(d, axis=0)])

def share_open(args):
    n, seed = args; rng = np.random.default_rng(seed); p = EnergyParams(); top = Topology(n)
    x = open_chain(n, rng); terms = decomposed_gradient(x, top, None, p)
    nm = {k: float(np.linalg.norm(g)) for k, (_, g) in terms.items()}
    return n, nm["closure"] / sum(nm.values()), float(np.linalg.norm(x[-1] - x[0]))

def run_anneal(args):
    n, sched, seed = args; rng = np.random.default_rng(seed); p = EnergyParams(); top = Topology(n)
    x = open_chain(n, rng); n_steps, lr, temp0 = 2000, 5e-4, 0.5; q = 0.2
    for step in range(n_steps):
        temp = temp0 * (1 - step / n_steps)
        d_cl = float(np.linalg.norm(x[-1] - x[0]))
        if sched == "constant": lam = 1.0
        elif sched == "sigmoid": lam = lam_closure_aware(d_cl)
        else:
            t1 = decomposed_gradient(x, top, None, p, lam_closure=1.0)
            nm = {k: float(np.linalg.norm(g)) for k, (_, g) in t1.items()}
            other = sum(v for k, v in nm.items() if k != "closure")
            lam = min(1.0, q / (1 - q) * other / max(nm["closure"], 1e-12))
        terms = decomposed_gradient(x, top, None, p, lam_closure=lam)
        g = sum(gr for _, gr in terms.values()); gn = np.linalg.norm(g)
        if gn > 1e3: g = g * (1e3 / gn)
        x = x - lr * g + rng.normal(scale=temp * np.sqrt(2 * lr), size=x.shape)
    bonds = np.linalg.norm(x[1:] - x[:-1], axis=1)
    return dict(n=n, sched=sched, closure=float(np.linalg.norm(x[-1] - x[0])), bond_rms=float(np.sqrt(np.mean((bonds - B0) ** 2))))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--procs", type=int, default=4); o = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    df = theory_v2(400, np.random.default_rng(20261201))
    df.round(4).to_csv(os.path.join(OUT, "phase5_theory_v2.csv"), index=False)
    pd.set_option("display.width", 220); print(df.round(3).to_string(index=False))
    print("RT4 f within 5%:", int((df.err_f.abs() <= .05).sum()), "of 6;  RT5 lambda* within 10%:", int((df.err_lam.abs() <= .10).sum()),
          "of 6;  RT6 enrichment within 5%:", int((df.err_enr.abs() <= .05).sum()), "of 6", flush=True)
    ss = np.random.SeedSequence(20261202).spawn(600)
    with Pool(o.procs) as pool:
        r = pool.map(share_open, [(n, int(ss[i * 200 + j].generate_state(1)[0])) for i, n in enumerate((12, 16, 24)) for j in range(200)])
    so = pd.DataFrame(r, columns=["n", "f_cl", "d_cl"]); so_s = so.groupby("n").mean().reset_index()
    so_s.round(4).to_csv(os.path.join(OUT, "phase5_open_chain_share.csv"), index=False); print(so_s.round(3).to_string(index=False))
    print("RS1 sizes with mean f_cl > 0.5:", int((so_s.f_cl > .5).sum()), "of 3", flush=True)
    ss = np.random.SeedSequence(20261203).spawn(900)
    jobs = [(n, s, int(ss[(i * 3 + k) * 100 + j].generate_state(1)[0])) for i, n in enumerate((12, 16, 24))
            for k, s in enumerate(("constant", "sigmoid", "capped")) for j in range(100)]
    with Pool(o.procs) as pool:
        res = pool.map(run_anneal, jobs, chunksize=5)
    a = pd.DataFrame(res); a["closed"] = a.closure < 5.0
    sm = a.groupby(["n", "sched"]).agg(closed=("closed", "mean"), closure=("closure", "mean"), bond_rms=("bond_rms", "mean")).reset_index()
    sm.round(4).to_csv(os.path.join(OUT, "phase5_schedules.csv"), index=False); print(sm.round(3).to_string(index=False))
    for n in (12, 16, 24):
        g = sm[sm.n == n].set_index("sched")
        print(f"n={n} RS2 capped closed>=0.9: {g.loc['capped','closed']>=.9}; RS3 capped bond_rms<constant: {g.loc['capped','bond_rms']<g.loc['constant','bond_rms']}; RS4 capped>=sigmoid-0.05: {g.loc['capped','closed']>=g.loc['sigmoid','closed']-.05}")

if __name__ == "__main__":
    main()
