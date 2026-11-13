#!/usr/bin/env python3
"""Closure share across all head-to-tail macrocyclic peptides found in a PDB snapshot.
Pre-registered in PREREGISTRATION_theory_12Nov2026.md (Amendment 7). Seed 20261208.
python3 phase9_pdb_population.py --pdb-dir <dir with <id>.ent> --cands cands.txt"""
import argparse, os, sys, re
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase1_closure import EnergyParams, Topology, decomposed_gradient, noise, parse_ca
from phase7_multi_restraint import measure
from phase7b_multi_restraint_v5 import f5
from phase1_chemistries import read_ssbonds, ca_resnums
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
DONE = {"9CDT", "9CDU", "9HGC", "9HGD", "9CDZ", "1SFI", "1JBL", "1NB1", "1K48"}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--pdb-dir", required=True); ap.add_argument("--cands", required=True); o = ap.parse_args()
    rng = np.random.default_rng(20261208); rows = []; seen_seq = {}
    cands = [l.rstrip("\n").split("\t") for l in open(o.cands)]
    import gzip
    seq = {}
    for h, s in zip(*[iter(gzip.open(os.path.join(os.path.dirname(os.path.abspath(o.cands)), "seqres.gz"), "rt"))] * 2):
        p = h[1:].split(); pid, ch = p[0].split("_"); seq[(pid, ch)] = s.strip()
    for pid, ch, L, name in sorted(cands, key=lambda r: r[0]):
        path = os.path.join(o.pdb_dir, f"{pid}.ent")
        if pid.upper() in DONE or not os.path.exists(path): continue
        try: x0 = parse_ca(path, ch)
        except Exception: continue
        if len(x0) != int(L): continue
        d = np.linalg.norm(x0[1:] - x0[:-1], axis=1); clo = float(np.linalg.norm(x0[-1] - x0[0]))
        if not (2.7 <= d.min() and d.max() <= 4.0 and clo <= 5.0): continue
        sq = seq.get((pid, ch), "")
        if sq in seen_seq: continue
        seen_seq[sq] = pid
        n = len(x0); rg = float(np.sqrt(((x0 - x0.mean(0)) ** 2).sum(1).mean()))
        # (A) all terms
        p_all = EnergyParams(); top = Topology(n); fs = []
        for t in (0.3, 0.45, 0.6, 0.75, 0.9):
            for _ in range(64):
                xt = noise(x0, t, rng, scale=rg); terms = decomposed_gradient(xt, top, None, p_all)
                nm = {k: float(np.linalg.norm(g)) for k, (_, g) in terms.items()}; fs.append(nm["closure"] / sum(nm.values()))
        # (B) bond only vs T5
        p_b = EnergyParams(k_angle=0.0, k_steric=0.0); fm = measure(x0, [(n - 1, 0)], 1.0, 400, rng, p_b); fp = f5(n, rg, [clo])
        try: nss = len(read_ssbonds(path, ch, ca_resnums(path, ch)))
        except Exception: nss = -1
        rows.append(dict(pdb=pid.upper(), chain=ch, n=n, name=name, closure_A=round(clo, 2), n_disulfide=nss, f_allterms=float(np.mean(fs)), f_bondonly=fm, f_T5=fp, err=fm / fp - 1))
    df = pd.DataFrame(rows); df.round(4).to_csv(os.path.join(OUT, "phase9_pdb_population.csv"), index=False)
    pd.set_option("display.width", 220); print(df.round(3).to_string(index=False))
    from scipy.stats import spearmanr
    print("N structures:", len(df))
    print("RX1 all-terms share < 0.2: %d of %d (%.0f%%)" % ((df.f_allterms < .2).sum(), len(df), 100 * (df.f_allterms < .2).mean()))
    print("RX2 bond-only within 10%% of T5: %d of %d (%.0f%%)" % ((df.err.abs() <= .10).sum(), len(df), 100 * (df.err.abs() <= .10).mean()))
    print("RX3 Spearman(T5, measured) = %.3f" % spearmanr(df.f_T5, df.f_bondonly)[0])
    print("max all-terms share:", df.f_allterms.max().round(3), "; median:", df.f_allterms.median().round(3))

if __name__ == "__main__":
    main()
