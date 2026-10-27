#!/usr/bin/env python3
"""
Experiment A sensitivity (manuscript Section 4.1, Supporting Information Table S2):
closure share of the guidance gradient under force-field rescaling and term removal.

COMMAND LINE AS RUN
-------------------
    python3 phase1_sensitivity.py --out results --samples 64 --seed 0 --local-dir <dir of .ent files>

Each condition changes one parameter of the baseline EnergyParams (k_bond 100, k_steric 10,
k_angle 20, k_iface 2 in the units of phase1_closure.py) and reruns the Experiment A
diagnostic on the same five structures. The reported value is the mean closure share over
t in [0.3, 0.9], the pre-registered window. Nothing else is varied.

NOTE ON PROVENANCE. The deposited results/experiment_a_sensitivity.csv (Table S2) was
produced by an earlier version of this analysis whose script was not deposited. This script
regenerates the table from deposited code; its output is written to
experiment_a_sensitivity_rerun.csv and the two are compared in the Supporting Information.
"""
import argparse
import csv
import dataclasses
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import phase1_closure as P  # noqa: E402

CONDITIONS = [
    ("baseline", {}),
    ("k_bond x0.1", {"k_bond": 10.0}),
    ("k_bond x10", {"k_bond": 1000.0}),
    ("k_bond x100", {"k_bond": 10000.0}),
    ("k_steric=0", {"k_steric": 0.0}),
    ("k_steric=100", {"k_steric": 100.0}),
    ("k_angle=0", {"k_angle": 0.0}),
    ("k_angle=200", {"k_angle": 200.0}),
    ("no interface term", {"k_iface": 0.0}),
    ("k_iface=50", {"k_iface": 50.0}),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="results")
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache", default="pdb_cache")
    ap.add_argument("--local-dir", default=None)
    a = ap.parse_args()
    structures = P.load_structures(False, a.cache, seed=a.seed, local_dir=a.local_dir)
    base = P.EnergyParams()
    rows = []
    for name, change in CONDITIONS:
        p = dataclasses.replace(base, **change)
        res = P.verdict_from_a(P.experiment_a(structures, p, n_samples=a.samples, seed=a.seed))
        rows.append({"condition": name,
                     "f_closure_mean": round(res["mean_f_closure_mid_t"], 4),
                     "per_bond_enrichment": round(res["mean_per_bond_enrichment_mid_t"], 3)})
        print(rows[-1])
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "experiment_a_sensitivity_rerun.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["condition", "f_closure_mean", "per_bond_enrichment"])
        w.writeheader()
        w.writerows(rows)
    print("wrote", path)


if __name__ == "__main__":
    main()
