#!/usr/bin/env python3
"""
Phase 1b: does the closure term dominate for DISULFIDE and BICYCLE cyclization?

WHY THIS EXISTS
---------------
phase1_closure.py measured head-to-tail macrocycles and found no closure domination
(f_cl ~ 0.155, flat in diffusion time). But CP-Composer's published 0% energy-guided
baseline was on **disulfide and bicycle** chemistries, not head-to-tail. A refutation that
only covers head-to-tail leaves the actual failure case untested, and the obvious objection
is that disulfide and bicycle closures are geometrically harder, so perhaps domination
appears there.

This script tests that objection on real deposited structures.

TOPOLOGIES
----------
Given one cyclic peptide backbone, the same coordinates are evaluated under three
topologies, differing only in which bonds are labelled "closure":

  head_to_tail  backbone bond n-1 -> 0 is the closure; disulfides ignored.
  disulfide     the backbone is treated as LINEAR (no n-1 -> 0 bond) and the Cys-Cys
                pair is the sole ring-closing restraint. This is the topology a
                disulfide-cyclized design actually has.
  bicycle       both the head-to-tail backbone bond AND every disulfide bridge are
                closure restraints, which is the bicyclic case.

Each closure restraint uses its own equilibrium length taken from the native structure,
so a Cys-Cys CA-CA restraint is not penalised for not being 3.8 A. Force constants are
identical to the sequential backbone bonds in every case. Nothing is tuned to make closure
dominate or not dominate.

STRUCTURES
----------
Real head-to-tail cyclic peptides with experimentally determined disulfides:
  1SFI chain I  SFTI-1 bound to trypsin, 14 res, 1 disulfide (bicyclic)
  1JBL chain A  SFTI-1 free, 14 res, 1 disulfide
  1NB1 chain A  kalata B1, 29 res, 3 disulfides (cyclic cystine knot)
  1K48 chain A  kalata B1, 29 res, 3 disulfides
Disulfide pairs are read from each file's SSBOND records, not assumed.

READING THE RESULT
------------------
The pre-registered threshold from phase1_closure.py applies unchanged: mean f_cl over
t in [0.3, 0.9] above 0.5 is domination, 0.2-0.5 partial, below 0.2 refuted. If disulfide
or bicycle topologies cross into domination where head-to-tail did not, the original
hypothesis survives for exactly the chemistries CP-Composer reported failing on, and that
is the finding. If they do not, the refutation extends to the failure case itself.

    python phase1_chemistries.py --pdb-dir <dir of .ent files> --out results/
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase1_closure import (                      # noqa: E402
    EnergyParams, _bond_energy_grad, _angle_energy_grad, _steric_energy_grad,
    _interface_energy_grad, noise, parse_ca,
)

# pdb_id -> (peptide chain, target chain or None, label)
STRUCTURES = {
    "1SFI": ("I", "A", "SFTI-1 + trypsin"),
    "1JBL": ("A", None, "SFTI-1 free"),
    "1NB1": ("A", None, "kalata B1"),
    "1K48": ("A", None, "kalata B1 (refined)"),
}

# NMR cyclic peptides contain cis peptide bonds, which shorten CA-CA to ~2.8 A. The
# head-to-tail check below is therefore looser than phase1_closure.py's, deliberately.
_CA_BOND_RANGE = (2.7, 4.0)
_CLOSURE_MAX = 5.0


def read_ssbonds(path: str, chain: str, ca_resnums: list[int]) -> list[tuple[int, int]]:
    """Disulfide pairs as indices into the CA array, from the file's own SSBOND records."""
    idx = {r: i for i, r in enumerate(ca_resnums)}
    out = []
    for line in open(path):
        if not line.startswith("SSBOND"):
            continue
        c1, r1, c2, r2 = line[15], int(line[17:21]), line[29], int(line[31:35])
        if c1 == chain and c2 == chain and r1 in idx and r2 in idx:
            out.append((idx[r1], idx[r2]))
    return out


def ca_resnums(path: str, chain: str) -> list[int]:
    nums, seen, started = [], set(), False
    for line in open(path):
        if line[:6] in ("ENDMDL",) and started:
            break
        if line[:4] != "ATOM" or line[12:16] != " CA " or line[21] != chain:
            continue
        if line[16] not in (" ", "A"):
            continue
        r = int(line[22:26])
        if r in seen:
            continue
        seen.add(r)
        nums.append(r)
        started = True
    return nums


def build_topology(n: int, ss: list[tuple[int, int]], mode: str, x0: np.ndarray):
    """Return (sequential_bonds, closure_bonds_with_b0, angle_triples, steric_pairs)."""
    if mode == "disulfide":
        seq = [(i, i + 1) for i in range(n - 1)]           # linear backbone
        closure = [(i, j) for i, j in ss]
    elif mode == "bicycle":
        seq = [(i, i + 1) for i in range(n - 1)]
        closure = [(n - 1, 0)] + [(i, j) for i, j in ss]
    elif mode == "head_to_tail":
        seq = [(i, i + 1) for i in range(n - 1)]
        closure = [(n - 1, 0)]
    else:
        raise ValueError(mode)
    # Each closure restraint keeps its own native equilibrium length.
    closure_b0 = [(i, j, float(np.linalg.norm(x0[i] - x0[j]))) for i, j in closure]
    cyclic = mode in ("head_to_tail", "bicycle")
    triples = [((i - 1) % n, i, (i + 1) % n) for i in range(n)] if cyclic else \
              [(i - 1, i, i + 1) for i in range(1, n - 1)]
    sep = 3
    steric = [(i, j) for i in range(n) for j in range(i + sep, n)
              if (min(abs(i - j), n - abs(i - j)) if cyclic else abs(i - j)) >= sep]
    return (np.array(seq), closure_b0, np.array(triples),
            np.array(steric) if steric else np.zeros((0, 2), dtype=int))


def decompose(x, seq, closure_b0, triples, steric, target, p: EnergyParams):
    out = {}
    out["bond"] = _bond_energy_grad(x, seq[:, 0], seq[:, 1], p.k_bond, p.b0)
    g_cl = np.zeros_like(x)
    e_cl = 0.0
    for i, j, b0 in closure_b0:
        e, g = _bond_energy_grad(x, np.array([i]), np.array([j]), p.k_bond, b0)
        e_cl += e
        g_cl += g
    out["closure"] = (e_cl, g_cl)
    out["angle"] = _angle_energy_grad(x, triples, p.k_angle, p.theta0)
    out["steric"] = _steric_energy_grad(x, steric, p.k_steric, p.r_steric)
    if target is not None:
        out["interface"] = _interface_energy_grad(x, target, p.k_iface, p.r_iface)
    return out


def run(pdb_dir: str, n_t: int = 21, n_samples: int = 128, seed: int = 0):
    p = EnergyParams()
    rng = np.random.default_rng(seed)
    ts = np.linspace(0.0, 1.0, n_t)
    rows = []
    for pid, (pch, tch, label) in STRUCTURES.items():
        path = os.path.join(pdb_dir, f"{pid.lower()}.ent")
        if not os.path.exists(path):
            print(f"  {pid}: SKIPPED - no file at {path}", file=sys.stderr)
            continue
        x0 = parse_ca(path, pch)
        tgt = parse_ca(path, tch) if tch else None
        d = np.linalg.norm(x0[1:] - x0[:-1], axis=1)
        clo = float(np.linalg.norm(x0[-1] - x0[0]))
        if not (_CA_BOND_RANGE[0] <= d.min() and d.max() <= _CA_BOND_RANGE[1]
                and clo <= _CLOSURE_MAX):
            print(f"  {pid}: SKIPPED - not a contiguous head-to-tail macrocycle "
                  f"(CA-CA {d.min():.2f}-{d.max():.2f}, closure {clo:.2f})", file=sys.stderr)
            continue
        ss = read_ssbonds(path, pch, ca_resnums(path, pch))
        if not ss:
            print(f"  {pid}: SKIPPED - no intra-chain SSBOND records", file=sys.stderr)
            continue
        print(f"  {pid} ({label}): n={len(x0)} CA, {len(ss)} disulfide(s) "
              f"{ss}, head-to-tail closure {clo:.2f} A"
              + (f", target {len(tgt)} CA" if tgt is not None else ""), file=sys.stderr)
        rg = float(np.sqrt(((x0 - x0.mean(0)) ** 2).sum(1).mean()))
        for mode in ("head_to_tail", "disulfide", "bicycle"):
            seq, cb0, tri, ste = build_topology(len(x0), ss, mode, x0)
            for t in ts:
                fr, enr = [], []
                for _ in range(n_samples):
                    xt = noise(x0, float(t), rng, scale=rg)
                    terms = decompose(xt, seq, cb0, tri, ste, tgt, p)
                    nm = {k: float(np.linalg.norm(g)) for k, (_, g) in terms.items()}
                    tot = sum(nm.values())
                    fr.append(nm["closure"] / tot if tot > 1e-12 else np.nan)
                    per = nm["bond"] / max(len(seq), 1)
                    enr.append(nm["closure"] / per if per > 1e-12 else np.nan)
                rows.append({
                    "structure": pid, "label": label, "n_residues": len(x0),
                    "n_disulfides": len(ss), "topology": mode,
                    "n_closure_restraints": len(cb0), "t": round(float(t), 3),
                    "f_closure_mean": float(np.nanmean(fr)),
                    "closure_group_enrichment": float(np.nanmean(enr)),
                })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pdb-dir", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--samples", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    print("Loading disulfide-containing cyclic peptides...", file=sys.stderr)
    rows = run(args.pdb_dir, n_samples=args.samples, seed=args.seed)
    if not rows:
        raise SystemExit("no structures usable")

    path = os.path.join(args.out, "experiment_b1_chemistries.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    summary = {}
    print(f"\n{'topology':<14}{'f_cl':>8}{'closure-group enrichment':>28}   verdict", file=sys.stderr)
    for mode in ("head_to_tail", "disulfide", "bicycle"):
        sel = [r for r in rows if r["topology"] == mode and 0.3 <= r["t"] <= 0.9]
        f = float(np.mean([r["f_closure_mean"] for r in sel]))
        e = float(np.mean([r["closure_group_enrichment"] for r in sel]))
        v = ("DOMINATION" if f > 0.5 else "partial" if f >= 0.2 else "refuted")
        summary[mode] = {"f_closure": round(f, 4), "enrichment": round(e, 3), "verdict": v}
        print(f"{mode:<14}{f:>8.3f}{e:>28.2f}   {v}", file=sys.stderr)

    with open(os.path.join(args.out, "experiment_b1_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nwrote {path}", file=sys.stderr)


if __name__ == "__main__":
    main()
