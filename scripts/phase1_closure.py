#!/usr/bin/env python3
"""
Phase 1: is the closure term dominating the energy-guidance gradient?

COMMAND LINE AS RUN (Section 4.1 of the manuscript)
---------------------------------------------------
    python3 phase1_closure.py --out results --seed 0

Structures are downloaded to --cache (default pdb_cache/) on first run. If RCSB is
unreachable, fetch the same accessions from the wwPDB AWS mirror into a directory and
pass --local-dir instead.

THE CLAIM BEING TESTED
----------------------
CP-Composer reports its energy-guided baseline scoring 0% on disulfide and bicycle
cyclization. The hypothesis in this project is that this is not a failure of energy
guidance as such, but an artefact of *when* the closure restraint is applied. During
denoising the ring is not yet closed, so a force field evaluated on a topology that
asserts the ring-closing bond sees an enormous deviation on that one bond. Its gradient
then dominates, and guidance degenerates into dragging the two closure partners toward
each other while the rest of the geometry is sacrificed.

If that is right, it is fixable with a closure-aware schedule: ramp the closure restraint
by how geometrically feasible closure currently is, instead of imposing it from step one.

WHAT THIS SCRIPT DOES, AND WHAT IT DOES NOT
-------------------------------------------
Experiment A (the real diagnostic) measures the fraction of the total energy gradient
contributed by the ring-closing bond, as a function of diffusion time, on noised real
macrocyclic peptide backbones. It needs no generative model and no GPU. It is a direct,
cheap measurement of the mechanism the hypothesis names.

Experiment B is a TOY. It runs annealed Langevin refinement under two closure schedules
and compares the end states. It illustrates the consequence; it does not establish it.
A toy cannot stand in for running CP-Composer itself.

Neither experiment can confirm the hypothesis for CP-Composer specifically, because
CP-Composer diffuses in a learned latent space and this works in Cartesian coordinates.
What a positive result here buys is justification for spending GPU time on the real
experiment. What a negative result buys is the knowledge not to.

A NOTE ON METRIC CHOICE, RECORDED BEFORE ANY REAL STRUCTURE WAS RUN
-------------------------------------------------------------------
Experiment A reports three numbers, and the distinction between them is deliberate.

The PRIMARY metric, f_cl, is the closure bond's share of the total summed gradient norm.
Its thresholds below were fixed in advance and are not to be moved.

Two SECONDARY metrics were added on 12 Oct 2026 after a smoke run on SYNTHETIC rings
returned f_cl ~ 0.18, i.e. below the domination threshold. They were added because that
run exposed a measurement problem, not because the number was unwelcome:

  - f_cl compares ONE bond against the aggregate of n-1 sequential bonds. For a 14-mer
    that is one against thirteen, so even a closure bond carrying far more gradient than
    any individual bond lands at a small share. The denominator grows with ring size,
    which makes f_cl not comparable across peptides of different length.
  - What guidance actually suffers from is not the magnitude of the closure gradient but
    the extent to which the step taken is the step closure wanted. That is a direction
    question, and a norm ratio cannot answer it.

So: `closure_per_bond_enrichment` = ||g_closure|| / (||g_bond|| / n_sequential_bonds),
and `closure_gradient_alignment` = cos(g_total, g_closure).

These two have NO pre-registered threshold and do not decide the verdict. They are
reported so the effect can be described honestly whichever way f_cl falls. If the paper
ends up leaning on them, that must be stated as a post-hoc choice with this note cited.
The synthetic smoke number is not evidence about peptides and is recorded only to show
when and why the metrics changed.

HOW TO READ THE RESULT
----------------------
Experiment A reports f_cl(t), the closure share of the gradient. Interpretation agreed
in advance, so it cannot be moved after seeing the numbers:

  f_cl > 0.5 over most of t in [0.3, 0.9]   -> consistent with the hypothesis; proceed
  f_cl in [0.2, 0.5]                        -> closure matters but does not dominate;
                                               the framing must be softened to "a
                                               substantial contribution", and the paper's
                                               central claim needs rewording
  f_cl < 0.2                                -> hypothesis refuted at the cheap stage.
                                               Say so in the paper. Do not reframe to
                                               rescue it.

A single-chemistry result is not general: head-to-tail, disulfide and bicycle close
through different geometry, and CP-Composer's 0% was on disulfide and bicycle, not
head-to-tail. Run all three before claiming anything about CP-Composer's failure.

USAGE
-----
    pip install numpy scipy matplotlib requests
    python phase1_closure.py --out results/

    # offline / smoke test, synthetic rings instead of PDB downloads
    python phase1_closure.py --synthetic --out results/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field

import numpy as np

# --------------------------------------------------------------------------------------
# Structures
# --------------------------------------------------------------------------------------

# Deposited complexes from the RFpeptides paper (Rettie et al., Nat Chem Biol 21:1948,
# 2025). Each assignment was verified individually on RCSB on 12 Oct 2026.
#   pdb_id -> (peptide_chain, target_chain, design_name)
RFPEPTIDES_COMPLEXES = {
    "9CDT": ("B", "A", "MCB_D2"),    # MCL-1 complex
    "9CDU": ("B", "A", "RBB_D10"),   # Rhombotarget A complex, 2.61 A
    "9HGC": ("B", "A", "GAB_D8"),    # GABARAPL1 complex, 2.52 A
    "9HGD": ("B", "A", "GAB_D23"),   # GABARAP complex, 1.50 A
    "9CDZ": ("B", "A", "AfCycDesign_MDM2"),  # MDM2 complex, 1.72 A -- AfCycDesign, not RFpeptides
}

# Sequential CA-CA spacing and ring-closure distance expected of a real head-to-tail
# macrocyclic peptide backbone. Every parsed peptide is checked against these before it is
# used, because a mis-specified chain id or a truncated download would otherwise produce a
# plausible-looking number from the wrong atoms.
_CA_BOND_RANGE = (3.6, 4.0)
_CLOSURE_MAX = 5.0


def validate_macrocycle(x, name: str) -> dict:
    """Confirm a parsed chain really is a cyclic peptide backbone. Raises if it is not."""
    d = np.linalg.norm(x[1:] - x[:-1], axis=1)
    closure = float(np.linalg.norm(x[-1] - x[0]))
    lo, hi = _CA_BOND_RANGE
    if not (d.min() >= lo and d.max() <= hi):
        raise ValueError(
            f"{name}: sequential CA-CA distances {d.min():.2f}-{d.max():.2f} A fall outside "
            f"{lo}-{hi} A. This is not a contiguous backbone -- check the chain id and that "
            f"the file is complete."
        )
    if closure > _CLOSURE_MAX:
        raise ValueError(
            f"{name}: first-to-last CA distance {closure:.2f} A exceeds {_CLOSURE_MAX} A, so "
            f"this chain is not head-to-tail cyclic. Do not use it for a closure experiment."
        )
    return {"n": int(len(x)), "ca_bond_min": float(d.min()), "ca_bond_max": float(d.max()),
            "closure_dist": closure}

RCSB_PDB_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"


def fetch_pdb(pdb_id: str, cache_dir: str) -> str:
    """Download a PDB file, caching it. Needs open network; blocked in some containers."""
    import requests

    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{pdb_id}.pdb")
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    r = requests.get(RCSB_PDB_URL.format(pdb_id=pdb_id), timeout=60)
    r.raise_for_status()
    with open(path, "w") as fh:
        fh.write(r.text)
    return path


def parse_ca(pdb_path: str, chain: str) -> np.ndarray:
    """Parse CA coordinates for one chain from a PDB file. No Biopython dependency.

    Takes altloc blank or 'A' only, and stops at the first TER/ENDMDL for that chain so a
    second copy in the asymmetric unit is not silently concatenated onto the first.
    """
    coords: list[list[float]] = []
    seen_chain = False
    with open(pdb_path) as fh:
        for line in fh:
            rec = line[:6]
            if rec in ("TER   ", "ENDMDL"):
                if seen_chain:
                    break
                continue
            if rec not in ("ATOM  ", "HETATM"):
                continue
            if line[21] != chain:
                continue
            if line[12:16].strip() != "CA":
                continue
            if line[16] not in (" ", "A"):
                continue
            seen_chain = True
            coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    if not coords:
        raise ValueError(f"no CA atoms found for chain {chain!r} in {pdb_path}")
    return np.asarray(coords, dtype=float)


def synthetic_ring(n: int, rng: np.random.Generator, bond: float = 3.8) -> np.ndarray:
    """A closed ring of n CA atoms with roughly correct CA-CA spacing, mildly puckered.

    Used only for the --synthetic smoke test. It is NOT a peptide: it has no real
    secondary structure and no real side chains, so numbers from it are for checking
    that the code runs, never for the paper.
    """
    radius = bond / (2.0 * np.sin(np.pi / n))
    ang = 2.0 * np.pi * np.arange(n) / n
    x = np.stack([radius * np.cos(ang), radius * np.sin(ang), np.zeros(n)], axis=1)
    x += rng.normal(scale=0.25, size=x.shape)
    return x


def synthetic_target(peptide: np.ndarray, rng: np.random.Generator, n: int = 60) -> np.ndarray:
    """A slab of fixed points sitting under the ring, standing in for an epitope."""
    centre = peptide.mean(axis=0)
    pts = rng.normal(scale=6.0, size=(n, 3))
    pts[:, 2] = -np.abs(pts[:, 2]) - 6.0
    return pts + centre


# --------------------------------------------------------------------------------------
# Energy
# --------------------------------------------------------------------------------------

@dataclass
class EnergyParams:
    """Coarse CA-level force field.

    Deliberately simple, and that is a limitation to state in the paper: a CA-only model
    cannot represent side-chain packing or electrostatics, so the absolute energies are
    meaningless. What it CAN represent faithfully is the thing under test — the relative
    magnitude of a harmonic restraint on one badly violated bond versus the rest of the
    bonded and non-bonded terms. The closure bond here is given exactly the same force
    constant and equilibrium length as every other backbone bond. Nothing is done to make
    it dominate. If it dominates, that is geometry, not parameterisation.
    """
    k_bond: float = 100.0        # kcal/mol/A^2, virtual CA-CA bond
    b0: float = 3.80             # A, trans peptide CA-CA distance
    k_angle: float = 20.0        # kcal/mol/rad^2
    theta0: float = 1.92         # rad, ~110 deg
    k_steric: float = 10.0       # kcal/mol/A^2
    r_steric: float = 4.5        # A, onset of soft repulsion
    k_iface: float = 2.0         # kcal/mol/A^2
    r_iface: float = 6.0         # A, flat-bottom radius for interface attraction
    sep_min: int = 3             # minimum |i-j| for a steric pair


def _bond_energy_grad(x, idx_i, idx_j, k, b0):
    d = x[idx_j] - x[idx_i]
    r = np.linalg.norm(d, axis=1)
    r_safe = np.where(r > 1e-8, r, 1e-8)
    e = k * np.sum((r - b0) ** 2)
    coef = (2.0 * k * (r - b0) / r_safe)[:, None] * d
    g = np.zeros_like(x)
    np.add.at(g, idx_i, -coef)
    np.add.at(g, idx_j, coef)
    return e, g


def _angle_energy_grad(x, triples, k, theta0):
    """Harmonic on the CA-CA-CA pseudo-angle. Analytic gradient."""
    g = np.zeros_like(x)
    e = 0.0
    i, j, m = triples[:, 0], triples[:, 1], triples[:, 2]
    u = x[i] - x[j]
    v = x[m] - x[j]
    nu = np.linalg.norm(u, axis=1)
    nv = np.linalg.norm(v, axis=1)
    nu = np.where(nu > 1e-8, nu, 1e-8)
    nv = np.where(nv > 1e-8, nv, 1e-8)
    cos = np.clip(np.einsum("ij,ij->i", u, v) / (nu * nv), -1.0 + 1e-7, 1.0 - 1e-7)
    theta = np.arccos(cos)
    e = k * np.sum((theta - theta0) ** 2)
    dE_dtheta = 2.0 * k * (theta - theta0)
    sin = np.sqrt(1.0 - cos ** 2)
    dtheta_dcos = -1.0 / sin
    # d(cos)/du and d(cos)/dv
    dcos_du = (v / (nu * nv)[:, None]) - (cos / nu ** 2)[:, None] * u
    dcos_dv = (u / (nu * nv)[:, None]) - (cos / nv ** 2)[:, None] * v
    gu = (dE_dtheta * dtheta_dcos)[:, None] * dcos_du
    gv = (dE_dtheta * dtheta_dcos)[:, None] * dcos_dv
    np.add.at(g, i, gu)
    np.add.at(g, m, gv)
    np.add.at(g, j, -(gu + gv))
    return e, g


def _steric_energy_grad(x, pairs, k, r0):
    if len(pairs) == 0:
        return 0.0, np.zeros_like(x)
    i, j = pairs[:, 0], pairs[:, 1]
    d = x[j] - x[i]
    r = np.linalg.norm(d, axis=1)
    r_safe = np.where(r > 1e-8, r, 1e-8)
    viol = r < r0
    e = k * np.sum((r0 - r[viol]) ** 2)
    coef = np.zeros(len(pairs))
    coef[viol] = -2.0 * k * (r0 - r[viol]) / r_safe[viol]
    cv = coef[:, None] * d
    g = np.zeros_like(x)
    np.add.at(g, i, -cv)
    np.add.at(g, j, cv)
    return e, g


def _interface_energy_grad(x, target, k, r0):
    """Flat-bottom attraction of each peptide atom to its nearest target point.

    Zero inside r0, harmonic outside. This stands in for the interface term a real
    guidance scheme would use. Nearest-neighbour is recomputed each call, so the term is
    piecewise smooth; that is fine for gradient-magnitude accounting and is noted as a
    limitation for the Langevin run.
    """
    diff = x[:, None, :] - target[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    nn = np.argmin(dist, axis=1)
    rows = np.arange(len(x))
    r = dist[rows, nn]
    d = diff[rows, nn]
    r_safe = np.where(r > 1e-8, r, 1e-8)
    viol = r > r0
    e = k * np.sum((r[viol] - r0) ** 2)
    coef = np.zeros(len(x))
    coef[viol] = 2.0 * k * (r[viol] - r0) / r_safe[viol]
    g = coef[:, None] * d
    return e, g


@dataclass
class Topology:
    n: int
    seq_bonds: np.ndarray = field(init=False)
    closure_bond: np.ndarray = field(init=False)
    triples: np.ndarray = field(init=False)
    steric_pairs: np.ndarray = field(init=False)

    def __post_init__(self):
        n = self.n
        self.seq_bonds = np.array([[i, i + 1] for i in range(n - 1)])
        # THE closure bond: the one that makes the chain a ring.
        self.closure_bond = np.array([[n - 1, 0]])
        self.triples = np.array([[(i - 1) % n, i, (i + 1) % n] for i in range(n)])
        sep = self.sep_min_default = 3
        self.steric_pairs = np.array(
            [[i, j] for i in range(n) for j in range(i + sep, n)
             if min(abs(i - j), n - abs(i - j)) >= sep]
        ) if n > 2 * sep else np.zeros((0, 2), dtype=int)


def decomposed_gradient(x, top: Topology, target, p: EnergyParams, lam_closure: float = 1.0):
    """Return {term: (energy, gradient)} with the closure bond kept separate.

    lam_closure scales ONLY the ring-closing bond. At 1.0 the closure bond is treated
    exactly like every other backbone bond, which is the condition the hypothesis is about.
    """
    out = {}
    out["bond"] = _bond_energy_grad(x, top.seq_bonds[:, 0], top.seq_bonds[:, 1], p.k_bond, p.b0)
    e_cl, g_cl = _bond_energy_grad(
        x, top.closure_bond[:, 0], top.closure_bond[:, 1], p.k_bond, p.b0)
    out["closure"] = (lam_closure * e_cl, lam_closure * g_cl)
    out["angle"] = _angle_energy_grad(x, top.triples, p.k_angle, p.theta0)
    out["steric"] = _steric_energy_grad(x, top.steric_pairs, p.k_steric, p.r_steric)
    if target is not None:
        out["interface"] = _interface_energy_grad(x, target, p.k_iface, p.r_iface)
    return out


# --------------------------------------------------------------------------------------
# Forward diffusion
# --------------------------------------------------------------------------------------

def cosine_alpha_bar(t: np.ndarray, s: float = 0.008) -> np.ndarray:
    """Nichol & Dhariwal cosine schedule, t in [0, 1]."""
    f = np.cos(((t + s) / (1.0 + s)) * np.pi / 2.0) ** 2
    f0 = np.cos((s / (1.0 + s)) * np.pi / 2.0) ** 2
    return np.clip(f / f0, 1e-8, 1.0)


def noise(x0: np.ndarray, t: float, rng: np.random.Generator, scale: float) -> np.ndarray:
    """q(x_t | x_0) for coordinate-space diffusion, on centred coordinates.

    `scale` sets the variance of the terminal distribution in Angstrom. It is set from the
    native radius of gyration so that t=1 is a cloud of roughly protein-like extent rather
    than unit-variance noise, which would be a much easier problem than the real one.
    """
    ab = cosine_alpha_bar(np.array([t]))[0]
    c = x0 - x0.mean(axis=0)
    eps = rng.normal(size=c.shape) * scale
    return np.sqrt(ab) * c + np.sqrt(1.0 - ab) * eps + x0.mean(axis=0)


# --------------------------------------------------------------------------------------
# Experiment A: gradient decomposition
# --------------------------------------------------------------------------------------

def experiment_a(structures, p: EnergyParams, n_t: int = 21, n_samples: int = 64, seed: int = 0):
    """Closure share of the gradient as a function of diffusion time.

    f_cl(t) = ||g_closure|| / sum_terms ||g_term||

    Reported as a per-structure mean with a bootstrap 95% interval over samples, because
    a single number with no spread would be the kind of claim this project is trying not
    to make.
    """
    rng = np.random.default_rng(seed)
    ts = np.linspace(0.0, 1.0, n_t)
    rows = []
    for name, (pep, tgt) in structures.items():
        top = Topology(len(pep))
        rg = float(np.sqrt(((pep - pep.mean(0)) ** 2).sum(1).mean()))
        for t in ts:
            fracs, enrich, align, closure_dists = [], [], [], []
            n_seq = len(top.seq_bonds)
            for _ in range(n_samples):
                xt = noise(pep, float(t), rng, scale=rg)
                terms = decomposed_gradient(xt, top, tgt, p)
                norms = {k: float(np.linalg.norm(g)) for k, (_, g) in terms.items()}
                total = sum(norms.values())

                # PRIMARY, pre-registered: closure's share of the summed gradient norm.
                fracs.append(norms["closure"] / total if total > 1e-12 else np.nan)

                # SECONDARY 1: per-bond enrichment. The closure bond is ONE bond; the
                # "bond" term aggregates n-1. Comparing one against many understates it
                # by construction and by a factor that grows with ring size.
                per_seq = norms["bond"] / max(n_seq, 1)
                enrich.append(norms["closure"] / per_seq if per_seq > 1e-12 else np.nan)

                # SECONDARY 2: directional alignment. The question that actually matters
                # for guidance is not how big the closure gradient is but how much of the
                # step taken is the step closure wanted.
                g_tot = sum(g for _, g in terms.values())
                g_cl = terms["closure"][1]
                den = np.linalg.norm(g_tot) * np.linalg.norm(g_cl)
                align.append(float((g_tot * g_cl).sum() / den) if den > 1e-12 else np.nan)

                closure_dists.append(float(np.linalg.norm(xt[-1] - xt[0])))

            fr = np.asarray(fracs)
            boot = np.array([
                np.nanmean(rng.choice(fr, size=len(fr), replace=True)) for _ in range(500)
            ])
            rows.append({
                "structure": name,
                "n_residues": int(len(pep)),
                "t": float(t),
                "alpha_bar": float(cosine_alpha_bar(np.array([t]))[0]),
                "f_closure_mean": float(np.nanmean(fr)),
                "f_closure_lo95": float(np.percentile(boot, 2.5)),
                "f_closure_hi95": float(np.percentile(boot, 97.5)),
                "closure_per_bond_enrichment": float(np.nanmean(enrich)),
                "closure_gradient_alignment": float(np.nanmean(align)),
                "closure_dist_mean_A": float(np.mean(closure_dists)),
            })
    return rows


def verdict_from_a(rows) -> dict:
    """Apply the pre-registered thresholds from the module docstring. No post-hoc moving."""
    mid = [r["f_closure_mean"] for r in rows if 0.3 <= r["t"] <= 0.9]
    if not mid:
        return {"verdict": "no data", "mean_f_closure_mid_t": None}
    m = float(np.mean(mid))
    frac_over_half = float(np.mean([v > 0.5 for v in mid]))
    enr = float(np.mean([r["closure_per_bond_enrichment"] for r in rows
                         if 0.3 <= r["t"] <= 0.9]))
    aln = float(np.mean([r["closure_gradient_alignment"] for r in rows
                         if 0.3 <= r["t"] <= 0.9]))
    if m > 0.5 and frac_over_half > 0.5:
        v = "CONSISTENT with closure domination - proceed to the CP-Composer experiment"
    elif m >= 0.2:
        v = ("PARTIAL - closure contributes substantially but does not dominate. "
             "Reword the central claim; do not call it domination.")
    else:
        v = ("REFUTED at the cheap stage. Report this. The paper's framing must change, "
             "and CP-Composer's 0% needs a different explanation.")
    return {
        "verdict": v,
        "mean_f_closure_mid_t": m,
        "fraction_of_t_over_0.5": frac_over_half,
        "mean_per_bond_enrichment_mid_t": enr,
        "mean_gradient_alignment_mid_t": aln,
        "secondary_note": (
            "Enrichment and alignment are SECONDARY and carry no pre-registered "
            "threshold. They describe the effect; only f_closure decides the verdict."
        ),
    }


# --------------------------------------------------------------------------------------
# Experiment B (TOY): constant vs closure-aware schedule
# --------------------------------------------------------------------------------------

def lam_closure_aware(d_cl: float, d_tol: float = 8.0, tau: float = 2.0,
                      lam_max: float = 1.0) -> float:
    """lambda_cl(t) = lam_max * sigmoid((d_tol - d_cl) / tau).

    Near zero while the closure partners are far apart, approaching lam_max as they come
    within reach. The restraint is applied when it is geometrically meaningful rather than
    from step one.
    """
    return float(lam_max / (1.0 + np.exp(-(d_tol - d_cl) / tau)))


def anneal(x_init, top, target, p, schedule: str, n_steps=2000, lr=5e-4,
           temp0=0.5, seed=0):
    """Annealed gradient descent with decaying noise. A toy optimiser, not a sampler.

    schedule: 'constant' applies the closure restraint at full strength throughout;
              'aware'    ramps it by current closure distance.
    """
    rng = np.random.default_rng(seed)
    x = x_init.copy()
    traj = []
    for step in range(n_steps):
        frac = step / n_steps
        temp = temp0 * (1.0 - frac)
        d_cl = float(np.linalg.norm(x[-1] - x[0]))
        lam = 1.0 if schedule == "constant" else lam_closure_aware(d_cl)
        terms = decomposed_gradient(x, top, target, p, lam_closure=lam)
        g = sum(gr for _, gr in terms.values())
        gn = np.linalg.norm(g)
        if gn > 1e3:                      # clip, or the first steps explode
            g = g * (1e3 / gn)
        x = x - lr * g + rng.normal(scale=temp * np.sqrt(2.0 * lr), size=x.shape)
        if step % 50 == 0:
            traj.append({"step": step, "lam": lam, "d_closure": d_cl,
                         "E": float(sum(e for e, _ in terms.values()))})
    return x, traj


def kabsch_rmsd(a: np.ndarray, b: np.ndarray) -> float:
    ac = a - a.mean(0)
    bc = b - b.mean(0)
    u, _, vt = np.linalg.svd(ac.T @ bc)
    d = np.sign(np.linalg.det(u @ vt))
    r = u @ np.diag([1.0, 1.0, d]) @ vt
    return float(np.sqrt(((ac @ r - bc) ** 2).sum(1).mean()))


def experiment_b(structures, p: EnergyParams, n_restarts: int = 8, seed: int = 0):
    rng = np.random.default_rng(seed)
    rows = []
    for name, (pep, tgt) in structures.items():
        top = Topology(len(pep))
        rg = float(np.sqrt(((pep - pep.mean(0)) ** 2).sum(1).mean()))
        for rep in range(n_restarts):
            x0 = noise(pep, 0.9, rng, scale=rg)
            for sched in ("constant", "aware"):
                xf, traj = anneal(x0, top, tgt, p, sched, seed=int(rng.integers(1 << 30)))
                bonds = np.linalg.norm(xf[1:] - xf[:-1], axis=1)
                rows.append({
                    "structure": name, "restart": rep, "schedule": sched,
                    "rmsd_to_native_A": kabsch_rmsd(xf, pep),
                    "closure_dist_A": float(np.linalg.norm(xf[-1] - xf[0])),
                    "bond_rmsd_A": float(np.sqrt(np.mean((bonds - p.b0) ** 2))),
                    "radius_gyration_A": float(
                        np.sqrt(((xf - xf.mean(0)) ** 2).sum(1).mean())),
                    "native_radius_gyration_A": rg,
                })
    return rows


# --------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------

def load_structures(synthetic: bool, cache_dir: str, seed: int = 0,
                    local_dir: str | None = None):
    """Load peptide/target CA coordinate pairs.

    local_dir: a directory of already-downloaded `pdb<id>.ent` / `<id>.ent` files, for
    environments whose network cannot reach RCSB. Files obtained any other way are fine so
    long as they are the real archive files; every peptide is validated geometrically below
    either way, so a wrong or corrupt file fails loudly rather than quietly.
    """
    rng = np.random.default_rng(seed)
    out = {}
    if synthetic:
        for n in (11, 13, 15):
            pep = synthetic_ring(n, rng)
            out[f"synthetic_n{n}"] = (pep, synthetic_target(pep, rng))
        return out
    for pdb_id, (pep_chain, tgt_chain, design) in RFPEPTIDES_COMPLEXES.items():
        try:
            if local_dir:
                import glob
                hits = (glob.glob(os.path.join(local_dir, f"pdb{pdb_id.lower()}.ent"))
                        or glob.glob(os.path.join(local_dir, f"{pdb_id.lower()}.ent"))
                        or glob.glob(os.path.join(local_dir, f"{pdb_id.upper()}.pdb")))
                if not hits:
                    raise FileNotFoundError(f"no local file for {pdb_id} in {local_dir}")
                path = hits[0]
            else:
                path = fetch_pdb(pdb_id, cache_dir)
            pep = parse_ca(path, pep_chain)
            tgt = parse_ca(path, tgt_chain)
            chk = validate_macrocycle(pep, f"{pdb_id} chain {pep_chain}")
        except Exception as exc:                       # noqa: BLE001
            print(f"  {pdb_id} ({design}): SKIPPED - {exc}", file=sys.stderr)
            continue
        print(f"  {pdb_id} ({design}): peptide {chk['n']} CA "
              f"(CA-CA {chk['ca_bond_min']:.2f}-{chk['ca_bond_max']:.2f} A, "
              f"closure {chk['closure_dist']:.2f} A), target {len(tgt)} CA", file=sys.stderr)
        out[f"{pdb_id}_{design}"] = (pep, tgt)
    if not out:
        raise SystemExit(
            "No structures loaded. If the network is blocked here, rerun with --synthetic "
            "for a smoke test, but do not put synthetic numbers in the paper."
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="phase1_results")
    ap.add_argument("--cache", default="pdb_cache")
    ap.add_argument("--synthetic", action="store_true",
                    help="use synthetic rings instead of downloading PDB files")
    ap.add_argument("--local-dir", default=None,
                    help="directory of already-downloaded PDB .ent files, for networks that "
                         "cannot reach RCSB")
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--restarts", type=int, default=8)
    ap.add_argument("--skip-b", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    p = EnergyParams()

    print("Loading structures...", file=sys.stderr)
    structures = load_structures(args.synthetic, args.cache, args.seed,
                                 local_dir=args.local_dir)

    print("Experiment A: gradient decomposition vs diffusion time...", file=sys.stderr)
    rows_a = experiment_a(structures, p, n_samples=args.samples, seed=args.seed)
    _write_csv(os.path.join(args.out, "experiment_a_gradient_decomposition.csv"), rows_a)
    v = verdict_from_a(rows_a)

    rows_b = []
    if not args.skip_b:
        print("Experiment B (toy): constant vs closure-aware schedule...", file=sys.stderr)
        rows_b = experiment_b(structures, p, n_restarts=args.restarts, seed=args.seed)
        _write_csv(os.path.join(args.out, "experiment_b_schedules.csv"), rows_b)

    summary = {
        "synthetic": bool(args.synthetic),
        "structures": {k: int(len(v2[0])) for k, v2 in structures.items()},
        "experiment_a_verdict": v,
    }
    if rows_b:
        summary["experiment_b_toy"] = _summarise_b(rows_b)
    with open(os.path.join(args.out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    print("\n=== Experiment A ===", file=sys.stderr)
    print(f"mean closure gradient share over t in [0.3, 0.9]: "
          f"{v['mean_f_closure_mid_t']:.3f}", file=sys.stderr)
    print(f"VERDICT: {v['verdict']}", file=sys.stderr)
    if args.synthetic:
        print("\nNOTE: --synthetic was used. These rings are not peptides. This run checks "
              "that the code executes; it is not evidence about anything.", file=sys.stderr)
    if rows_b:
        print("\n=== Experiment B (toy, illustrative only) ===", file=sys.stderr)
        for k, val in summary["experiment_b_toy"].items():
            print(f"  {k}: {val}", file=sys.stderr)
    print(f"\nwrote {args.out}/", file=sys.stderr)


def _summarise_b(rows):
    out = {}
    for sched in ("constant", "aware"):
        sel = [r for r in rows if r["schedule"] == sched]
        if not sel:
            continue
        for key in ("rmsd_to_native_A", "closure_dist_A", "bond_rmsd_A",
                    "radius_gyration_A"):
            vals = np.array([r[key] for r in sel])
            out[f"{sched}.{key}"] = f"{vals.mean():.2f} +/- {vals.std():.2f}"
    return out


def _write_csv(path, rows):
    import csv
    if not rows:
        return
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
