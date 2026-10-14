#!/usr/bin/env python3
"""Finite-difference check of every analytic gradient in phase1_closure.py.

This exists because Experiment A's entire result is a ratio of gradient norms. A sign
error or a missing term in one gradient would not crash anything -- it would quietly
produce a plausible-looking number and a wrong conclusion. Run this before trusting any
output of phase1_closure.py.
"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
from phase1_closure import (                      # noqa: E402
    EnergyParams, Topology, decomposed_gradient, synthetic_ring, synthetic_target,
    kabsch_rmsd, cosine_alpha_bar, lam_closure_aware,
)

TOL = 2e-4


def numeric_grad(fn, x, h=1e-5):
    g = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        i = it.multi_index
        xp = x.copy(); xp[i] += h
        xm = x.copy(); xm[i] -= h
        g[i] = (fn(xp) - fn(xm)) / (2 * h)
        it.iternext()
    return g


def main():
    rng = np.random.default_rng(7)
    p = EnergyParams()
    fails = 0

    for n in (9, 14):
        pep = synthetic_ring(n, rng)
        pep += rng.normal(scale=1.5, size=pep.shape)      # perturb so no term is at zero
        tgt = synthetic_target(pep, rng)
        top = Topology(n)

        terms = decomposed_gradient(pep, top, tgt, p)
        for name in terms:
            def energy_of(x, _name=name):
                return decomposed_gradient(x, top, tgt, p)[_name][0]
            ana = terms[name][1]
            num = numeric_grad(energy_of, pep)
            denom = max(np.abs(num).max(), 1.0)
            err = np.abs(ana - num).max() / denom
            ok = err < TOL
            fails += not ok
            print(f"  n={n:<3} {name:<10} max rel err {err:.2e}  {'ok' if ok else 'FAIL'}")

    # Sanity properties that do not depend on the force field.
    x = synthetic_ring(12, rng)
    assert abs(kabsch_rmsd(x, x)) < 1e-8, "RMSD of a structure with itself must be 0"
    r = rng.normal(size=(3, 3)); q, _ = np.linalg.qr(r); q *= np.sign(np.linalg.det(q))
    assert kabsch_rmsd(x, x @ q.T + 5.0) < 1e-6, "RMSD must be invariant to rotation+translation"
    print("  kabsch_rmsd    invariance                      ok")

    ab = cosine_alpha_bar(np.array([0.0, 0.5, 1.0]))
    assert ab[0] > 0.99 and ab[2] < 0.01 and ab[0] > ab[1] > ab[2], f"bad schedule {ab}"
    print("  cosine_alpha_bar monotone, ~1 at t=0, ~0 at t=1 ok")

    # The closure-aware schedule must be near 0 when far and near lam_max when closed.
    assert lam_closure_aware(30.0) < 0.01, "schedule must be off when closure is far"
    # At the equilibrium CA-CA distance the restraint should be most of the way on.
    # With the defaults d_tol=8.0, tau=2.0 this is sigmoid(2.1) = 0.891, not ~1.0: the
    # schedule is deliberately soft, so the restraint is still ramping as the ring closes
    # rather than switching on. If tau is retuned, this number moves and that is expected.
    assert lam_closure_aware(3.8) > 0.85, "schedule must be mostly on when closure is in reach"
    assert abs(lam_closure_aware(8.0) - 0.5) < 1e-9, "schedule must be half-on at d_tol"
    print("  lam_closure_aware endpoints                    ok")

    if fails:
        print(f"\n{fails} gradient check(s) FAILED")
        sys.exit(1)
    print("\nall gradient and invariance checks passed")


if __name__ == "__main__":
    main()
