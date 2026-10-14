#!/usr/bin/env python3
"""
Extract the RFpeptides per-design table from the paper's Supplementary Information.

COMMAND LINE AS RUN
-------------------
    pdftotext -layout <MOESM1 supplement>.pdf si.txt
    python3 extract_rfpeptides_si.py --si-text si.txt --out data/tier1_rfpeptides_per_design.csv

SOURCE
------
Rettie SA et al., "Accurate de novo design of high-affinity protein-binding macrocycles
using deep learning", Nat Chem Biol 21:1948-1956 (2025), doi 10.1038/s41589-025-01929-w.
Supplementary Information PDF (MOESM1), obtained from the PMC Open Access mirror
(PMC12643943). Parse the text with `pdftotext -layout`.

WHAT IS JOINED
--------------
  Supplementary Tables 1b, 2b, 3b, 4b   per-design in-silico metrics for every DESIGN
                                        (contact molecular surface, Rosetta ddG, SAP, iPAE)
  Supplementary Tables 7, 8, 9, 10      which designs were chemically SYNTHESISED (purity)
  Supplementary Figures 6, 9, 12, 19    which designs BOUND in the SPR screen (captions)

That join is what makes a binder-vs-non-binder evaluation possible: the metrics alone do not
say which designs were tested, and the binder lists alone carry no scores.

TWO DEFINITIONS OF "BINDER", AND WHY BOTH ARE KEPT
--------------------------------------------------
The SPR screens report designs with "detectable binding at or below 100 uM"; a subset was
then "chosen for further characterization" and given a measured affinity. These give
different counts, and the difference is entirely in GABARAP:

    target    detectable (screen)                      further characterised
    MCL1      MCB_D2, MCB_D7, MCB_D26            (3)   MCB_D2, MCB_D26        (2)
    MDM2      MDB_D8, MDB_D10, MDB_D11           (3)   MDB_D8                 (1)
    GABARAP   GAB_D1, GAB_D8, GAB_D23, GAB_D28   (4)   GAB_D8, GAB_D23        (2)
    RbtA      RBB_D2, RBB_D7, RBB_D10, RBB_D14   (4)   RBB_D10                (1)

Detectable binding totals 14, which is why an earlier curation of this project that recorded
"12 binders" from the main text was counting GABARAP as 2 rather than 4. The screen criterion
is the one that is applied uniformly across all four targets and is used as the primary
label; `binder_characterised` is kept as a secondary, stricter label. Report which one any
statistic uses.

CENSORING
---------
A design that appears in a metrics table but not in a purity table was designed and scored
but never tested, in most cases because of poor cyclization yield. It has no binding label
and is marked censored=True. It must not be scored as a non-binder: the failure is a
synthetic-chemistry outcome that may correlate with the very structural features being
evaluated.

    python extract_rfpeptides_si.py --si-text si1.txt --out data/tier1_rfpeptides_per_design.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import sys

TARGETS = {
    "MCB": "MCL1",
    "MDB": "MDM2",
    "GAB": "GABARAP",
    "RBB": "RbtA",
}

# Read off the Supplementary Figure captions (6, 9, 12, 19). "MDB_11" is the SI's own
# typo for MDB_D11.
BINDERS_SCREEN = {
    "MCL1": {"MCB_D2", "MCB_D7", "MCB_D26"},
    "MDM2": {"MDB_D8", "MDB_D10", "MDB_D11"},
    "GABARAP": {"GAB_D1", "GAB_D8", "GAB_D23", "GAB_D28"},
    "RbtA": {"RBB_D2", "RBB_D7", "RBB_D10", "RBB_D14"},
}
BINDERS_CHARACTERISED = {
    "MCL1": {"MCB_D2", "MCB_D26"},
    "MDM2": {"MDB_D8"},
    "GABARAP": {"GAB_D8", "GAB_D23"},
    "RbtA": {"RBB_D10"},
}

# Measured affinities stated in the main text, for the designs that have one.
AFFINITY_NM = {
    "MCB_D2": 2000.0,     # ~2 uM, SPR
    "MDB_D8": 1900.0,     # ~1.9 uM, SPR
    "GAB_D8": 6.0,
    "GAB_D23": 36.0,
    "RBB_D10": 9.4,
}

_METRIC_ROW = re.compile(
    r"^\s*((?:MCB|MDB|GAB|RBB)_D\d+)\s+"
    r"(-?\d+\.?\d*)\s+(-?\d+\.?\d*)\s+(-?\d+\.?\d*)\s+(-?\d+\.?\d*)\s*$")
_PURITY_ROW = re.compile(
    r"^\s*((?:MCB|MDB|GAB|RBB)_D\d+)\s+([A-Z]{5,})\s+(\d+\.?\d*)\s*$")


def parse(si_text: str):
    metrics, purity = {}, {}
    for line in open(si_text):
        m = _METRIC_ROW.match(line)
        if m:
            d = m.group(1)
            metrics[d] = {
                "contact_molecular_surface_A2": float(m.group(2)),
                "rosetta_ddg_kcal_mol": float(m.group(3)),
                "sap": float(m.group(4)),
                "ipae": float(m.group(5)),
            }
            continue
        p = _PURITY_ROW.match(line)
        if p:
            purity[p.group(1)] = {"sequence": p.group(2), "purity_pct": float(p.group(3))}
    return metrics, purity


def build(metrics, purity):
    rows = []
    for design in sorted(metrics, key=lambda d: (d.split("_")[0], int(d.split("_D")[1]))):
        prefix = design.split("_")[0]
        target = TARGETS[prefix]
        synthesised = design in purity
        rows.append({
            "design_id": design,
            "target": target,
            "synthesised": synthesised,
            "censored": not synthesised,
            "binder_screen": (design in BINDERS_SCREEN[target]) if synthesised else "",
            "binder_characterised": (design in BINDERS_CHARACTERISED[target]) if synthesised else "",
            "affinity_nM": AFFINITY_NM.get(design, ""),
            "sequence": purity.get(design, {}).get("sequence", ""),
            "purity_pct": purity.get(design, {}).get("purity_pct", ""),
            **metrics[design],
            "source": "rettie2025rfpeptides",
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--si-text", required=True,
                    help="output of: pdftotext -layout <MOESM1>.pdf si1.txt")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    metrics, purity = parse(args.si_text)
    rows = build(metrics, purity)

    # Integrity checks against counts stated independently in the paper's main text.
    # These are the reason to trust the parse; if one fails, the extraction is wrong.
    problems = []
    n_synth = sum(r["synthesised"] for r in rows)
    if n_synth != 39:
        problems.append(f"synthesised={n_synth}, main text says 39")
    for tgt, expect in (("MCL1", 14), ("MDM2", 8), ("GABARAP", 6), ("RbtA", 11)):
        got = sum(r["synthesised"] for r in rows if r["target"] == tgt)
        if got != expect:
            problems.append(f"{tgt} synthesised={got}, main text says {expect}")
    for tgt, s in BINDERS_SCREEN.items():
        missing = s - {r["design_id"] for r in rows if r["synthesised"]}
        if missing:
            problems.append(f"{tgt} binders not found among synthesised: {sorted(missing)}")
    no_metrics = [d for s in BINDERS_SCREEN.values() for d in s if d not in metrics]
    if no_metrics:
        problems.append(f"binders with no published metrics: {sorted(no_metrics)}")

    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"designs with metrics : {len(rows)}", file=sys.stderr)
    print(f"synthesised          : {n_synth}", file=sys.stderr)
    print(f"censored (designed, never tested): {sum(r['censored'] for r in rows)}",
          file=sys.stderr)
    print(f"binders (screen)     : {sum(r['binder_screen'] is True for r in rows)}",
          file=sys.stderr)
    print(f"binders (characterised): "
          f"{sum(r['binder_characterised'] is True for r in rows)}", file=sys.stderr)
    for tgt in TARGETS.values():
        g = [r for r in rows if r["target"] == tgt]
        print(f"  {tgt:<9} designed {len(g):>3}  synthesised "
              f"{sum(r['synthesised'] for r in g):>3}  binders "
              f"{sum(r['binder_screen'] is True for r in g):>2}", file=sys.stderr)
    if problems:
        print("\nINTEGRITY CHECK FAILED:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        sys.exit(1)
    print(f"\nall integrity checks passed; wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
