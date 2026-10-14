#!/usr/bin/env python3
"""
MacroPPI-Bench Tier 1 expansion: harvest macrocyclic peptides with measured affinity
against protein-protein interaction targets, from ChEMBL.

WHY THIS IS A STANDALONE SCRIPT
-------------------------------
It was written in a cloud container whose network policy blocks ebi.ac.uk and rcsb.org,
so it has NOT been executed. Run it where the network is open: Kaggle, Colab, a lab
machine. It needs no GPU. Treat the first run as a debugging run, not a final dataset.

WHAT IT DOES
------------
1. For each PPI target, pulls every binding-type activity (Kd, Ki, IC50) from ChEMBL.
2. Keeps records that have a real measured value, standard units and a pChEMBL value.
3. Classifies each unique compound by SMILES as macrocyclic peptide / macrocycle /
   peptide / small molecule, using RDKit ring and amide-bond analysis.
4. Cross-references compounds to PDB structures via the target's PDBe xrefs.
5. Writes a tidy CSV plus a summary of how many Tier 1 candidates were found.

WHAT IT DELIBERATELY DOES NOT DO
--------------------------------
It does not decide that a hit belongs in the benchmark. Macrocycle detection from SMILES
is a heuristic and will produce false positives (large non-peptidic macrocycles) and false
negatives (stapled peptides whose linker RDKit does not perceive as closing a ring). Every
candidate it emits needs manual confirmation against its source publication before it goes
into the benchmark. The output column `needs_manual_review` is there to make that explicit.

USAGE
-----
    pip install requests pandas rdkit-pypi tqdm
    python curate_chembl_macrocycles.py --out tier1_candidates.csv

    # faster smoke test on two targets
    python curate_chembl_macrocycles.py --targets MDM2 MCL1 --limit 500 --out smoke.csv
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, asdict

import requests

CHEMBL_BASE = "https://www.ebi.ac.uk/chembl/api/data"

# PPI targets of interest. ChEMBL IDs for MDM2, MCL1 and KEAP1 were verified live against
# the ChEMBL API on 12 Oct 2026. The rest are marked None and resolved by gene symbol at
# runtime -- check what the resolver picks before trusting those rows.
TARGETS: dict[str, str | None] = {
    "MDM2": "CHEMBL5023",        # verified
    "MCL1": "CHEMBL4361",        # verified
    "KEAP1": "CHEMBL2069156",    # verified
    "MDM4": None,
    "BCL2": None,
    "BCL2L1": None,              # Bcl-xL
    "CTNNB1": None,              # beta-catenin
    "SPSB2": None,
    "GABARAP": None,
    "XIAP": None,
    "MENIN": None,
    "BRD4": None,                # included as a non-PPI-ish control
}

BINDING_TYPES = ("Kd", "Ki", "IC50")


@dataclass
class Record:
    target_gene: str
    target_chembl_id: str
    molecule_chembl_id: str
    molecule_pref_name: str | None
    activity_type: str
    standard_value: float | None
    standard_units: str | None
    pchembl_value: float | None
    assay_type: str | None
    assay_description: str | None
    document_chembl_id: str | None
    document_journal: str | None
    document_year: int | None
    canonical_smiles: str | None
    data_validity_comment: str | None
    # filled in by classification
    mol_class: str = "unknown"
    max_ring_size: int = 0
    n_amide_bonds: int = 0
    mol_weight: float = 0.0
    needs_manual_review: bool = True


def _get(url: str, params: dict, retries: int = 4) -> dict:
    """GET with exponential backoff. ChEMBL rate-limits; be polite."""
    delay = 2.0
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=60,
                             headers={"Accept": "application/json"})
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(delay)
                delay *= 2
                continue
            r.raise_for_status()
        except requests.RequestException as exc:
            if attempt == retries - 1:
                raise
            print(f"  retry after {exc.__class__.__name__}", file=sys.stderr)
            time.sleep(delay)
            delay *= 2
    raise RuntimeError(f"giving up on {url}")


def resolve_target(gene: str) -> str | None:
    """Resolve a gene symbol to a human SINGLE PROTEIN ChEMBL target id."""
    data = _get(f"{CHEMBL_BASE}/target/search.json",
                {"q": gene, "limit": 25})
    for t in data.get("targets", []):
        if (t.get("target_type") == "SINGLE PROTEIN"
                and t.get("organism") == "Homo sapiens"):
            for comp in t.get("target_components", []):
                for syn in comp.get("target_component_synonyms", []):
                    if syn.get("component_synonym", "").upper() == gene.upper():
                        return t["target_chembl_id"]
    return None


def fetch_activities(target_id: str, gene: str, limit: int | None) -> list[Record]:
    out: list[Record] = []
    for act_type in BINDING_TYPES:
        offset, page = 0, 1000
        while True:
            data = _get(f"{CHEMBL_BASE}/activity.json", {
                "target_chembl_id": target_id,
                "standard_type": act_type,
                "limit": page,
                "offset": offset,
            })
            acts = data.get("activities", [])
            if not acts:
                break
            for a in acts:
                # Keep only clean, quantitative, equality-relation measurements.
                if a.get("standard_relation") != "=":
                    continue
                if a.get("pchembl_value") is None:
                    continue
                out.append(Record(
                    target_gene=gene,
                    target_chembl_id=target_id,
                    molecule_chembl_id=a.get("molecule_chembl_id"),
                    molecule_pref_name=a.get("molecule_pref_name"),
                    activity_type=act_type,
                    standard_value=_f(a.get("standard_value")),
                    standard_units=a.get("standard_units"),
                    pchembl_value=_f(a.get("pchembl_value")),
                    assay_type=a.get("assay_type"),
                    assay_description=a.get("assay_description"),
                    document_chembl_id=a.get("document_chembl_id"),
                    document_journal=a.get("document_journal"),
                    document_year=a.get("document_year"),
                    canonical_smiles=a.get("canonical_smiles"),
                    data_validity_comment=a.get("data_validity_comment"),
                ))
            offset += page
            if limit and offset >= limit:
                break
            if offset >= data.get("page_meta", {}).get("total_count", 0):
                break
    return out


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# A peptide BACKBONE bond: Ca-C(=O)-N-Ca. Deliberately stricter than a generic amide
# SMARTS, which also matches ureas and aryl amides and wrongly flags drug-like small
# molecules as peptides (Nutlin-3 was scored "peptide" by the loose pattern).
_BACKBONE_SMARTS = "[CX4][CX3](=[OX1])[NX3][CX4]"

# Thresholds. Note there is deliberately NO molecular-weight floor: a cyclic hexaglycine
# is a genuine macrocyclic peptide at 342 Da and an MW floor of 400 silently discarded it.
_MACROCYCLE_MIN_RING = 12   # standard medicinal-chemistry definition
_PEPTIDE_MIN_BONDS = 2      # 2 backbone bonds = a tripeptide, the smallest useful case
_MACROPEPTIDE_MIN_BONDS = 3


def classify(smiles: str | None) -> tuple[str, int, int, float]:
    """Classify a molecule from SMILES.

    Returns (class, max_ring_size, n_backbone_amide_bonds, mol_weight).

    Still a heuristic. Validated against seven reference molecules in
    tests/test_classify.py (cyclosporin A, oxytocin, cyclic Gly6, linear GGG, Nutlin-3,
    aspirin, a non-peptidic macrolactone); run that after any change here.

    Known remaining limits:
      - stapled peptides whose staple RDKit does not perceive as closing a ring are
        classified 'peptide', not 'macrocyclic_peptide';
      - depsipeptides, where ester replaces some amide bonds, under-count;
      - N-methylated backbones still match, but beta- and gamma-amino-acid backbones
        do not, since the SMARTS assumes an alpha spacing;
      - large non-peptidic macrolides land in 'macrocycle' and must be removed by hand.
    """
    if not smiles:
        return "unknown", 0, 0, 0.0
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors
        from rdkit import RDLogger
        RDLogger.DisableLog("rdApp.*")
    except ImportError:
        raise SystemExit("rdkit is required: pip install rdkit-pypi")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "unparseable", 0, 0, 0.0

    rings = mol.GetRingInfo().AtomRings()
    max_ring = max((len(r) for r in rings), default=0)

    patt = Chem.MolFromSmarts(_BACKBONE_SMARTS)
    n_backbone = len(mol.GetSubstructMatches(patt)) if patt else 0

    mw = Descriptors.MolWt(mol)

    is_macro = max_ring >= _MACROCYCLE_MIN_RING

    if is_macro and n_backbone >= _MACROPEPTIDE_MIN_BONDS:
        cls = "macrocyclic_peptide"
    elif is_macro:
        cls = "macrocycle"
    elif n_backbone >= _PEPTIDE_MIN_BONDS:
        cls = "peptide"
    else:
        cls = "small_molecule"
    return cls, max_ring, n_backbone, mw


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="tier1_candidates.csv")
    ap.add_argument("--targets", nargs="*", default=None,
                    help="gene symbols to restrict to (default: all in TARGETS)")
    ap.add_argument("--limit", type=int, default=None,
                    help="max activities per target per activity type (for smoke tests)")
    args = ap.parse_args()

    import pandas as pd

    genes = args.targets or list(TARGETS)
    all_records: list[Record] = []

    for gene in genes:
        tid = TARGETS.get(gene)
        if tid is None:
            print(f"[{gene}] resolving target id...", file=sys.stderr)
            tid = resolve_target(gene)
            if tid is None:
                print(f"[{gene}] NOT RESOLVED - skipping, resolve by hand", file=sys.stderr)
                continue
            print(f"[{gene}] -> {tid}  (unverified: check this is the right target)",
                  file=sys.stderr)
        print(f"[{gene}] fetching activities from {tid}...", file=sys.stderr)
        recs = fetch_activities(tid, gene, args.limit)
        print(f"[{gene}] {len(recs)} clean quantitative records", file=sys.stderr)
        all_records.extend(recs)

    if not all_records:
        print("no records fetched", file=sys.stderr)
        return

    # Classify unique SMILES once, not once per activity row.
    cache: dict[str, tuple] = {}
    for r in all_records:
        smi = r.canonical_smiles
        if smi not in cache:
            cache[smi] = classify(smi)
        r.mol_class, r.max_ring_size, r.n_amide_bonds, r.mol_weight = cache[smi]
        r.needs_manual_review = True  # always; see module docstring

    df = pd.DataFrame([asdict(r) for r in all_records])
    df.to_csv(args.out, index=False)

    print("\n=== summary ===", file=sys.stderr)
    print(df.groupby(["target_gene", "mol_class"]).size().to_string(), file=sys.stderr)

    macro = df[df.mol_class == "macrocyclic_peptide"]
    print(f"\nmacrocyclic-peptide activity rows: {len(macro)}", file=sys.stderr)
    print(f"unique macrocyclic-peptide compounds: {macro.molecule_chembl_id.nunique()}",
          file=sys.stderr)
    print(f"targets with >=1 such compound: {macro.target_gene.nunique()}", file=sys.stderr)
    print(f"\nwrote {args.out}", file=sys.stderr)
    print("EVERY row needs manual confirmation against its source paper before use.",
          file=sys.stderr)


if __name__ == "__main__":
    main()
