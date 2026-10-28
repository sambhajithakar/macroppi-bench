# MacroPPI-Bench v1.2 — data and code for *Generative AI for Macrocyclic Peptides at Cancer and Other Interfaces: One Refuted Mechanism and One Inconclusive Filter Test*

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23370597.svg)](https://doi.org/10.5281/zenodo.23370597)

This archive contains everything needed to reproduce every number, table and figure in the
manuscript. It is pure Python 3 (NumPy, Matplotlib, pandas). No GPU, no force-field package and
no licensed software are required; the full set of analyses runs in minutes on a laptop CPU.

## Contents

```
data/
  tier1_designs.csv                   MacroPPI-Bench Tier 1 seed: 10 designs with
                                      published binding values, with provenance per row.
                                      (The 65 synthesised designs / 26 binders / 6 targets
                                      in the manuscript are tallied from the three source
                                      papers; per-design records exist for one campaign.)
  tier1_rfpeptides_per_design.csv      All 77 RFpeptides designs joined from the published
                                      Supporting Information to synthesis and binding
                                      outcome: four in-silico filter scores, synthesis
                                      status, censoring flag, screen and characterised
                                      binder flags, affinity, purity, sequence.
scripts/
  phase1_closure.py                   Experiment A — closure-share diagnostic on head-to-tail
                                      macrocycles, with the force-field scaling and
                                      term-removal sensitivity sweeps.
  phase1_sensitivity.py               Experiment A sensitivity table, regenerated (see below).
  phase1_chemistries.py               Experiment B — the same diagnostic for disulfide and
                                      bicycle topologies.
  extract_rfpeptides_si.py            Rebuilds tier1_rfpeptides_per_design.csv from the
                                      published Supporting Information PDF. Aborts if the
                                      extracted counts disagree with the counts the source
                                      paper states independently of its tables.
  phase2_retrospective.py             Experiment C — within-target pooled AUROC for the four
                                      published filters, stratified bootstrap and label
                                      permutation null.
  phase3_power.py                     Simulated power of the retrospective test (seed 20261027);
                                      writes results/phase3_power.csv. Simulation, not measured data.
  figure4_retrospective.py            Figure 3 of the manuscript.
  curate_chembl_macrocycles.py        Tier 2 harvester (not used for any manuscript result).
tests/
  test_phase1_gradients.py            Analytic-gradient checks for the Ca potential.
  test_classify.py                    Macrocycle classifier reference cases.
results/                              Every output file the manuscript cites, as generated.
figures/                              The three manuscript figures at 300 dpi.
```

## Reproducing the manuscript

```
# Section 4.1 — closure-share diagnostic, head-to-tail (downloads structures on first run)
python3 scripts/phase1_closure.py --out results --seed 0

# Section 4.2 — disulfide and bicycle topologies
python3 scripts/phase1_chemistries.py --pdb-dir <directory of .ent files> --out results

# Section 4.3 — published filters, bootstrap and permutation null
python3 scripts/phase2_retrospective.py --designs data/tier1_rfpeptides_per_design.csv \
    --outdir results --n-boot 10000 --n-perm 10000 --seed 20261013

# Figure 3
python3 scripts/figure4_retrospective.py --result results/experiment_c_retrospective.json \
    --designs data/tier1_rfpeptides_per_design.csv --out results/figure4_retrospective.png
```

`phase1_closure.py` downloads the structures it needs into `--cache` (default `pdb_cache/`); if
RCSB is unreachable, the same accessions can be fetched from the wwPDB AWS mirror and passed with
`--local-dir`. `phase1_chemistries.py` reads the disulfide structures (1SFI, 1JBL, 1NB1, 1K48)
from a directory of `.ent` files obtained the same way. Each script's docstring carries the
command line exactly as it was run.

All stochastic steps are seeded. Rerun on 27 Oct 2026 (NumPy 2.5.3, structures from the wwPDB AWS snapshot of 1 Jan 2026), `phase1_closure.py --seed 0 --samples 128`, `phase1_chemistries.py` and `phase2_retrospective.py` reproduce the deposited outputs byte for byte. The Experiment A sensitivity table (`results/experiment_a_sensitivity.csv`) was produced by an earlier script that was not deposited; `phase1_sensitivity.py` regenerates it (`results/experiment_a_sensitivity_rerun.csv`, agreeing within 0.002 in every condition).

`extract_rfpeptides_si.py` additionally requires the Supporting Information PDF of Rettie et al.,
*Nat. Chem. Biol.* (2025), which is not redistributed here — the script regenerates the joined
table from the publisher's or PubMed Central's copy.

## Structural data

Coordinates are taken from the wwPDB under accession codes 9CDT, 9CDU, 9CDV, 9HGC, 9HGD, 9CDZ,
1SFI, 1JBL, 1NB1 and 1K48, and are not redistributed here.

## Licence

Code is released under the MIT Licence (LICENSE-CODE). Data files curated by us — the two CSVs
and the result files — are released under CC BY 4.0 (LICENSE-DATA). Third-party data retain the
licences of their sources.

## Citing

Archived at https://doi.org/10.5281/zenodo.23370597 (concept DOI; always resolves to the
newest version). Machine-readable metadata is in CITATION.cff.
