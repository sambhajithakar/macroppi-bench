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

## Additions in v1.3 (13 Nov 2026): closure-share law, population test, identification bounds, latent test
All numbers in the accompanying manuscript's Sections 4.5 and 4.6 regenerate from these scripts and results; each experiment's rule was written to `docs/PREREGISTRATION_theory_12Nov2026.md` (Amendments 1 to 8) before it was run. This plan was not lodged in an external registry, so it is internally pre-specified only. Simulations are labelled as such.

| Script | What it does | Output (`results/`) | Seed |
|---|---|---|---|
| `phase4_theory.py`, `phase5_theory_v2_schedule.py` | Closure-share law v1, v2 (both failed their rules), open-chain share, closure-aware schedule (not supported) | `phase4_theory.csv`, `phase5_*.csv` | see script |
| `phase6_theory_v3.py` | Jensen-corrected law v3 on new ring sizes | `phase6_theory_v3.csv` | see script |
| `phase7_multi_restraint.py` | v4, multi-restraint law (failed its rules) | `phase7_multi_restraint.csv`, `phase7_real_descriptive.csv` | 20261205 |
| `phase7b_multi_restraint_v5.py` | v5, native-length correction, new sizes | `phase7b_multi_restraint_v5.csv` | 20261207 |
| `phase8_bounds.py` | Identification bounds for the 38 untested designs and decision-level utility | `phase8_bounds.csv`, `phase8_utility.csv` | 20261206 |
| `phase9_pdb_population.py` | 89 head-to-tail macrocycles from the 1 Jan 2024 wwPDB snapshot | `phase9_pdb_population.csv` | 20261208 |
| `phase10_latent_closure.py` | Closure share through the decoder of the published CP-Composer autoencoder (CPU); needs github.com/jdp22/CP-Composer_final (code and `checkpoints/autoencoder.pth`) and `torch_scatter_shim.py` | `phase10_latent_closure.csv` | 20261209 |
| `figure5_law_bounds.py` | Figures 4 and 5 | `figures/figure4_law.*`, `figure5_bounds.*` | none |

`phase10_latent_closure.py` uses a third-party model that is not redistributed here; the autoencoder is the authors' published one and nothing was trained by us.
