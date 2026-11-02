# IsoScreen

Data, code and web tool of the paper *Resolution limits of predicted selectivities in isomer separation* (submitted to AIChE Journal).

**Web tool:** https://muyingjie71-oss.github.io/isoscreen/  
**Archive:** [doi:10.5281/zenodo.23877328](https://doi.org/10.5281/zenodo.23877328)

The paper compares machine-learning predictions of isomer selectivities with experiments on 31,541 isomer pairs. Models resolve the selectivities of isomer pairs they have seen in training nearly as finely as experiments, but resolve those of new isomer families only coarsely. IsoScreen applies the resulting rules to solvent screening: for any pair of constitutional isomers and a temperature between 273 and 423 K, it ranks 6,911 solvents by predicted selectivity in the browser and labels each prediction by how far it can be trusted.

## Contents

| Folder | Contents |
|---|---|
| `data/` | The dataset: 68,763 infinite-dilution activity coefficients of 336 solutes in 771 solvents from 485 studies, 31,541 isomer pairs, molecules, solvents, studies and test folds. `isomer_selectivity_dataset.xlsx` is a readable version with a README sheet |
| `curation/` | How the dataset was built from the sources: extraction and checks per source, merging of repeated listings, and the public files |
| `training/` | Code of D-MPNN and GH-GNN for both tests and for the production models |
| `validation/` | Out-of-fold predictions of all models, records of every training run, ECFP + LightGBM and COSMO-SAC |
| `figures/` | Code for Figures 1–6 |
| `models/` | Production models of the web tool |
| `webtool/` | The web tool (`site/`), the code that builds it from the models, its solvent library and its tests |

## Reproducing the results

1. `python training/prepare_data.py` builds the training tables from `data/` and checks them against those of the paper.
2. `training/README.md` gives the commands for both tests and for the production models.
3. `python figures/run_all.py` draws Figures 1–6 from the out-of-fold predictions.
4. `webtool/README.md` describes how the web tool is built from the production models.

## Notes on the dataset

The structures of eight solvents were corrected after the analyses (`data/structure_corrections.csv`). Measured values, records, pairs and folds are not affected. The models in the paper were trained with the original structures, which `training/inputs/molecules_as_trained.csv` keeps, so that the training tables and figures reproduce the paper exactly.

## Licence

Code: MIT (`LICENSE`). Data, models and solvent library: CC BY 4.0, except the solvent identities from the CHEM21 guide, which remain under CC BY-NC 3.0. Third-party code: GH-GNN (MIT), RDKit (BSD 3-Clause) and ONNX Runtime (MIT).
