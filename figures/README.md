# Code for Figures 1–6

Draws the six figures of the paper from the training tables and the out-of-fold predictions in `predictions/`.

## Running

```bash
python ../training/prepare_data.py              # once, writes the training tables read here
pip install -r requirements.txt
python -m playwright install chromium     # only Figure 1 needs it, to convert SVG to PDF and PNG
python run_all.py                          # all six figures, written to out/
python run_all.py 2 6                      # only Figures 2 and 6
```

Each figure can also be drawn alone, for example `python fig3.py . out`. Every figure is written as SVG, as PDF with Type 42 fonts (editable text) and as PNG at 600 dpi. The font is Arial, replaced by Liberation Sans where Arial is not installed, which leaves the layout essentially unchanged.

## Files

| File or folder | Contents |
|---|---|
| `fig1.py` – `fig6.py` | One script per figure |
| `run_all.py` | Runs all scripts in turn |
| `scripts/load_all.py` | Reads the training tables (`training/data/new_isomer_test/`) and the predictions for Figures 2–6 and joins measured values, isomer pairs and the out-of-fold predictions of every model in one table |
| `predictions/oof_predictions_all_models.csv.gz` | Out-of-fold predictions of the new-isomer test for all models: D-MPNN (`A_JOINT_*`), D-MPNN trained on activity coefficients only (`A_ABS_*`), GH-GNN and ECFP + LightGBM |
| `predictions/oof_ln_gamma_both_tests.csv.gz` | Predictions of both tests in one table. Figure 4 takes the new-solvent predictions (`newsolv_*` columns) from it |
| `outputs/` | Summary tables used by Figures 2 and 4: metrics of every training run and deviations between cross-study replicates |
| `plot_data/fig1/` | Inputs of Figure 1: example isomer pairs, boiling points and the number of pairs in each resolution range |

## Relation to the published figures

The submitted figures were produced by these scripts, after which spacing and panel heights were adjusted in Illustrator. All data, numbers, text and label styles in the scripts match the submitted figures.

## Numbers printed while running

The scripts print the key numbers of the paper for comparison with the text.

- Figure 2: R² of ln γ∞ is 0.92 for D-MPNN and GH-GNN and 0.81 for ECFP + LightGBM. R² of the selectivity is 0.42, 0.34 and −1.65.
- Figure 3: share of predicted selectivities beyond 1.12 is 38% for D-MPNN, 47% for GH-GNN and 79% for ECFP + LightGBM.
- Figure 4: correlation r between the errors of the two isomers is 0.86 for experiments, 0.54 for D-MPNN and 0.55 for GH-GNN in the new-isomer test, and 0.91 and 0.92 in the new-solvent test. Resolution is 1.122 and 1.145 for new isomers and 1.053 and 1.062 for new solvents.
- Figure 6: hit rate above a predicted selectivity of 1.65 is 84% for D-MPNN and 77% for GH-GNN. The median number of solvents tested until the best one, over 48 pairs, is 20.5 for D-MPNN, 8.5 for GH-GNN and 31 in random order, and 52 and 27 for 1-hexene and cyclohexane.

## Notes

- Before Figure 6c is computed, the A/B order of each isomer pair is unified across studies and the selectivities are averaged over studies. The three pairs with cyclohexane from Arumugam et al. (2020) are left out of the ranking (Supporting Information, S1.4).
- The figures use the structures as used for training, so every number matches the paper.
- Tested with Python 3.12, numpy 2.5, pandas 3.0, matplotlib 3.11 and RDKit 2026.09. Rerunning reproduces the submitted figures exactly, apart from internal SVG identifiers and the adjustments made in Illustrator.
