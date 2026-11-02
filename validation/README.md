# Validation results

Out-of-fold predictions of the two tests in the paper: whole isomer families held out (new-isomer test) and whole solvents held out (new-solvent test), five folds each, three training seeds (20261007, 20261008, 20261009). All models were trained with the structures as used in the paper (`training/inputs/molecules_as_trained.csv`).

| File | Contents |
|---|---|
| `oof_predictions_all_models.csv.gz` | Predicted ln γ∞ for all 68,763 records: `newiso_DMPNN_*` (D-MPNN trained on activity coefficients and selectivities, λ = 1), `newiso_DMPNNact_*` (D-MPNN trained on activity coefficients only, λ = 0), `newiso_GHGNN_*`, `newiso_ECFP` (ECFP + LightGBM), `newsolv_DMPNN_*` and `newsolv_GHGNN_*`, one column per seed, to ten significant digits. GH-GNN columns are empty for the 225 records it cannot represent. The figures are drawn from this file |
| `oof_predictions_ln_gamma.csv` | The same predictions of D-MPNN (λ = 1) and GH-GNN on the common domain of 68,538 records, as used for the tables below |
| `pairs_common_domain.csv` | The 31,496 isomer pairs of the common domain. `Isomer_Relation` uses the labels of `data/`, `relation_en` those of the figures and the web tool |
| `calibration_by_predicted_S.csv`, `hit_rate_by_isomer_type.csv`, `model_summary_by_regime.csv` | Hit rates by predicted selectivity and by isomer type, and summary metrics by test and model (made by `make_calibration.py`) |
| `test_solvent_similarity_solvent_out.csv` | Highest Tanimoto similarity of each test solvent to the training solvents of its fold |
| `runs/` | Records of every training run: five-fold reports, per-fold manifests, training histories and summaries |
| `ecfp_lightgbm/` | Code, report and out-of-fold predictions of ECFP + LightGBM (one run, deterministic). Rerunning `ecfp_baseline.py` on the tables from `training/prepare_data.py` reproduces the predictions exactly |
| `cosmosac/` | COSMO-SAC predictions for isomer pairs (Supporting Information, S4) |

The predicted selectivity of a pair is exp|ln γ∞(A) − ln γ∞(B)| from the predictions of its two records, and model values in the paper are means over the three seeds.

## COSMO-SAC

`cosmosac/code/` computes γ∞ with COSMO-SAC 2010 (three sigma profiles) and the 2014 dispersion correction (`cosmosac_model.py`), following the open-source implementation of NIST (https://github.com/usnistgov/COSMOSAC), and compares the resulting selectivities with experiment and D-MPNN (`run_cosmosac.py`). Sigma profiles of the University of Delaware database are downloaded from the NIST repository at run time and are not redistributed here under its licence. The calculation was run before repeated listings were merged, on the out-of-fold D-MPNN predictions of that dataset, which are not included here.

`cosmosac/outputs/cosmosac_pair_predictions.csv` holds the 2,913 pairs that could be computed, with the measured, D-MPNN and COSMO-SAC values of ln S∞ (`D_experimental`, `D_predicted`, `D_cosmo`) and the combinatorial, residual and dispersion parts of `D_cosmo`. `Pair_ID` links 2,874 of the pairs to `data/isomer_pairs.csv`, and the other 39 pairs are not in the final dataset. `cosmosac_cross_solvent_differences.csv` gives the change of ln S∞ between two solvents of one study, and `summary_scores.csv` the mean absolute errors over pairs and averaged over families, studies and isomer types. Record IDs, isomer types and sources use the labels of `data/`. The paper uses the 2,769 pairs with complete dispersion parameters, of which 2,730 are in the final dataset.
