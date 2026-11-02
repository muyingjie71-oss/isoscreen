# Dataset

`isomer_selectivity_dataset.xlsx` is the readable version: one workbook with a README sheet that defines every column, structures next to each value and references written out. The CSV files below hold the same data.

| File | Contents |
|---|---|
| `activity_coefficients.csv` | 68,763 infinite-dilution activity coefficients, one row per measurement |
| `isomer_pairs.csv` | 31,541 isomer pairs measured in the same study, solvent and temperature |
| `molecules.csv` | 981 solutes and solvents: Mol_ID, InChIKey, SMILES |
| `solvents.csv` | 771 solvents, with cation and anion of the ionic liquids |
| `studies.csv` | 485 original studies: identifier, DOI and reference |
| `folds_new_isomer_test.json` | Folds of the test that holds out whole isomer families |
| `folds_new_solvent_test.json`, `folds_new_solvent_test_summary.csv` | Folds of the test that holds out whole solvents |
| `cross_study_replicates.csv` | The same isomer pair, solvent and temperature measured in two independent studies, used for the experimental resolution |
| `structure_corrections.csv` | Eight solvent structures corrected after the analyses |

## activity_coefficients.csv

| Column | Meaning |
|---|---|
| `canonical_record_id` | Record ID: `REC_` followed by a hash of the record |
| `Family_ID`, `Member_ID` | Isomer family (molecular formula) and isomer |
| `Study_ID` | Original study: `DOI:`, `BIB:[journal, volume, first page]`, `TITLE:` key or `REF:` key. Empty for 6,718 records listed only in a compilation |
| `Source` | Database or compilation the record was taken from |
| `Original_DOI` | DOI of the original study, where known |
| `Solute`, `Solvent` | Names as given in the source |
| `T_K`, `gamma_inf`, `ln_gamma_inf` | Temperature in K, γ∞ and ln γ∞ (`ln_gamma_inf` is the source value) |
| `Solute_Mol_ID`, `Solvent_Mol_ID`, `solute_mol_index`, `solvent_mol_index` | Links to `molecules.csv` |
| `Effective_Conditions`, `Reported_Condition_Declaration` | Normalized conditions used for pairing: pressure, phase, standard state, method, other |
| `Reported_Pressure_kPa`, `Reported_Phase`, `Reported_Standard_State`, `Reported_Method`, `Reported_Other_Conditions` | Conditions as reported by the source |
| `Original_Source_File`, `Original_Source_Location` | File in the source archive and location in it (sheet and cell, page, or JSON path). The files of individually collected papers are given English names |
| `Study_Context_IDs` | All study identifiers under which the measurement was listed |
| `Study_ID_Before_Merge` | Study identifier before identifiers of the same paper were merged |
| `Special_Record_Flag` | Kept but flagged: converted from another property, solvent in a liquid-crystal phase, or solute labelled gaseous |
| `Merged_Copy_Count` | Number of listings of the same measurement merged into this record |
| `Fold_ID` | Fold of the isomer family in the new-isomer test |

## isomer_pairs.csv

`Pair_ID` (`PAIR_` and a hash), `MatchedGroup_ID` (`GROUP_` and a hash; the pairs of one study, solvent, temperature and conditions), `Family_ID`, `Study_ID`, `Solvent_Mol_ID`, `Solvent`, `T_K`, the matched conditions, the two isomers (`Isomer_A`, `Isomer_B`) with their record IDs, γ∞ and ln γ∞, `D_AB` = ln γ∞(A) − ln γ∞(B), so that the selectivity is exp|D_AB|, `Isomer_Relation` (type of isomerism), `Fold_ID` and `n_study_contexts`.

## Structures

`molecules.csv` and `solvents.csv` hold the corrected structures. Mol_IDs were assigned from the InChIKey at curation and are kept, so for one corrected solvent the Mol_ID differs from its current InChIKey. The models in the paper used the structures before correction (`training/inputs/molecules_as_trained.csv`).
