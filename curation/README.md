# Data curation

The dataset was built in three steps. Steps 2 and 3 can be rerun from this repository. Step 1 needs the source files, which are not redistributed here.

| Folder | Step |
|---|---|
| `1_extract_sources/` | Extraction and checks for each source, then merging of the sources, deduplication within studies, model eligibility and pairing |
| `2_merge_repeated_listings/` | Merging of listings of the same measurement across databases and unification of study identifiers |
| `3_public_dataset/` | The public files in `data/` and the readable workbook |
| `bibliography_all_listings.csv` | Every reference string under which a study is listed in the sources |

## 1. Extraction and checks

One folder per source holds its entry points: `01_MASTER` (ThermoML and ILThermo only) reads the raw records, `02_IDAC` builds the table of infinite-dilution activity coefficients and `03_ISOMERS` the isomer families and pairs. `08_Dataset_Curation` merges the seven sources, removes repeated values within one study (values within 0.5% across the group), sets model eligibility and forms the pairs. The shared code is in `_pipeline/`, and `_pipeline/assets/` holds the reviewed decisions: the structure assigned to every solute and solvent name, the audited structure registries of the sources and the fixed assignment of isomer families to folds.

Each entry point is run as `python 06_Sherman/02_IDAC.py --work-dir <folder>` from a folder that also contains the source files under `<source>/00_Source/` and writes its tables to the work folder. The source files are the NIST ThermoML archive (version 2020-09-30), NIST ILThermo (activity data, queried on 6 October 2026), the compilations of Brouwer et al. (2021), Tan et al. (2022), Li et al. (2022) and Sherman et al. (1996), and individual papers and theses. They are available from their publishers and were not redistributed. The output of this step, 84,147 checked measurements of which 73,243 are eligible for modelling, is the input of step 2.

## 2. Repeated listings and study identifiers

```bash
cd curation/2_merge_repeated_listings
bash run_all.sh
```

The inputs from step 1 are in `inputs/`. The step changes only which records are kept and how studies are identified. No measured value is changed, and isomer families and folds stay as they are.

1. **Study identifiers.** One paper can be listed under a DOI, a journal–volume–page key or a title key in different databases. Such identifiers were merged in 16 cases checked by hand, and only when the title or the journal, volume and first page agree. Two papers that report the same values are not merged.
2. **Repeated listings of one measurement.** Two records are the same measurement when they have the same solute, solvent and temperature (within 0.01 K), ln γ∞ within 0.005, and a likely common origin. A common origin is assumed when both come from the same study with compatible conditions (an empty field matches any value, and pressures within 0.5 kPa are equal), when the two studies share at least three such identical values, when one record comes from a compilation that reprints earlier data (Kojima et al. 1997, Sherman et al. 1996), or when the original study of a record is unknown. Different primary studies that share only one or two identical values are kept as independent measurements.
3. **Record kept.** Known studies are preferred over unknown ones, primary studies over compilations, earlier over later publication, DOI over other identifiers, and ThermoML over ILThermo over compilations. The kept record keeps its original value, and values are not averaged.
4. **Transcription conflicts.** In 44 cases, a compilation and ThermoML or ILThermo gave values for the same paper and conditions that differ by more than 0.5%. The database value was kept and the compilation value held back.
5. **Pairing.** Pairs are formed as in step 1: same original study, solvent and temperature, and compatible conditions. Records with ambiguous conditions, repeated records of one isomer under the same conditions and records of unknown study are not paired.

| | Before | After |
|---|---:|---:|
| Records eligible for modelling | 73,243 | 68,763 |
| Isomer pairs | 33,330 | 31,549, and 31,541 in the training tables |
| Study identifiers | 502 | 485 |

`outputs/record_decisions.csv.gz` gives the decision for each of the 73,243 records: kept, merged into another record (`copy_cluster` names the kept record) or held back. `outputs/study_identifier_merges.csv` lists the 16 merged identifiers, `outputs/study_links.csv` the 91 pairs of studies that share identical values with the decision taken, and `outputs/counts.json` the counts above. The training tables (`outputs/training_tables/`) drop eight further pairs whose conditions conflict once pressures are normalized or whose two temperatures differ slightly. `scripts/verify.py` checks the result against the input tables.

Record IDs from step 1 receive the prefix `REC_` in the tables of this step (`scripts/rename_record_ids.py`), and pair and matched-group IDs the prefixes `PAIR_` and `GROUP_`. The hash part of each ID is kept, so every record can be traced back to the tables in `inputs/`. Rerunning this step reproduces the training tables of the paper apart from these prefixes and the source file paths, which are given relative to the source archive. The cross-study replicates agree to within 2 × 10⁻¹⁶.

## 3. Public files

```bash
python curation/3_public_dataset/make_public_dataset.py
python curation/3_public_dataset/make_study_list.py
python curation/3_public_dataset/make_workbook.py
```

`make_public_dataset.py` writes `data/` from the training tables of step 2 with English file names, column names and labels, and with the corrected structures of eight solvents (`data/structure_corrections.csv`). The solvent table and the folds of the new-solvent test are read from `3_public_dataset/inputs/`. `make_study_list.py` writes the reference of each study, and `make_workbook.py` the readable workbook.
