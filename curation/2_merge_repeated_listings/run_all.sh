#!/usr/bin/env bash
# Merge repeated listings of one measurement and unify study identifiers.
# Measured values, isomer families and folds are not changed.
# Inputs are in inputs/ (see scripts/paths.py). Needs python3 with pandas and numpy, and RDKit for the last step.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"; S="$HERE/scripts"
python3 -I "$S/find_repeated_listings.py"   # study links, identifier merges, clusters of repeated listings
python3 -I "$S/build_records.py"            # kept records and pairs; compilation values in conflict are held back
python3 -I "$S/replicates.py"               # cross-study replicates and reported uncertainties
python3 -I "$S/export_tables.py"            # record decisions, identifier merges, study links, counts
python3 -I "$S/verify.py"                   # checks against the extracted tables, all must pass
PKG="${TRAINING_TABLES_DIR:-$HERE/outputs/training_tables}"; LA="${EARLIER_TRAINING_DATA:-$HERE/inputs/earlier_training_data}"; mkdir -p "$PKG/data"
for f in molecules.csv split_manifest.json molecule_graphs.npz; do cp "$LA/$f" "$PKG/data/"; done
python3 -I "$S/make_training_tables.py"
python3 -I "$S/rename_record_ids.py"          # record IDs get the prefix REC_ in the written tables
cp "${MERGE_WORK:-$HERE/work}/replicates.csv" "${MERGE_OUT:-$HERE/outputs}/cross_study_replicates.csv"
