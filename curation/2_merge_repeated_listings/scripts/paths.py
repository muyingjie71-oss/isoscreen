"""Locations used by this step. Defaults are relative to this folder and can be changed with environment variables.

EXTRACTED_TABLES_DIR   tables from step 1: deduplicated_experiments.csv.gz, reviewed_model_task2.csv.gz, family_fold_manifest.json
EARLIER_TRAINING_DATA  training tables built from step 1: task1.csv.gz, molecules.csv, split_manifest.json and
                       molecule_graphs.npz
MERGE_WORK             intermediate files
MERGE_OUT              output tables
TRAINING_TABLES_DIR    folder whose data/ receives the training tables
"""
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
def _d(env, default):
    return str(Path(os.environ.get(env, default)).resolve()) + '/'
X = _d('EXTRACTED_TABLES_DIR', ROOT / 'inputs' / 'extracted_tables')
AUDIT = X
LA = _d('EARLIER_TRAINING_DATA', ROOT / 'inputs' / 'earlier_training_data')
W = _d('MERGE_WORK', ROOT / 'work')
O = _d('MERGE_OUT', ROOT / 'outputs')
P = _d('TRAINING_TABLES_DIR', ROOT / 'outputs' / 'training_tables')
LIB = str(ROOT / 'lib')
for d in (W, O, P + 'data/'):
    os.makedirs(d, exist_ok=True)
