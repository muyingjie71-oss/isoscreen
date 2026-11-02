"""Locations shared by the build scripts of the web tool."""
import sys
from pathlib import Path

BUILD = Path(__file__).resolve().parent
REPO = BUILD.parents[1]
DATA = REPO / 'data'
VALIDATION = REPO / 'validation'
MODELS = REPO / 'models'
TRAINING = REPO / 'training'
TRAINING_DATA = TRAINING / 'data' / 'new_isomer_test'   # written by training/prepare_data.py
LIBRARY = REPO / 'webtool' / 'solvent_library'
SITE = REPO / 'webtool' / 'site'
WORK = BUILD / 'work'                                     # intermediate files, not part of the site
PARAMS = WORK / 'params'
ASSETS = WORK / 'assets'

for p in (TRAINING / 'dmpnn' / 'shared', TRAINING / 'dmpnn', TRAINING / 'ghgnn', TRAINING / 'ghgnn' / 'ghgnn', BUILD):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
