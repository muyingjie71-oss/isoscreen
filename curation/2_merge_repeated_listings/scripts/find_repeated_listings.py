"""Load the extracted tables, find study pairs that share identical values, then unify study aliases
and cluster measurement copies (rules in identity.py). Writes pickles to MERGE_WORK."""
import sys, os, runpy
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from paths import W
from load import load_exp, load_pairs
load_exp().to_pickle(W + 'exp_extracted.pkl'); load_pairs().to_pickle(W + 'pairs_extracted.pkl')
runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'links.py'))
from identity import run
d = pd.read_pickle(W + 'exp_extracted.pkl'); L = pd.read_pickle(W + 'study_links.pkl'); M = pd.read_pickle(W + 'study_meta.pkl')
d2, canon, C = run(d, L, M.to_dict('index'))
d2.to_pickle(W + 'exp_merged.pkl'); C.to_pickle(W + 'copies.pkl'); pd.Series(canon).to_pickle(W + 'alias.pkl')
print('stage1 done:', len(d2), 'records,', len(C), 'copy rows')
