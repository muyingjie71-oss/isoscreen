"""Single entry point for every figure: loads the data and the out-of-fold predictions (new-isomer test) of all models.

Usage (from any directory):
    import sys; sys.path.insert(0, '<package>/scripts'); from load_all import load
    d = load()                      # common domain (GH-GNN coverage): 68,538 observations, 31,496 pairs
    d = load(common_domain=False)   # all records (GH-GNN columns NaN for 225 records)

Returned dict:
  obs    : one row per measurement (task1) + one column per model run with predicted ln gamma_inf
  pairs  : one row per isomer pair under matched conditions (task2); D = ln gA - ln gB (experimental),
           plus D_<model> = predicted difference for every model column, and helper columns
           (abs_D, S_inf = exp|D|, relation_en, solvent_is_IL, pair_key = sorted solute Mol_IDs)
  models : {'JOINT': [3 seed columns], 'ABS': [...], 'GHGNN': [...], 'ECFP': ['newiso_ECFP']}
           JOINT is D-MPNN trained on activity coefficients and selectivities (lambda = 1),
           ABS is D-MPNN trained on activity coefficients only (lambda = 0)
  ens    : ensemble (seed-mean) column names: JOINT_ens, ABS_ens, GHGNN_ens
  mol    : Mol_ID -> SMILES
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / 'training' / 'data' / 'new_isomer_test'   # written by training/prepare_data.py
PREDICTIONS = ROOT.parent / 'validation' / 'oof_predictions_all_models.csv.gz'

RELATION_EN = {
    '位置异构': 'positional',
    '骨架/链异构': 'skeletal (chain branching)',
    '环-链异构': 'ring-chain',
    '连接/分配异构': 'connectivity (metamer)',
    '官能团异构': 'functional group',
    '骨架/环系异构': 'skeletal (ring system)',
    '不饱和类型异构': 'unsaturation type',
}

MODELS = {
    'JOINT': ['newiso_DMPNN_20261007', 'newiso_DMPNN_20261008', 'newiso_DMPNN_20261009'],
    'ABS': ['newiso_DMPNNact_20261007', 'newiso_DMPNNact_20261008', 'newiso_DMPNNact_20261009'],
    'GHGNN': ['newiso_GHGNN_20261007', 'newiso_GHGNN_20261008', 'newiso_GHGNN_20261009'],
    'ECFP': ['newiso_ECFP'],
}


def load(common_domain=True):
    t = pd.read_csv(DATA / 'task1.csv', dtype={'canonical_record_id': str}, keep_default_na=False, encoding='utf-8-sig')
    p = pd.read_csv(DATA / 'task2.csv', dtype={'canonical_record_id_A': str, 'canonical_record_id_B': str},
                    keep_default_na=False, encoding='utf-8-sig')
    pr = pd.read_csv(PREDICTIONS, dtype={'canonical_record_id': str})
    pr = pr[['canonical_record_id'] + sum(MODELS.values(), [])]
    mol = pd.read_csv(DATA / 'molecules.csv', encoding='utf-8-sig')
    smi = dict(zip(mol.Mol_ID, mol.SMILES))
    for c in ('T_K', 'ln_gamma_inf', 'gamma_inf'):
        t[c] = t[c].astype(float)
    for c in ('T_K', 'D_AB', 'ln_gamma_A', 'ln_gamma_B'):
        p[c] = p[c].astype(float)
    obs = t.merge(pr, on='canonical_record_id', how='left', validate='one_to_one')
    if common_domain:
        obs = obs[obs[MODELS['GHGNN'][0]].notna()].reset_index(drop=True)
        keep = set(obs.canonical_record_id)
        p = p[p.canonical_record_id_A.isin(keep) & p.canonical_record_id_B.isin(keep)].reset_index(drop=True)
    ens = {}
    for k in ('JOINT', 'ABS', 'GHGNN'):
        obs[k + '_ens'] = obs[MODELS[k]].mean(axis=1)
        ens[k] = k + '_ens'
    idx = obs.set_index('canonical_record_id')
    mid = idx.Solute_Mol_ID
    pairs = p.rename(columns={'D_AB': 'D'}).copy()
    pairs['abs_D'] = pairs.D.abs()
    pairs['S_inf'] = np.exp(pairs.abs_D)           # selectivity >= 1 (direction-free)
    pairs['relation_en'] = pairs.Isomer_Relation.map(RELATION_EN).fillna(pairs.Isomer_Relation)
    pairs['solvent_is_IL'] = [('.' in smi.get(s, '')) or ('+' in smi.get(s, '')) for s in pairs.Solvent_Mol_ID]
    ma = mid.reindex(pairs.canonical_record_id_A).values
    mb = mid.reindex(pairs.canonical_record_id_B).values
    pairs['mol_A'], pairs['mol_B'] = ma, mb
    pairs['pair_key'] = [('|'.join(sorted((a, b)))) for a, b in zip(ma, mb)]
    for c in sum(MODELS.values(), []) + list(ens.values()):
        v = idx[c]
        pairs['D_' + c] = v.reindex(pairs.canonical_record_id_A).values - v.reindex(pairs.canonical_record_id_B).values
    return dict(obs=obs, pairs=pairs, models=MODELS, ens=ens, mol=smi)


if __name__ == '__main__':
    d = load()
    print('observations', len(d['obs']), ' pairs', len(d['pairs']), ' families with pairs', d['pairs'].Family_ID.nunique())
