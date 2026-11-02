"""COSMO-SAC predictions of the selectivity of isomer pairs (Supporting Information, S4).

COSMO-SAC 2010 with three sigma profiles and the 2014 dispersion correction (cosmosac_model.py), as
implemented by NIST (https://github.com/usnistgov/COSMOSAC). Sigma profiles of the University of Delaware
database are downloaded from that repository into a local cache at run time and are not redistributed.

Inputs (folder given by --inputs; not included in this repository):
  pairs.csv                isomer pairs with the measured and the out-of-fold D-MPNN value of
                           D = ln gamma_inf(A) - ln gamma_inf(B) (D_experimental, D_predicted)
  cross_solvent_pairs.csv  the same isomer pair measured in two solvents of one study
Both are from the dataset before repeated listings were merged (curation step 2).

Outputs (folder given by --out): cosmosac_pair_predictions.csv, cosmosac_cross_solvent_differences.csv,
summary_scores.csv, failed_pairs.csv, qa_profile_availability.json and qa_final.json.

Usage:
  python run_cosmosac.py --inputs INPUTS --network              # downloads missing sigma profiles
  python run_cosmosac.py --inputs INPUTS --cache SIGMA_CACHE    # offline, from a filled cache
"""
import argparse
import hashlib
import json
import pathlib
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache

import numpy as np
import pandas as pd
import requests

import cosmosac_model as model

HERE = pathlib.Path(__file__).resolve().parent
URL_BASE = 'https://raw.githubusercontent.com/usnistgov/COSMOSAC/master/profiles/UD/'
NIST_REPO = 'https://github.com/usnistgov/COSMOSAC'
SIGMA_DIR = 'sigma3'
KEY_FIELDS = ('Solute_Mol_ID_A', 'Solute_Mol_ID_B', 'Solvent_Mol_ID')


def read_inputs(folder):
    pairs = pd.read_csv(folder / 'pairs.csv')
    edges = pd.read_csv(folder / 'cross_solvent_pairs.csv')
    if not pairs.Pair_ID.is_unique or not edges.Edge_ID.is_unique:
        raise ValueError('Repeated Pair_ID or Edge_ID in the inputs')
    return pairs, edges


def fetch_bytes(url, max_retries=4):
    last = None
    for i in range(max_retries):
        try:
            r = requests.get(url, timeout=(10, 30)); r.raise_for_status()
            return r.content
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code in (400, 401, 403, 404, 410):
                raise RuntimeError(f'HTTP {e.response.status_code}: {url}') from e
            last = e
        except Exception as e:
            last = e
            if i < max_retries - 1:
                time.sleep(1.5 * (i + 1))
    raise RuntimeError(f'Download failed {url}: {last}')


def get_index(cache, network):
    """Standard InChIKeys of all molecules in the NIST profile database."""
    cache.mkdir(parents=True, exist_ok=True)
    p = cache / 'complist.txt'
    if not p.exists():
        if not network:
            raise RuntimeError('The NIST compound list is not in the cache. Run with --network to download it.')
        p.write_bytes(fetch_bytes(URL_BASE + 'complist.txt'))
    raw = p.read_text(encoding='utf8')
    keys = set(re.findall(r'\b[A-Z]{14}-[A-Z]{10}-[A-Z]\b', raw))
    if len(keys) < 1000:
        raise RuntimeError(f'NIST compound list has only {len(keys)} InChIKeys. Delete {p} and run again.')
    return keys, hashlib.sha256(raw.encode()).hexdigest()


def parse_sigma_file(path, expected_key):
    lines = path.read_text('utf8').splitlines()
    assert lines[0].startswith('# meta: '), f'Bad metadata header in {path.name}'
    meta = json.loads(lines[0][len('# meta: '):])
    assert meta['standard_INCHIKEY'] == expected_key, (expected_key, meta['standard_INCHIKEY'])
    data = [x.strip().split() for x in lines if x.strip() and not x.lstrip().startswith('#')]
    if len(data) != 153:
        raise ValueError(f'Expected 153 sigma bins in {path.name}, got {len(data)}')
    vals = np.asarray([[float(i) for i in x] for x in data], dtype=float)
    grid = np.tile(np.linspace(-.025, .025, 51), 3)
    assert np.max(np.abs(vals[:, 0] - grid)) < 1e-8, 'Wrong sigma grid'
    a = float(meta['area [A^2]']); v = float(meta['volume [A^3]'])
    if a <= 0 or v <= 0 or (vals[:, 1] < -1e-10).any():
        raise ValueError('Invalid COSMO volume, area or distribution')
    if abs(vals[:, 1].sum() / a - 1) > 3e-5:
        raise ValueError('Sigma profile does not add up to the COSMO area')
    nz = [(int(j), float(x)) for j, x in enumerate(vals[:, 1]) if x > 1e-13]
    return {'area': a, 'volume': v, 'dsp': float(meta.get('disp. e/kB [K]', 0.0) or 0.0),
            'flag': meta.get('disp. flag', 'NHB'), 'nz': nz, 'full_inchikey': expected_key}


def fetch_one(cache, key, network):
    p = cache / SIGMA_DIR / (key + '.sigma'); p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        if not network:
            raise FileNotFoundError(p)
        p.write_bytes(fetch_bytes(URL_BASE + SIGMA_DIR + '/' + key + '.sigma'))
    return parse_sigma_file(p, key)


def fetch_all(keys, cache, network, workers):
    avail, errors = {}, {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(fetch_one, cache, k, network): k for k in sorted(keys)}
        for done, fut in enumerate(as_completed(futures), 1):
            key = futures[fut]
            try:
                avail[key] = fut.result()
            except Exception as e:
                errors[key] = str(e)
            if done % 50 == 0 or done == len(keys):
                print(f'sigma profiles: {done}/{len(keys)} checked, {len(errors)} missing', flush=True)
    return avail, errors


def cross_solvent_differences(edges, pairs):
    """Change of D between two solvents of one study, measured, from D-MPNN and from COSMO-SAC.
    An edge is kept when every pair listed for both solvents was computed and its listings agree."""
    D = pairs.set_index('Pair_ID').D_cosmo.to_dict()
    prefix = str(pairs.Pair_ID.iloc[0]).split('_')[0] + '_'
    rx = re.compile(re.escape(prefix) + r'[0-9a-f]+')
    rows = []
    for r in edges.itertuples(index=False):
        a = rx.findall(str(r.original_Pair_IDs_S1)); b = rx.findall(str(r.original_Pair_IDs_S2))
        if not a or not b or any(k not in D for k in a + b):
            continue
        v1 = [D[k] for k in a]; v2 = [D[k] for k in b]
        if max(v1) - min(v1) > 1e-7 or max(v2) - min(v2) > 1e-7:
            continue
        rows.append([r.Edge_ID, r.Panel_ID, r.Family_ID, r.Isomer_Relation, r.Study_ID, r.T_K, r.Fold_ID, r.Solvent_1, r.Solvent_2,
                     r.deltaD_exp_S2minusS1, r.deltaD_pred_S2minusS1, float(np.mean(v2) - np.mean(v1))])
    out = pd.DataFrame(rows, columns=['Edge_ID', 'Panel_ID', 'Family_ID', 'Isomer_Relation', 'Study_ID', 'T_K', 'Fold_ID',
                                      'Solvent_1', 'Solvent_2', 'deltaD_exp', 'deltaD_DMPNN', 'deltaD_COSMO'])
    out['ae_DMPNN'] = (out.deltaD_DMPNN - out.deltaD_exp).abs()
    out['ae_COSMO'] = (out.deltaD_COSMO - out.deltaD_exp).abs()
    out['ae_null'] = out.deltaD_exp.abs()
    return out


def run(inputs, out, cache, network, workers, limit=None):
    pairs, edges = read_inputs(inputs)
    out.mkdir(parents=True, exist_ok=True)
    index, index_hash = get_index(cache, network)
    # only standard InChIKeys of neutral molecules without stereo layer are matched to the NIST database
    standard = re.compile(r'^MOL_([A-Z]{14}-UHFFFAOYSA-N)$')

    def key_of(s):
        m = standard.fullmatch(str(s))
        return m.group(1) if m else None
    for f in KEY_FIELDS:
        pairs[f + '_fullkey'] = pairs[f].map(key_of)
        pairs[f + '_index_covered'] = pairs[f + '_fullkey'].isin(index)
    eligible = pairs[pairs[[f + '_index_covered' for f in KEY_FIELDS]].all(axis=1)].copy()
    if limit is not None:   # small test run only
        eligible = eligible.sort_values('Pair_ID').head(limit).copy()
    if not len(eligible):
        raise RuntimeError('No pair has all three molecules in the NIST database')
    keys = set(eligible.Solute_Mol_ID_A_fullkey) | set(eligible.Solute_Mol_ID_B_fullkey) | set(eligible.Solvent_Mol_ID_fullkey)
    print(f'pairs with all three molecules in the NIST database: {len(eligible)}, molecules: {len(keys)}', flush=True)
    profiles, failed_profiles = fetch_all(keys, cache, network, workers)
    model.PROFILES.clear(); model.PROFILES.update(profiles)

    @lru_cache(maxsize=120000)
    def ln_gamma(solvent, solute, T):
        return model.ln_gamma_inf(T, solute, solvent)

    qa = {'NIST_index_sha256': index_hash, 'NIST_index_keys': len(index), 'candidate_pairs': len(eligible),
          'molecules': len(keys), 'profiles_available': len(profiles), 'profile_failures': len(failed_profiles),
          'NIST_repo': NIST_REPO, 'model': 'COSMO-SAC 2010 (Hsieh) with 2014 dispersion correction, NIST implementation',
          'status': 'pending'}
    # reference values of NIST (tests/catch_tests.cpp): n-pentane in n-heptane at 293.15 K
    pentane, heptane = 'OFBQJSOFQDEBGM-UHFFFAOYSA-N', 'IMNFDUFMRHMDMM-UHFFFAOYSA-N'
    ref, ref_err = fetch_all([pentane, heptane], cache, network, workers)
    if ref_err:
        raise RuntimeError('Reference sigma profiles unavailable: ' + str(ref_err))
    model.PROFILES.update(ref)
    z = ln_gamma(heptane, pentane, 293.15)
    if max(abs(z[1] + 0.0317), abs(z[2] - 0.0001), abs(z[3])) >= 0.002:
        raise RuntimeError(f'NIST reference values not reproduced: {z}')
    qa['NIST_reference_293.15K'] = z
    (out / 'qa_profile_availability.json').write_text(json.dumps({'metadata': qa, 'failed_profiles': failed_profiles}, indent=2), encoding='utf8')

    rows, failures = [], []
    for i, row in enumerate(eligible.itertuples(index=False)):
        a, b, s, T = row.Solute_Mol_ID_A_fullkey, row.Solute_Mol_ID_B_fullkey, row.Solvent_Mol_ID_fullkey, float(row.T_K)
        if any(k in failed_profiles for k in (a, b, s)):
            failures.append({'Pair_ID': row.Pair_ID, 'reason': 'sigma profile unavailable'}); continue
        try:
            A = ln_gamma(s, a, T); B = ln_gamma(s, b, T)
            rows.append({'Pair_ID': row.Pair_ID, 'D_cosmo': A[0] - B[0], 'D_cosmo_comb': A[1] - B[1],
                         'D_cosmo_resid': A[2] - B[2], 'D_cosmo_disp': A[3] - B[3],
                         'ln_gamma_A_COSMO': A[0], 'ln_gamma_B_COSMO': B[0]})
        except Exception as e:
            failures.append({'Pair_ID': row.Pair_ID, 'reason': repr(e)[:240]})
        if (i + 1) % 500 == 0:
            print(f'computed {i + 1}/{len(eligible)}', flush=True)
    if not rows:
        raise RuntimeError('No COSMO-SAC prediction could be computed')
    final = eligible.merge(pd.DataFrame(rows), on='Pair_ID', how='inner', validate='one_to_one')
    final['ae_COSMO'] = (final.D_experimental - final.D_cosmo).abs()
    final['ae_DMPNN'] = (final.D_experimental - final.D_predicted).abs()
    final['ae_Dzero'] = final.D_experimental.abs()
    final.to_csv(out / 'cosmosac_pair_predictions.csv', index=False)
    pd.DataFrame(failures, columns=['Pair_ID', 'reason']).to_csv(out / 'failed_pairs.csv', index=False)
    # mean absolute errors on the same pairs: over pairs, and over families, studies and isomer types
    score = []
    for label, key in [('record_micro', None), ('family_macro', 'Family_ID'), ('study_macro', 'Study_ID'), ('type_macro', 'Isomer_Relation')]:
        cols = ['ae_COSMO', 'ae_DMPNN', 'ae_Dzero']
        v = final[cols].mean() if key is None else final.groupby(key)[cols].mean().mean()
        score.append({'cohort': label, 'groups': len(final) if key is None else final[key].nunique(), **v.to_dict()})
    pd.DataFrame(score).to_csv(out / 'summary_scores.csv', index=False)
    cross = cross_solvent_differences(edges, final)
    cross.to_csv(out / 'cosmosac_cross_solvent_differences.csv', index=False)
    qa.update(status='complete', computed_pairs=len(final), failed_pairs=len(failures), matched_crosssolvent_edges=len(cross))
    (out / 'qa_final.json').write_text(json.dumps(qa, indent=2), encoding='utf8')
    print(json.dumps(qa, indent=2), flush=True)


def main():
    ap = argparse.ArgumentParser(description='COSMO-SAC predictions for isomer pairs')
    ap.add_argument('--inputs', type=pathlib.Path, required=True, help='folder with pairs.csv and cross_solvent_pairs.csv')
    ap.add_argument('--cache', type=pathlib.Path, default=HERE / 'sigma_cache', help='local cache of NIST sigma profiles')
    ap.add_argument('--out', type=pathlib.Path, default=HERE / 'outputs')
    ap.add_argument('--network', action='store_true', help='download missing sigma profiles')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--limit', type=int, default=None, help='compute only the first N pairs (test run)')
    a = ap.parse_args()
    run(a.inputs, a.out, a.cache, a.network, a.workers, a.limit)


if __name__ == '__main__':
    main()
