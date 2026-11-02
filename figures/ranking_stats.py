#!/usr/bin/env python
"""Solvent ranking statistics of Section 4 and Table S5 (48 isomer pairs measured in at least 30 solvents).
Selection, orientation and averaging are identical to Figure 6c. For each pair and model it gives the number of solvents
tested until the best one is found and the Spearman rank correlation rho between predicted and measured selectivity
(S >= 1) across solvents, for the new-isomer and the new-solvent test, and the paired Wilcoxon signed-rank tests.
Usage: python ranking_stats.py <this folder> [out.csv]"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr, wilcoxon
ROOT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'scripts'))
from load_all import load


def table(exclude_flag=True):
    d = load(); p, M = d['pairs'].copy(), d['models']
    b = pd.read_csv(ROOT.parent / 'validation' / 'oof_predictions_ln_gamma.csv', dtype={'canonical_record_id': str})
    for tag, pre in (('nsJ', 'newsolv_DMPNN'), ('nsG', 'newsolv_GHGNN')):
        m = dict(zip(b.canonical_record_id, b[[c for c in b.columns if c.startswith(pre)]].mean(axis=1)))
        p['D_' + tag] = p.canonical_record_id_A.map(m) - p.canonical_record_id_B.map(m)
    p['Tr'] = p.T_K.round(0)
    if exclude_flag:   # Arumugam et al. 2020, cyclohexane (SI S1.4)
        flag = (p.Study_ID == 'BIB:["fluidphaseequilib","505","112362"]') & \
               (p.Isomer_A.str.lower().eq('cyclohexane') | p.Isomer_B.str.lower().eq('cyclohexane'))
        p = p[~flag].copy()
    sgn = np.where(p.mol_A == p.pair_key.str.split('|').str[0], 1.0, -1.0)
    p['m'] = p.D * sgn
    p['j'] = p[['D_' + c for c in M['JOINT']]].mean(axis=1) * sgn
    p['g'] = p[['D_' + c for c in M['GHGNN']]].mean(axis=1) * sgn
    p['nj'] = p.D_nsJ * sgn; p['ng'] = p.D_nsG * sgn
    rows, T = [], {}
    for (key, Tr), g in p.groupby(['pair_key', 'Tr']):
        t = g.groupby('Solvent_Mol_ID')[['m', 'j', 'g', 'nj', 'ng']].mean()
        if len(t) >= 30:
            T[(key, Tr)] = t; rows.append(dict(key=key, T=Tr, ns=len(t)))
    u = pd.DataFrame(rows).sort_values('ns', ascending=False).drop_duplicates('key')
    out = []
    for _, r in u.iterrows():
        t = T[(r.key, r['T'])]; S = np.exp(t.m.abs().values); best = int(np.argmax(S)); o = np.sort(S)[::-1]
        rec = dict(pair_key=r.key, T_K=r['T'], n_solvents=len(t), best_over_second=o[0] / o[1], random=(len(t) + 1) / 2)
        for c in ('j', 'g', 'nj', 'ng'):
            v = t[c].abs().values
            order = np.argsort(-v, kind='stable')
            rec['tests_' + c] = int(np.where(order == best)[0][0]) + 1
            rec['rho_' + c] = spearmanr(v, np.abs(t.m.values)).correlation
        out.append(rec)
    return pd.DataFrame(out)


if __name__ == '__main__':
    u = table()
    lab = {'j': 'D-MPNN new isomers', 'g': 'GH-GNN new isomers', 'nj': 'D-MPNN new solvents', 'ng': 'GH-GNN new solvents'}
    print('pairs', len(u), ' random order median', u.random.median())
    for c, n in lab.items():
        print(f'{n:22s} tests median {u["tests_" + c].median():5.1f}  earlier than random {(u["tests_" + c] < u.random).mean():.2f}'
              f'  rho median {u["rho_" + c].median():.2f}  rho > 0 {(u["rho_" + c] > 0).mean():.2f}')
    w = lambda a, b: wilcoxon(a, b).pvalue
    print('tests vs random, new isomers: D-MPNN p=%.3g  GH-GNN p=%.3g' % (w(u.tests_j, u.random), w(u.tests_g, u.random)))
    print('tests vs random, new solvents: D-MPNN p=%.3g  GH-GNN p=%.3g' % (w(u.tests_nj, u.random), w(u.tests_ng, u.random)))
    print('tests D-MPNN vs GH-GNN, new isomers: p=%.3g' % w(u.tests_j, u.tests_g))
    print('tests new solvents vs new isomers: D-MPNN p=%.3g  GH-GNN p=%.3g' % (w(u.tests_nj, u.tests_j), w(u.tests_ng, u.tests_g)))
    print('rho new solvents vs new isomers: D-MPNN p=%.3g  GH-GNN p=%.3g' % (w(u.rho_nj, u.rho_j), w(u.rho_ng, u.rho_g)))
    print('second-best solvent within 5%% of the best: %.2f of pairs' % (u.best_over_second < 1.05).mean())
    a = table(exclude_flag=False)
    print('with the three excluded pairs: D-MPNN tests median', a.tests_j.median(), ' GH-GNN', a.tests_g.median())
    if len(sys.argv) > 2:
        u.to_csv(sys.argv[2], index=False)
