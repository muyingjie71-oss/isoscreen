"""Usage: python -I make_calibration.py   (about 1 minute)
Calibration tables for the web tool, from out-of-fold predictions of the paper runs.
Regimes: new_isomers (leave-family-out CV), known_isomers_new_solvents (solvent-held-out CV).
Predictions: seed ensemble (mean ln gamma over 3 seeds) per model, and the mean of the two models (AVG)."""
import numpy as np, pandas as pd, json, sys
from pathlib import Path
H = Path(__file__).resolve().parent
o = pd.read_csv(H / 'oof_predictions_ln_gamma.csv'); p = pd.read_csv(H / 'pairs_common_domain.csv')
m = dict(zip(o.canonical_record_id, range(len(o)))); ia = p.canonical_record_id_A.map(m).values; ib = p.canonical_record_id_B.map(m).values
d = lambda cols: np.mean([o[c].values[ia] - o[c].values[ib] for c in cols], axis=0)
S = lambda pre: [c for c in o.columns if c.startswith(pre)]
P = {('new_isomers', 'D-MPNN'): d(S('newiso_DMPNN_')), ('new_isomers', 'GH-GNN'): d(S('newiso_GHGNN_')),
     ('known_isomers_new_solvents', 'D-MPNN'): d(S('newsolv_DMPNN_')), ('known_isomers_new_solvents', 'GH-GNN'): d(S('newsolv_GHGNN_'))}
for r in ('new_isomers', 'known_isomers_new_solvents'): P[(r, 'AVG')] = (P[(r, 'D-MPNN')] + P[(r, 'GH-GNN')]) / 2
D = p.D.values; FLAG = ((p.Study_ID == 'BIB:["fluidphaseequilib","505","112362"]') & (p.Isomer_A.str.lower().eq('cyclohexane') | p.Isomer_B.str.lower().eq('cyclohexane'))).values
BINS = [1.0, 1.05, 1.12, 1.25, 1.65, np.inf]; LAB = ['1.00-1.05', '1.05-1.12', '1.12-1.25', '1.25-1.65', '>1.65']
cal, summ, typ = [], [], []
for (reg, mod), q in P.items():
    ps = np.exp(np.abs(q)); ms = np.exp(D * np.sign(q))          # measured S oriented in the predicted direction
    for lo, hi, lab in zip(BINS[:-1], BINS[1:], LAB):
        k = (ps >= lo) & (ps < hi)
        cal.append(dict(regime=reg, model=mod, predicted_S_bin=lab, lo=lo, hi=hi, n_pairs=int(k.sum()), share=round(k.mean(), 4),
                        hit_rate=round((ms[k] >= 1.05).mean(), 4), wrong_direction_rate=round((ms[k] <= 1 / 1.05).mean(), 4),
                        median_measured_S=round(float(np.median(ms[k])), 4)))
    err = np.abs(q - D); null = np.abs(D) < np.log(1.026)
    # tests to find the best solvent, pairs with >=30 solvents at one temperature (as Fig 6c, flagged study excluded)
    t = p.assign(q=q)[~FLAG].copy(); t['Tr'] = t.T_K.round(0); ranks, rnd = [], []
    groups = {kt: g.groupby('Solvent_Mol_ID').agg(m=('D', 'mean'), q=('q', 'mean')) for kt, g in t.groupby(['pair_key', 'Tr'])}
    sel = pd.Series({kt: len(g) for kt, g in groups.items() if len(g) >= 30}).reset_index(); sel.columns = ['k', 'T', 'n']
    sel = sel.sort_values('n', ascending=False).drop_duplicates('k')
    for k, T in zip(sel.k, sel['T']):
        g = groups[(k, T)]; b = g.m.abs().values.argmax()
        ranks.append(int(np.where(np.argsort(-g.q.abs().values, kind='stable') == b)[0][0]) + 1); rnd.append((len(g) + 1) / 2)
    ranks, rnd = np.array(ranks), np.array(rnd)
    summ.append(dict(regime=reg, model=mod, resolution=round(float(np.exp(np.median(err))), 4),
                     spurious_rate=round(float((np.abs(q[null]) > 0.11548).mean()), 4), hit_rate=round(float((ms >= 1.05).mean()), 4),
                     n_rank_pairs=len(ranks), median_tests_to_best=float(np.median(ranks)), median_tests_random=float(np.median(rnd)),
                     share_earlier_than_random=round(float((ranks < rnd).mean()), 3)))
    k = ps >= 1.12
    for ty, g in pd.Series(ms[k] >= 1.05).groupby(p.relation_en.values[k]):
        typ.append(dict(regime=reg, model=mod, isomer_type=ty, n_pairs_pred_ge_1_12=len(g), hit_rate=round(g.mean(), 4)))
out = sys.argv[1] if len(sys.argv) > 1 else str(H)
pd.DataFrame(cal).to_csv(f'{out}/calibration_by_predicted_S.csv', index=False)
pd.DataFrame(summ).to_csv(f'{out}/model_summary_by_regime.csv', index=False)
pd.DataFrame(typ).to_csv(f'{out}/hit_rate_by_isomer_type.csv', index=False)
print(pd.DataFrame(summ).to_string()); print(pd.DataFrame(cal).pivot_table(index=['regime', 'model'], columns='predicted_S_bin', values='hit_rate').to_string())
