"""Noise floor of the isomer difference D = ln g_A - ln g_B (written into counts.json by export_tables.py).

1. Cross-study replicates: the same isomer pair (solute Mol_IDs sorted, D oriented accordingly), same solvent and
   temperature (rounded to 0.1 K) measured in two different (unified) studies; D is first averaged within a study.
   -> replicates.csv (= cross_study_replicates.csv) and noise_rep.json
      q = quartiles and 90th percentile of |dD|; sigma = 1.4826 * median|dD| / sqrt(2); studypair_median = median over
      study pairs of the per-study-pair median |dD|.
2. Reported uncertainty: u_rel = reported u(gamma) / gamma.
   -> noise_u.json: u_rel_median over merged single records with u_rel > 0; uD_median = median of sqrt(u_A^2 + u_B^2)
      over merged pairs where both ends report u_rel > 0 (upper bound if the two errors were independent).
"""
import sys, os, json, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from paths import W
p = pd.read_pickle(W + 'task2_merged.pkl'); e = pd.read_pickle(W + 'exp_merged.pkl').set_index('canonical_record_id')
t1 = pd.read_pickle(W + 'task1_merged.pkl')
mid = e.Solute_Mol_ID
a = p.canonical_record_id_A.map(mid); b = p.canonical_record_id_B.map(mid); flip = a > b
q = p.assign(pair=np.where(flip, b + '|' + a, a + '|' + b), Do=np.where(flip, -p.D_AB, p.D_AB), T=p.T_K.round(1))
g = q.groupby(['pair', 'Solvent_Mol_ID', 'T', 'Study_ID']).Do.mean().reset_index()
rows = []
for (pr, s, T), x in g.groupby(['pair', 'Solvent_Mol_ID', 'T']):
    if x.Study_ID.nunique() < 2:
        continue
    x = x.sort_values('Study_ID')
    for (_, r1), (_, r2) in itertools.combinations(x.iterrows(), 2):
        rows.append(dict(pair=pr, solvent=s, T=T, s1=r1.Study_ID, s2=r2.Study_ID, diff=abs(r1.Do - r2.Do), D1=r1.Do, D2=r2.Do))
R = pd.DataFrame(rows)
R.to_csv(W + 'replicates.csv', index=False)
med = float(np.median(R['diff']))
rep = dict(n=len(R), n_study_pairs=int(R.groupby(['s1', 's2']).ngroups),
           q=[round(float(v), 3) for v in np.quantile(R['diff'], [0.25, 0.5, 0.75, 0.9])],
           sigma=1.4826 * med / np.sqrt(2), studypair_median=float(R.groupby(['s1', 's2'])['diff'].median().median()))
json.dump(rep, open(W + 'noise_rep.json', 'w'))
u = pd.to_numeric(e['Raw_活度系数不确定度'], errors='coerce') / pd.to_numeric(e.gamma_inf, errors='coerce')
ut = u.reindex(t1.canonical_record_id)
ua = p.canonical_record_id_A.map(u); ub = p.canonical_record_id_B.map(u); ok = ua.notna() & ub.notna() & (ua > 0) & (ub > 0)
pd.Series(dict(u_rel_median=float(ut[ut > 0].median()), uD_median=float(np.median(np.sqrt(ua[ok] ** 2 + ub[ok] ** 2))),
               n_pairs_u=float(ok.sum()))).to_json(W + 'noise_u.json')
print('noise:', rep, '| reported-u pairs', int(ok.sum()))
