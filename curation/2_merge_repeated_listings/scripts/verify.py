"""Independent checks on exported merged files against untouched extracted audit tables."""
import pandas as pd, numpy as np, json, sys
import sys,os; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from paths import O, AUDIT as X
t=pd.read_csv(O+'task1_merged.csv',float_precision='round_trip'); p=pd.read_csv(O+'task2_merged.csv'); dcs=pd.read_csv(O+'record_decisions.csv')
extracted=pd.read_csv(X+'deduplicated_experiments.csv.gz',usecols=['canonical_record_id','gamma_inf','T_K','Family_ID','Member_ID','Solvent_Mol_ID','Model_Eligible','Fold_ID'],low_memory=False,float_precision='round_trip').set_index('canonical_record_id')
fold=json.load(open(X+'family_fold_manifest.json'))
ok=lambda c,msg: print(('PASS ' if c else 'FAIL ')+msg) or c
r=[]
r.append(ok(t.canonical_record_id.is_unique,'task1 IDs unique'))
r.append(ok(t.canonical_record_id.isin(extracted.index).all(),'task1 IDs all exist in extracted canonical table'))
r.append(ok((extracted.loc[t.canonical_record_id,'Model_Eligible']=='是').all(),'task1 ⊆ extracted model-eligible'))
r.append(ok(np.array_equal(extracted.loc[t.canonical_record_id,'gamma_inf'].values,t.gamma_inf.values),'gamma values byte-identical to extracted (no rewriting/averaging)'))
r.append(ok(np.allclose(t.ln_gamma_inf,np.log(t.gamma_inf)),'ln_gamma consistent'))
r.append(ok(len(dcs)==int((extracted.Model_Eligible=='是').sum()) and set(dcs.canonical_record_id)==set(extracted.index[extracted.Model_Eligible=='是']),'every extracted model record has exactly one merged decision'))
r.append(ok(set(dcs.canonical_record_id[dcs.Decision=='KEEP'])==set(t.canonical_record_id),'KEEP decisions == task1'))
merged=dcs[dcs.Decision.str.startswith('MERGED_INTO_')]
tgt=merged.Decision.str.replace('MERGED_INTO_','')
r.append(ok(tgt.isin(t.canonical_record_id).all(),'every merged copy points to a kept record'))
gm=dict(zip(t.canonical_record_id,t.gamma_inf)); gc=dict(zip(dcs.canonical_record_id,dcs.gamma_inf))
span=np.abs(np.log(merged.gamma_inf.values)-np.log(tgt.map(gm).values))
r.append(ok(span.max()<0.01,f'merged copies within 1% of representative (max |dlnγ|={span.max():.4f})'))
r.append(ok(p.Pair_ID.is_unique and not p.duplicated(['canonical_record_id_A','canonical_record_id_B']).any(),'pairs unique'))
r.append(ok(p.canonical_record_id_A.isin(t.canonical_record_id).all() and p.canonical_record_id_B.isin(t.canonical_record_id).all(),'pair endpoints in task1'))
ta=t.set_index('canonical_record_id')
A=ta.loc[p.canonical_record_id_A]; B=ta.loc[p.canonical_record_id_B]
r.append(ok((A.Family_ID.values==B.Family_ID.values).all() and (A.Member_ID.values!=B.Member_ID.values).all(),'pairs: same family, different connectivity member'))
r.append(ok((A.Solvent_Mol_ID.values==B.Solvent_Mol_ID.values).all() and (A.Solvent_Mol_ID.values==p.Solvent_Mol_ID.values).all(),'pairs: same solvent structure'))
r.append(ok((np.abs(A.T_K.values-B.T_K.values)<=0.005).all(),'pairs: same temperature (±0.005 K)'))
r.append(ok(np.allclose(p.D_AB,np.log(A.gamma_inf.values)-np.log(B.gamma_inf.values)),'D_AB = lnγA − lnγB from task1 values'))
fm=fold['assignment'] if 'assignment' in fold else fold
r.append(ok((t.Family_ID.map(fm)==t.Fold_ID).all(),'task1 folds equal the fixed family→fold manifest'))
r.append(ok((p.Family_ID.map(fm)==p.Fold_ID).all(),'pair folds equal the fixed manifest'))
r.append(ok((t.Study_Provenance_Status[t.canonical_record_id.isin(set(p.canonical_record_id_A)|set(p.canonical_record_id_B))]=='CONFIRMED').all(),'all pair endpoints have confirmed original study'))
cross=(A.Study_ID.values!=p.Study_ID.values)|(B.Study_ID.values!=p.Study_ID.values)
print(f'INFO pairs whose endpoint is a representative from another publication (copy remap): {int(cross.sum())}')
print('ALL PASS' if all(r) else 'SOME CHECKS FAILED')
