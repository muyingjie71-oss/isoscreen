"""Write the training tables (task1.csv, task2.csv) in the schema read by the D-MPNN code, and dataset.json."""
import sys, json, hashlib, csv, gzip
import pandas as pd, numpy as np
sys.path.insert(0,__import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from paths import P, W, O, LA, LIB
sys.path.insert(0,LIB)
import data as D

h=lambda *s: hashlib.sha1('|'.join(map(str,s)).encode()).hexdigest()[:24]
ext_t1=list(csv.DictReader(gzip.open(LA+'task1.csv.gz','rt',encoding='utf-8-sig',newline=''))); cols1=list(ext_t1[0].keys())
extracted=pd.read_csv(LA+'task1.csv.gz',dtype=str,keep_default_na=False).set_index('canonical_record_id')
merged_t1=pd.read_csv(O+'task1_merged.csv',dtype=str,keep_default_na=False)
merged_t2=pd.read_csv(O+'task2_merged.csv',dtype=str,keep_default_na=False)
e=pd.read_pickle(W+'exp_merged.pkl').set_index('canonical_record_id'); C=pd.read_pickle(W+'copies.pkl')
# publication contexts of each kept measurement
ctx={i:{s} for i,s in zip(merged_t1.canonical_record_id,merged_t1.Study_ID) if s}
for rep,g in C.groupby('copy_cluster'):
    if rep in ctx:
        ctx[rep]|={s for s in e.loc[g.canonical_record_id,'Study_ID_Merged'] if isinstance(s,str)}
# ---- task1: verbatim extracted D-MPNN row projection; only Study_ID unified; new columns appended ----
T=extracted.loc[merged_t1.canonical_record_id].reset_index()[cols1].copy()
T['Study_ID']=merged_t1.Study_ID.values
T['Study_Context_IDs']=[json.dumps(sorted(ctx.get(i,set())),ensure_ascii=False) if len(ctx.get(i,set()))>1 else '' for i in T.canonical_record_id]
for c in ['Study_ID_Before_Merge','Special_Record_Flag','Merged_Copy_Count','Fold_ID']: T[c]=merged_t1[c].values
assert (T.ln_gamma_inf.values==extracted.loc[merged_t1.canonical_record_id,'ln_gamma_inf'].values).all()
recs={r['canonical_record_id']:r for r in T.to_dict('records')}
# ---- task2: signatures via the package's own normaliser ----
keep=[];drop=collections=__import__('collections').Counter()
rows=[]
for r in merged_t2.to_dict('records'):
    a,b=recs[r['canonical_record_id_A']],recs[r['canonical_record_id_B']]
    if abs(float(a['T_K'])-float(b['T_K']))>1e-9: drop['endpoint_T_differs_>1e-9K']+=1; continue
    sigs=[D._condition_values(x['Effective_Conditions'],'e') for x in (a,b)]+[D._reported_conditions(x) for x in (a,b)]+[D._condition_values(r['Matched_Conditions'],'ctx')]
    out=['']*5; ok=True
    for s in sigs:
        for i,v in enumerate(s):
            if v and out[i] and v!=out[i]: ok=False
            if v and not out[i]: out[i]=v
    if not ok: drop['condition_conflict_after_normalisation']+=1; continue
    sig=json.dumps(out,ensure_ascii=False,separators=(',',':'))
    r['Matched_Conditions']=sig
    r['Condition_Signature_ID']='COND_'+h(sig)
    r['MatchedGroup_ID']='GROUP_'+h(r['Family_ID'],r['Study_ID'],r['Solvent_Mol_ID'],format(float(a['T_K']),'.12g'),sig)
    r['T_K']=a['T_K']
    rows.append(r)
P2=pd.DataFrame(rows)
cols2=['Pair_ID','MatchedGroup_ID','Family_ID','Study_ID','Solvent_Mol_ID','Solvent','T_K','Matched_Conditions','Condition_Signature_ID','Isomer_A','Isomer_B','canonical_record_id_A','canonical_record_id_B','gamma_A','gamma_B','ln_gamma_A','ln_gamma_B','D_AB','Isomer_Relation','Fold_ID','n_study_contexts']
# D_AB recomputed from the exact task1 strings
P2['D_AB']=[repr(float(recs[x]['ln_gamma_inf'])-float(recs[y]['ln_gamma_inf'])) for x,y in zip(P2.canonical_record_id_A,P2.canonical_record_id_B)]
P2=P2[cols2].sort_values(['MatchedGroup_ID','Pair_ID'])
T=T.sort_values('canonical_record_id')
T.to_csv(P+'data/task1.csv',index=False,encoding='utf-8-sig',quoting=csv.QUOTE_MINIMAL)
P2.to_csv(P+'data/task2.csv',index=False,encoding='utf-8-sig')
fz={'cohort':'merged','requires_matched_condition_signature':True,
    'expected_counts':{'task1':len(T),'task2':len(P2),'matched_groups':int(P2.MatchedGroup_ID.nunique())},
    'note':'Repeated listings merged and study identifiers unified; values, families and folds unchanged.',
    'pair_drops_in_training_tables':dict(drop)}
json.dump(fz,open(P+'data/dataset.json','w'),ensure_ascii=False,indent=2)
print(json.dumps({'task1':len(T),'task2':len(P2),'groups':int(P2.MatchedGroup_ID.nunique()),'drops':dict(drop),'multi_context_records':int((T.Study_Context_IDs!='').sum())},indent=1))
