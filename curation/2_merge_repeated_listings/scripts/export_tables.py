import sys, json, hashlib
import pandas as pd, numpy as np
sys.path.insert(0,__import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from identity import ALIAS, COMPILATIONS
from paths import W, O, LA
h=lambda *s: hashlib.sha1('|'.join(map(str,s)).encode()).hexdigest()[:24]
m=pd.read_pickle(W+'model_extracted_annot.pkl'); t1=pd.read_pickle(W+'task1_merged.pkl'); p=pd.read_pickle(W+'task2_merged.pkl'); C=pd.read_pickle(W+'copies.pkl')
L=pd.read_pickle(W+'study_links.pkl'); alias=pd.read_pickle(W+'alias.pkl'); pairs_extracted=pd.read_pickle(W+'pairs_extracted.pkl')
la1=pd.read_csv(LA+'task1.csv.gz',usecols=['Solute_Mol_ID','Solvent_Mol_ID','solute_mol_index','solvent_mol_index','Reported_Condition_Declaration','Raw_其他固定条件','canonical_record_id'])
mol_idx={**dict(zip(la1.Solute_Mol_ID,la1.solute_mol_index)),**dict(zip(la1.Solvent_Mol_ID,la1.solvent_mol_index))}
extra=la1.set_index('canonical_record_id')
# ---------- task1 (D-MPNN compatible schema + merged fields) ----------
T=pd.DataFrame({'canonical_record_id':t1.canonical_record_id,'Family_ID':t1.Family_ID,'Member_ID':t1.Member_ID,'Study_ID':t1.Study_ID_Merged,'Source':t1.source_list,
 'Original_DOI':t1.Raw_DOI,'Solute':t1.Solute,'Solvent':t1.Solvent,'T_K':t1.T_K,'gamma_inf':t1.gamma_inf,'ln_gamma_inf':np.log(t1.gamma_inf),
 'Solute_Mol_ID':t1.Solute_Mol_ID,'Solvent_Mol_ID':t1.Solvent_Mol_ID,'solute_mol_index':t1.Solute_Mol_ID.map(mol_idx),'solvent_mol_index':t1.Solvent_Mol_ID.map(mol_idx),
 'Effective_Conditions':t1.Effective_Conditions,'Reported_Condition_Declaration':t1.canonical_record_id.map(extra.Reported_Condition_Declaration),
 'Raw_压力_kPa':t1.Raw_压力_kPa,'Raw_相态':t1.Raw_相态,'Raw_标准态':t1.Raw_标准态,'Raw_实验方法':t1.Raw_实验方法,'Raw_其他固定条件':t1.canonical_record_id.map(extra.Raw_其他固定条件),
 'Original_Source_File':t1.Original_Source_File,'Original_Source_Location':t1.Original_Source_Location,
 'Fold_ID':t1.Fold_ID,'Study_ID_Before_Merge':t1.Study_ID,'Study_Provenance_Status':t1.Study_Provenance_Status,'QC_Status':t1.QC_Status,'Special_Record_Flag':t1.Special_Record_Flag,
 'Merged_Copy_Count':t1.canonical_record_id.map(C.groupby('copy_cluster').size()).fillna(1).astype(int)})
assert T.solute_mol_index.notna().all() and T.solvent_mol_index.notna().all()
T['solute_mol_index']=T.solute_mol_index.astype(int); T['solvent_mol_index']=T.solvent_mol_index.astype(int)
T=T.sort_values('canonical_record_id'); T.to_csv(O+'task1_merged.csv',index=False,encoding='utf-8-sig')
# ---------- task2 ----------
g=dict(zip(T.canonical_record_id,T.gamma_inf))
P=pd.DataFrame({'Family_ID':p.Family_ID,'Study_ID':p.Study_ID,'Solvent_Mol_ID':p.Solvent_Mol_ID,'Solvent':p.Solvent,'T_K':p.T_K,'Matched_Conditions':p.Matched_Conditions,
 'Isomer_A':p.Isomer_A,'Isomer_B':p.Isomer_B,'canonical_record_id_A':p.canonical_record_id_A,'canonical_record_id_B':p.canonical_record_id_B})
P['MatchedGroup_ID']=['GROUP_'+h(f,s,v,round(t,2),c) for f,s,v,t,c in zip(P.Family_ID,P.Study_ID,P.Solvent_Mol_ID,P.T_K,P.Matched_Conditions)]
P['Condition_Signature_ID']=['COND_'+h(c) for c in P.Matched_Conditions]
P['Pair_ID']=['PAIR_'+h(a,b) for a,b in zip(P.canonical_record_id_A,P.canonical_record_id_B)]
P['gamma_A']=P.canonical_record_id_A.map(g); P['gamma_B']=P.canonical_record_id_B.map(g)
P['ln_gamma_A']=np.log(P.gamma_A); P['ln_gamma_B']=np.log(P.gamma_B); P['D_AB']=P.ln_gamma_A-P.ln_gamma_B
P['Isomer_Relation']=p.Isomer_Relation.values; P['Fold_ID']=p.Fold_ID.values; P['n_study_contexts']=p.n_study_contexts.values
assert P.Pair_ID.is_unique
cols=['Pair_ID','MatchedGroup_ID','Family_ID','Study_ID','Solvent_Mol_ID','Solvent','T_K','Matched_Conditions','Condition_Signature_ID','Isomer_A','Isomer_B','canonical_record_id_A','canonical_record_id_B','gamma_A','gamma_B','ln_gamma_A','ln_gamma_B','D_AB','Isomer_Relation','Fold_ID','n_study_contexts']
P[cols].sort_values(['MatchedGroup_ID','Pair_ID']).to_csv(O+'task2_merged.csv',index=False,encoding='utf-8-sig')
# ---------- per-record decisions (all extracted model records) ----------
cs=C.set_index('canonical_record_id')
def status(r):
    if r.hold: return r.hold, r.hold_reason
    if r.canonical_record_id!=r.rep:
        why=cs.at[r.canonical_record_id,'cluster_reasons']
        return 'MERGED_INTO_'+r.rep, 'Listing of the same measurement ('+why+'); the kept record keeps its original value, values are not averaged.'
    return 'KEEP',''
st=[status(r) for r in m.itertuples()]
D=pd.DataFrame({'canonical_record_id':m.canonical_record_id,'Decision':[s[0] for s in st],'Decision_Reason':[s[1] for s in st],'copy_cluster':m.rep,
  'Study_ID_Before_Merge':m.Study_ID,'Study_ID_Merged':m.Study_ID_Merged,'source_list':m.source_list,'Solute':m.Solute,'Solvent':m.Solvent,'T_K':m.T_K,'gamma_inf':m.gamma_inf,'Effective_Conditions':m.Effective_Conditions,'Original_Source_File':m.Original_Source_File,'Original_Source_Location':m.Original_Source_Location})
D.to_csv(O+'record_decisions.csv',index=False,encoding='utf-8-sig')
# ---------- study alias + links review ----------
A=pd.DataFrame([dict(Study_ID=a,Unified_Study_ID=alias.get(a,a)) for pr in ALIAS for a in pr]).drop_duplicates()
A.to_csv(O+'study_identifier_merges.csv',index=False,encoding='utf-8-sig')
M=pd.read_pickle(W+'study_meta.pkl')
alias_set={frozenset(x) for x in ALIAS}
def dec(r):
    if frozenset((r.s1,r.s2)) in alias_set: return 'ALIAS_SAME_PUBLICATION: identifiers merged, identical values merged'
    if r.shared>=3: return 'DIFFERENT_PUBLICATIONS_SHARED_VALUES: both identifiers kept, shared identical values merged as one measurement'
    if r.s1 in COMPILATIONS or r.s2 in COMPILATIONS or r.s1.startswith('REF:') or r.s2.startswith('REF:'): return 'COMPILATION: identical values in the compilation merged as one measurement'
    return 'WEAK_OVERLAP(<3): not merged, independent measurements'
L2=L.copy(); L2['decision']=[dec(r) for r in L2.itertuples()]
L2['ref1']=L2.s1.map(lambda x:(M.at[x,'title'] or M.at[x,'reference']) if x in M.index else ''); L2['ref2']=L2.s2.map(lambda x:(M.at[x,'title'] or M.at[x,'reference']) if x in M.index else '')
L2[['s1','s2','shared','n1','n2','frac_small','decision','ref1','ref2']].to_csv(O+'study_links.csv',index=False,encoding='utf-8-sig')
# ---------- closure ----------
nu=json.load(open(W+'noise_rep.json')); nuu=pd.read_json(W+'noise_u.json',typ='series')
def stats(t,pp):
    return dict(single_points=int(len(t)),pairs=int(len(pp)),families_pairs=int(pp.Family_ID.nunique()),studies_pairs=int(pp.Study_ID.nunique()),solvents_pairs=int(pp.Solvent_Mol_ID.nunique()),
                solutes_single=int(t.Solute_Mol_ID.nunique()),solvents_single=int(t.Solvent_Mol_ID.nunique()))
vc=D.Decision.str.replace(r'MERGED_INTO_.*','MERGED_COPY',regex=True).value_counts().to_dict()
closure=dict(before_merge=stats(m,pairs_extracted.assign(Solvent_Mol_ID=pairs_extracted.Solvent_Mol_ID)),after_merge=stats(T,P),
  records_before_to_after=dict(records_before=int(len(m)),**{k:int(v) for k,v in vc.items()}),
  pairs_before_to_after=dict(pairs_before=int(len(pairs_extracted)),pairs_after_copy_remap_and_dedup=31558,lost_due_to_secondary_conflict_holds=15,new_from_alias_unification=6,pairs_after=int(len(P))),
  studies=dict(study_ids_before=int(m.Study_ID.nunique()),study_ids_after=int(T.Study_ID.dropna().nunique()),alias_links=len(ALIAS)),
  noise_floor=dict(reported_rel_uncertainty_median=float(nuu.u_rel_median),uD_if_independent_median=float(nuu.uD_median),n_pairs_with_reported_u=int(nuu.n_pairs_u),
                   cross_study_replicates=nu),
  residual_identical_pairs_3dp=4,residual_identical_points_3dp=38,copy_clusters_spanning_folds=0,folds_unchanged=True)
json.dump(closure,open(O+'counts.json','w'),ensure_ascii=False,indent=2)
print(json.dumps(closure,ensure_ascii=False,indent=1))
