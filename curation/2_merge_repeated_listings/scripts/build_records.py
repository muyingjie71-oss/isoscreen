import sys, json, itertools, collections
import pandas as pd, numpy as np
sys.path.insert(0,__import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from paths import W
from pairing import sig, compat
PTOL=0.5
d=pd.read_pickle(W+'exp_merged.pkl'); C=pd.read_pickle(W+'copies.pkl'); p7=pd.read_pickle(W+'pairs_extracted.pkl')
m=d[d.Model_Eligible=='是'].copy()
rep=dict(zip(C.canonical_record_id,C.copy_cluster)); m['rep']=m.canonical_record_id.map(lambda i: rep.get(i,i))
m['Tr']=m.T_K.round(2); m['sg']=m.Effective_Conditions.map(sig)
mi=m.set_index('canonical_record_id')
# ---- special-record flags (kept; for sensitivity analysis) ----
meth=m.Raw_实验方法.fillna('').str.lower()
flag=np.where(m.Raw_相态.eq('Smectic liquid crystal'),'SOLVENT_LIQUID_CRYSTAL_PHASE',
     np.where(meth.str.contains("henry|extrapolat|estimated from"),'DERIVED_FROM_OTHER_PROPERTY',
     np.where(m.Raw_相态.eq('Gas'),'SOLUTE_GAS_PHASE_LABEL','')))
m['Special_Record_Flag']=flag
# ---- cross-database transcription conflicts inside a unified publication ----
PRIMARY=('ThermoML','ILThermo')
def is_secondary_only(sl): return not any(p in sl for p in PRIMARY)
hold={}
mm=m[m.canonical_record_id==m.rep]
for k,x in mm[mm.Study_Provenance_Status=='CONFIRMED'].groupby(['Study_ID_Merged','Solute_Mol_ID','Solvent_Audit_ID','Tr']):
    if len(x)<2: continue
    prim=x[~x.source_list.map(is_secondary_only)]; sec=x[x.source_list.map(is_secondary_only)]
    if len(prim)==0 or len(sec)==0: continue
    for i,r in sec.iterrows():
        if any(compat(r.sg,q,PTOL) for q in prim.sg):
            hold[r.canonical_record_id]=('HOLD_SECONDARY_TRANSCRIPTION_CONFLICT','The same paper differs by more than 0.5% between ThermoML/ILThermo and a compilation under compatible conditions; the database record is kept and the compilation value held back.')
m['hold']=m.canonical_record_id.map(lambda i: hold.get(i,('',''))[0])
m['hold_reason']=m.canonical_record_id.map(lambda i: hold.get(i,('',''))[1])
print('secondary transcription conflicts held:',len(hold))
# ---- task1 merged: representatives only ----
t1=m[(m.canonical_record_id==m.rep)&(m.hold=='')].copy()
# ---- pairs merged ----
pc=m[(m.Study_Provenance_Status=='CONFIRMED')&(~m.rep.isin(hold.keys()))].copy()
def merge_sig(sigs):
    out=['']*5
    for s in sigs:
        for i,v in enumerate(s):
            if out[i]=='' and v!='': out[i]=v
    return out
# collapse copies within the same unified study context
rows=[]
for (st,r),x in pc.groupby(['Study_ID_Merged','rep']):
    sgs=list(x.sg)
    # if copies within a study carry mutually incompatible signatures keep them separate
    if all(compat(a,b,PTOL) for a,b in itertools.combinations(sgs,2)):
        rows.append(dict(rid=r,study=st,sg=merge_sig(sgs),n_src=len(x)))
    else:
        for s in sgs: rows.append(dict(rid=r,study=st,sg=s,n_src=1))
P=pd.DataFrame(rows)
P=P.join(mi[['Family_ID','Member_ID','Solvent_Audit_ID','Solvent_Mol_ID','Tr','gamma_inf']],on='rid')
out={}
for k,x in P.groupby(['Family_ID','study','Solvent_Audit_ID','Tr']):
    if len(x)<2: continue
    ids=x.rid.values; mem=x.Member_ID.values; sg=list(x.sg)
    distinct=[]; [distinct.append(s) for s in sg if s not in distinct]
    amb=set()
    for i,s in enumerate(sg):
        c=[t for t in distinct if t!=s and compat(s,t,PTOL)]
        if any(not compat(a,b,PTOL) for a,b in itertools.combinations(c,2)): amb.add(i)
    repm=set()
    for i,j in itertools.combinations(range(len(x)),2):
        if i in amb or j in amb: continue
        if mem[i]==mem[j] and compat(sg[i],sg[j],PTOL): repm.update((i,j))
    for i,j in itertools.combinations(range(len(x)),2):
        if mem[i]==mem[j] or i in amb or j in amb or i in repm or j in repm: continue
        if compat(sg[i],sg[j],PTOL):
            a,b=sorted((ids[i],ids[j])); key=(a,b)
            out.setdefault(key,[]).append(dict(study=k[1],sig=json.dumps(merge_sig([sg[i],sg[j]]))))
pairs=[]
for (a,b),ctx in out.items():
    ra,rb=mi.loc[a],mi.loc[b]
    pairs.append(dict(canonical_record_id_A=a,canonical_record_id_B=b,Family_ID=ra.Family_ID,Study_ID=ctx[0]['study'],n_study_contexts=len(ctx),
        Solvent_Mol_ID=ra.Solvent_Mol_ID,Solvent=ra.Solvent,T_K=ra.T_K,Isomer_A=ra.Solute,Isomer_B=rb.Solute,
        ln_gamma_A=np.log(ra.gamma_inf),ln_gamma_B=np.log(rb.gamma_inf),D_AB=np.log(ra.gamma_inf)-np.log(rb.gamma_inf),Fold_ID=ra.Fold_ID,Matched_Conditions=ctx[0]['sig']))
p8=pd.DataFrame(pairs)
# relation labels from extracted pairs (by member pair)
lab={}
for a,b,r in zip(p7.canonical_record_id_A,p7.canonical_record_id_B,p7.Isomer_Relation):
    lab[frozenset((mi.at[a,'Member_ID'],mi.at[b,'Member_ID']))]=r
p8['Isomer_Relation']=[lab.get(frozenset((mi.at[a,'Member_ID'],mi.at[b,'Member_ID'])),'') for a,b in zip(p8.canonical_record_id_A,p8.canonical_record_id_B)]
t1.to_pickle(W+'task1_merged.pkl'); p8.to_pickle(W+'task2_merged.pkl'); m.to_pickle(W+'model_extracted_annot.pkl')
# ---- closure / comparison ----
extracted_set={frozenset((rep.get(a,a),rep.get(b,b))) for a,b in zip(p7.canonical_record_id_A,p7.canonical_record_id_B)}
merged_set={frozenset((a,b)) for a,b in zip(p8.canonical_record_id_A,p8.canonical_record_id_B)}
print('task1 extracted',len(m),'-> merged',len(t1),'(removed copies',len(m)-len(t1),')')
print('pairs extracted',len(p7),'| extracted after remap+dedupe',len(extracted_set),'| merged',len(p8))
print('extracted(remapped) not in merged:',len(extracted_set-merged_set),' new in merged:',len(merged_set-extracted_set))
print('unlabelled relation pairs:',(p8.Isomer_Relation=='').sum())
print('families',p8.Family_ID.nunique(),'studies',p8.Study_ID.nunique(),'solvents',p8.Solvent_Mol_ID.nunique())
print('flags task1:',t1.Special_Record_Flag.value_counts().to_dict())
