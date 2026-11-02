"""Study-pair overlap: count records shared with |dln gamma|<=0.005 at same solute/solvent/T."""
import sys, itertools, collections
import pandas as pd, numpy as np
sys.path.insert(0,__import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from paths import W
from studies import study_meta
d=pd.read_pickle(W+'exp_extracted.pkl'); c=d[d.Study_ID.notna()].copy()
c['Tr']=c.T_K.round(2); c['lg']=np.log(c.gamma_inf)
links=collections.Counter()
for k,x in c.groupby(['Solute_Mol_ID','Solvent_Audit_ID','Tr']):
    if x.Study_ID.nunique()<2: continue
    for (s1,l1),(s2,l2) in itertools.combinations(x[['Study_ID','lg']].values,2):
        if s1!=s2 and abs(l1-l2)<=0.005: links[tuple(sorted((s1,s2)))]+=1
size=c.groupby('Study_ID').size()
L=pd.DataFrame([(a,b,n,size[a],size[b]) for (a,b),n in links.items()],columns=['s1','s2','shared','n1','n2'])
L['frac_small']=L.shared/L[['n1','n2']].min(axis=1)
L.sort_values('shared',ascending=False).to_pickle(W+'study_links.pkl'); study_meta(d).to_pickle(W+'study_meta.pkl')
print('study links:',len(L))
