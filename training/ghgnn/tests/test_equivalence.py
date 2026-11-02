"""Cached graph builder == upstream mol2graph.sys2graph (bit-identical tensors and descriptors) on a random sample."""
import sys,json
from pathlib import Path
HERE=Path(__file__).resolve().parent.parent;sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE/'ghgnn'))
import numpy as np,pandas as pd,torch
from rdkit import Chem
import mol2graph as M
from run_ghgnn_job import mol_cache,build
mol=pd.read_csv(HERE/'data/molecules.csv');smi=dict(zip(mol.Mol_ID,mol.SMILES))
bad=set()
for k,s in smi.items():
    try:[M.atom_features(a) for a in Chem.MolFromSmiles(s).GetAtoms()]
    except Exception:bad.add(k)
t=pd.read_csv(HERE/'data/task1.csv',dtype={'canonical_record_id':str},keep_default_na=False)
t=t[~(t.Solute_Mol_ID.isin(bad)|t.Solvent_Mol_ID.isin(bad))].sample(400,random_state=0)
df=pd.DataFrame({'Molecule_Solvent':[Chem.MolFromSmiles(smi[x]) for x in t.Solvent_Mol_ID],'Molecule_Solute':[Chem.MolFromSmiles(smi[x]) for x in t.Solute_Mol_ID],'y':t.ln_gamma_inf.astype(float).values})
up_s,up_u=M.sys2graph(df,'Molecule_Solvent','Molecule_Solute','y')
C=mol_cache({k:smi[k] for k in set(t.Solute_Mol_ID)|set(t.Solvent_Mol_ID)})
my_s,my_u,_=build(list(zip(t.Solvent_Mol_ID,t.Solute_Mol_ID,t.ln_gamma_inf.astype(float),t.T_K.astype(float))),C)
for A,B in ((up_s,my_s),(up_u,my_u)):
    for g1,g2 in zip(A,B):
        for k in ('x','edge_index','edge_attr','y'):assert torch.equal(getattr(g1,k),getattr(g2,k)),k
        for k in ('ap','bp','topopsa','hb','inter_hb'):assert float(getattr(g1,k))==float(getattr(g2,k)),k
print(json.dumps({'status':'PASS','records_compared':len(t),'upstream':'mol2graph.sys2graph','fields':'x,edge_index,edge_attr,y,ap,bp,topopsa,hb,inter_hb'}))
