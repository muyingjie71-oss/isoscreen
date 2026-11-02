"""Acceptance checks for the production GH-GNN (run from the package root).

Full check:  python production_check_ghgnn.py --data data --models <export>/ghgnn --report r.json --grid g.csv
Grid only:   python production_check_ghgnn.py --data data --models <export>/ghgnn --grid g_cpu.csv --grid_only --device cpu

1. ln gamma_inf R2 on the training records (seed mean); target > 0.95.
2. Median selectivity error on all matched pairs, exp(median |D_pred - D|); target < 1.04.
3. Writes the fixed CPU/GPU check grid (check_grid.py).
Records with Al/Co/Fe/Sb molecules (outside the featuriser) are left out, as in training."""
import sys,json,argparse,time
from pathlib import Path
import numpy as np,pandas as pd,torch
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
from predict_ghgnn import GHGNNPredictor,supported
import check_grid

p=argparse.ArgumentParser()
p.add_argument('--data',required=True);p.add_argument('--models',required=True)
p.add_argument('--report');p.add_argument('--pred_out');p.add_argument('--grid',required=True);p.add_argument('--grid_only',action='store_true')
p.add_argument('--device')
a=p.parse_args();t0=time.time();D=Path(a.data)
torch.set_num_threads(max(1,torch.get_num_threads()))
P=GHGNNPredictor(a.models,a.device)
rows=check_grid.grid(D)
g=P.predict([r['solute_smiles'] for r in rows],[r['solvent_smiles'] for r in rows],[r['T_K'] for r in rows],per_seed=True)
check_grid.write(a.grid,rows,g,P.seeds)
print('grid',len(rows),'points on',P.dev,flush=True)
if a.grid_only:sys.exit(0)

t=pd.read_csv(D/'task1.csv',dtype={'canonical_record_id':str},keep_default_na=False)
pr=pd.read_csv(D/'task2.csv',dtype={'canonical_record_id_A':str,'canonical_record_id_B':str},keep_default_na=False)
mol=pd.read_csv(D/'molecules.csv');smi=dict(zip(mol.Mol_ID,mol.SMILES))
ok={k for k in set(t.Solute_Mol_ID)|set(t.Solvent_Mol_ID) if supported(smi[k])}
t=t[t.Solute_Mol_ID.isin(ok)&t.Solvent_Mol_ID.isin(ok)].reset_index(drop=True)
res=P.predict(t.Solute_Mol_ID.map(smi),t.Solvent_Mol_ID.map(smi),t.T_K.astype(float),per_seed=True)
y=t.ln_gamma_inf.astype(float).values;pred=res['mean']
r2=1-np.sum((pred-y)**2)/np.sum((y-y.mean())**2);err=np.abs(pred-y)
pm=dict(zip(t.canonical_record_id,pred))
q=pr[pr.canonical_record_id_A.isin(pm)&pr.canonical_record_id_B.isin(pm)]
Dx=q.D_AB.astype(float).values;Dh=q.canonical_record_id_A.map(pm).values-q.canonical_record_id_B.map(pm).values
derr=np.abs(Dh-Dx);dr2=1-np.sum((Dh-Dx)**2)/np.sum((Dx-Dx.mean())**2)
rep={'model':'GH-GNN (IL-extended, retrained)','seeds':P.seeds,'device':str(P.dev),'n_records':int(len(y)),'n_pairs':int(len(q)),
     'excluded_unsupported_molecules':sorted(k for k in set(pd.read_csv(D/'task1.csv',usecols=['Solute_Mol_ID','Solvent_Mol_ID']).stack()) if k not in ok),
     'ln_gamma_R2_train':float(r2),
     'ln_gamma_R2_train_per_seed':{s:float(1-np.sum((res[s]-y)**2)/np.sum((y-y.mean())**2)) for s in P.seeds},
     'ln_gamma_median_abs_error':float(np.median(err)),'typical_gamma_error_pct':float(100*(np.exp(np.median(err))-1)),
     'selectivity_median_error_factor':float(np.exp(np.median(derr))),'selectivity_lnS_R2_train':float(dr2),'seconds':time.time()-t0}
rep['pass']={'R2_gt_0.95':rep['ln_gamma_R2_train']>0.95,'selectivity_factor_lt_1.04':rep['selectivity_median_error_factor']<1.04}
if a.pred_out:pd.DataFrame({'canonical_record_id':t.canonical_record_id,'pred_ln_gamma_inf':pred}).to_csv(a.pred_out,index=False)
Path(a.report).write_text(json.dumps(rep,indent=1));print(json.dumps(rep,indent=1))
