"""Acceptance checks for the production D-MPNN (run from the package root).

Full check:  python shared/production_check.py --data data --models <export>/dmpnn --report r.json --grid g.csv
Grid only:   JAX_PLATFORMS=cpu python shared/production_check.py --data data --models <export>/dmpnn --grid g_cpu.csv --grid_only

1. ln gamma_inf R2 on the training records (SMILES path, seed mean); target > 0.95.
2. Median selectivity error on all matched pairs, exp(median |D_pred - D|); target < 1.04.
3. SMILES path == training path (cached graphs, trainer.predict) for every record and seed; target max |diff| < 1e-4.
4. Writes the fixed CPU/GPU check grid (check_grid.py)."""
import sys,json,argparse,time
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE.parent))
import jax
from predict_dmpnn import DMPNNPredictor
import check_grid

p=argparse.ArgumentParser()
p.add_argument('--data',required=True);p.add_argument('--models',required=True)
p.add_argument('--report');p.add_argument('--pred_out');p.add_argument('--grid',required=True);p.add_argument('--grid_only',action='store_true')
a=p.parse_args();t0=time.time()
P=DMPNNPredictor(a.models)
rows=check_grid.grid(a.data)
g=P.predict([r['solute_smiles'] for r in rows],[r['solvent_smiles'] for r in rows],[r['T_K'] for r in rows],per_seed=True)
check_grid.write(a.grid,rows,g,P.seeds)
print('grid',len(rows),'points on',jax.devices()[0].platform,flush=True)
if a.grid_only:sys.exit(0)

from data import Dataset
from trainer import predict as trainer_predict
ds=Dataset(a.data)
smi=[m['SMILES'] for m in ds.molecules]
su=[smi[i] for i in ds.sm];sv=[smi[i] for i in ds.vm];T=[float(r['T_K']) for r in ds.records]
pr=P.predict(su,sv,T,per_seed=True);pred=pr['mean'];y=ds.y64
r2=1-np.sum((pred-y)**2)/np.sum((y-y.mean())**2)
err=np.abs(pred-y)
dpred=pred[ds.pa]-pred[ds.pb];derr=np.abs(dpred-ds.dy64)
dr2=1-np.sum((dpred-ds.dy64)**2)/np.sum((ds.dy64-ds.dy64.mean())**2)
# training path: cached npz graphs through trainer.predict, per seed
class _S:pass
ids=np.arange(len(ds.records),dtype=np.int32);path_diff={}
for m in P.members:
    s=_S();s.params=m['params']
    tp=trainer_predict(ds,ids,s,lambda prm,b,fn=m['fn']:fn(prm,b),m['scalers'],512)
    path_diff[m['name']]=float(np.max(np.abs(tp-pr[m['name']])))
seed_r2={m['name']:float(1-np.sum((pr[m['name']]-y)**2)/np.sum((y-y.mean())**2)) for m in P.members}
rep={'model':'D-MPNN (A_JOINT, lambda=1)','seeds':P.seeds,'device':jax.devices()[0].platform,
     'n_records':int(len(y)),'n_pairs':int(len(ds.pairs)),
     'ln_gamma_R2_train':float(r2),'ln_gamma_R2_train_per_seed':seed_r2,
     'ln_gamma_median_abs_error':float(np.median(err)),'typical_gamma_error_pct':float(100*(np.exp(np.median(err))-1)),
     'selectivity_median_error_factor':float(np.exp(np.median(derr))),'selectivity_lnS_R2_train':float(dr2),
     'smiles_vs_training_path_max_abs_diff':path_diff,'seconds':time.time()-t0}
rep['pass']={'R2_gt_0.95':rep['ln_gamma_R2_train']>0.95,'selectivity_factor_lt_1.04':rep['selectivity_median_error_factor']<1.04,
             'smiles_path_equals_training_path_1e-4':max(path_diff.values())<1e-4}
if a.pred_out:
    import csv
    with open(a.pred_out,'w',encoding='utf-8',newline='') as f:
        w=csv.writer(f);w.writerow(['canonical_record_id','pred_ln_gamma_inf'])
        for r,v in zip(ds.records,pred):w.writerow([r['canonical_record_id'],repr(float(v))])
Path(a.report).write_text(json.dumps(rep,indent=1));print(json.dumps(rep,indent=1))
