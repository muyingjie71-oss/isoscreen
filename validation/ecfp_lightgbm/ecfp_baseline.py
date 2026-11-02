"""ECFP4 count fingerprints (solute ‖ solvent ‖ 1000/T) + LightGBM; absolute ln γ∞ only.
Same data and family folds as the other models; early stopping on the validation families; test families predicted once."""
import json,sys,time,numpy as np,pandas as pd,lightgbm as lgb
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
import pathlib;H=pathlib.Path(__file__).resolve().parent;D=str(H.parents[1]/'training'/'data'/'new_isomer_test')+'/';O=str(H)+'/'   # tables from training/prepare_data.py; outputs next to this script
t=pd.read_csv(D+'task1.csv',dtype={'canonical_record_id':str})
p=pd.read_csv(D+'task2.csv')
mol=pd.read_csv(D+'molecules.csv')
split=json.load(open(D+'split_manifest.json'))['rounds']
NB=1024;gen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=NB)
fp={}
for mid,smi in zip(mol.Mol_ID,mol.SMILES):
    m=Chem.MolFromSmiles(smi);fp[mid]=np.asarray(gen.GetCountFingerprintAsNumPy(m),np.float32)
X=np.hstack([np.stack([fp[x] for x in t.Solute_Mol_ID]),np.stack([fp[x] for x in t.Solvent_Mol_ID]),(1000./t.T_K.values)[:,None].astype(np.float32)])
y=t.ln_gamma_inf.values
pred=np.full(len(t),np.nan);info=[]
for k in range(1,6):
    r=split[str(k)];role=lambda f:t.Family_ID.isin(set(r[f+'_families'])).values
    tr,va,te=role('train'),role('validation'),role('test')
    t0=time.time()
    m=lgb.LGBMRegressor(n_estimators=4000,learning_rate=0.05,num_leaves=63,min_child_samples=20,subsample=0.8,subsample_freq=1,colsample_bytree=0.5,reg_lambda=1.0,objective='huber',alpha=1.0,random_state=20261007+k,n_jobs=2,verbose=-1)
    m.fit(X[tr],y[tr],eval_set=[(X[va],y[va])],eval_metric='l1',callbacks=[lgb.early_stopping(100,verbose=False)])
    pred[te]=m.predict(X[te],num_iteration=m.best_iteration_)
    info.append({'fold':k,'best_iteration':int(m.best_iteration_),'seconds':round(time.time()-t0,1),'test_mae':float(np.abs(pred[te]-y[te]).mean())})
    print(info[-1],flush=True)
assert np.isfinite(pred).all()
t['pred_ln_gamma_inf']=pred
t[['canonical_record_id','Family_ID','Fold_ID','Solute','Solvent','T_K','ln_gamma_inf','pred_ln_gamma_inf']].to_csv(O+'ECFP_LGBM_oof_observations.csv',index=False)
pm=dict(zip(t.canonical_record_id,pred))
p['pred_D_AB']=p.canonical_record_id_A.map(pm)-p.canonical_record_id_B.map(pm)
p[['Pair_ID','MatchedGroup_ID','Family_ID','Fold_ID','canonical_record_id_A','canonical_record_id_B','Isomer_Relation','D_AB','pred_D_AB']].to_csv(O+'ECFP_LGBM_oof_pairs.csv',index=False)
e=p.pred_D_AB-p.D_AB;s=p.D_AB.abs()>=0.05
res={'model':'ECFP4 count (1024+1024) + 1000/T, LightGBM Huber, absolute only','folds':info,
 'lng_mae':float(np.abs(pred-y).mean()),'lng_r2':float(1-((pred-y)**2).sum()/((y-y.mean())**2).sum()),
 'D_mae':float(e.abs().mean()),'D_zero_mae':float(p.D_AB.abs().mean()),
 'D_family_macro_mae':float(e.abs().groupby(p.Family_ID).mean().mean()),
 'sign_acc_|D|>=0.05':float((np.sign(p.pred_D_AB[s])==np.sign(p.D_AB[s])).mean()),'n_obs':int(len(t)),'n_pairs':int(len(p))}
json.dump(res,open(O+'ECFP_LGBM_report.json','w'),indent=2);print(json.dumps(res,indent=1))
