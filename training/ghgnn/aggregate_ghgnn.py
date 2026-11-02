"""Pool five folds per seed into OOF + summary (ln γ∞ MAE, pair D MAE vs D=0, sign accuracy |D|>=0.05)."""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--root',required=True);p.add_argument('--seeds',nargs='+',type=int,required=True)
a=p.parse_args();root=Path(a.root);D=Path(a.data)
t=pd.read_csv(D/'task1.csv',dtype={'canonical_record_id':str},keep_default_na=False);pr=pd.read_csv(D/'task2.csv',dtype=str,keep_default_na=False)
rows=[]
for s in a.seeds:
    d=root/f'GHGNN_seed{s}'
    if not all((d/f'Fold_{k}/completed.json').exists() for k in range(1,6)):rows.append({'seed':s,'status':'incomplete'});continue
    o=pd.concat([pd.read_csv(d/f'Fold_{k}/test_observations_predictions.csv',dtype={'canonical_record_id':str}) for k in range(1,6)])
    excl=set(json.loads((d/'Fold_1/run_manifest.json').read_text())['excluded_unsupported_molecules'])
    dom=t[~(t.Solute_Mol_ID.isin(excl)|t.Solvent_Mol_ID.isin(excl))]
    assert o.canonical_record_id.is_unique and set(o.canonical_record_id)==set(dom.canonical_record_id),'incomplete OOF coverage'
    o.to_csv(d/'oof_observations_predictions.csv',index=False);pm=dict(zip(o.canonical_record_id,o.pred_ln_gamma_inf))
    q=pr[pr.canonical_record_id_A.isin(pm)&pr.canonical_record_id_B.isin(pm)]
    Dx=q.D_AB.astype(float).values;Dh=q.canonical_record_id_A.map(pm).values-q.canonical_record_id_B.map(pm).values;m=np.abs(Dx)>=0.05
    rows.append({'seed':s,'status':'complete','lng_mae':float(np.abs(o.pred_ln_gamma_inf-o.ln_gamma_inf).mean()),'D_mae':float(np.abs(Dh-Dx).mean()),
                 'D_zero_mae':float(np.abs(Dx).mean()),'n_obs':int(len(o)),'n_pairs':int(len(q)),'sign_acc_|D|>=0.05':float((np.sign(Dh[m])==np.sign(Dx[m])).mean())})
pd.DataFrame(rows).to_csv(root/'GHGNN_SUMMARY.csv',index=False);print(json.dumps(rows,indent=1))
