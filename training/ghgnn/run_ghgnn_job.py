"""GH-GNN (IL-extended, Sanchez Medina & Minor, MIT) retrained from scratch on one fold.
Same protocol as the D-MPNN runs: train families -> fit; validation families -> pick best epoch; test families -> predicted once.
Published hyperparameters: hidden 113, lr 2.532501358651798e-4, batch 32, AdamW, ReduceLROnPlateau(0.8, patience 3) on train MAE, <=250 epochs.
Adapter changes (documented): cached graph construction (bit-identical to upstream sys2graph, see tests), torch_scatter -> scatter_compat,
T passed in deg C so the model's internal T+273.15 is kelvin, early stopping patience 30 on validation MAE."""
import argparse,json,csv,time,random,hashlib,os,sys
from pathlib import Path
import numpy as np,pandas as pd,torch
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE/'ghgnn'))
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from mordred.Polarizability import APol,BPol
from mordred.TopoPSA import TopoPSA
from torch_geometric.data import Data
import mol2graph as M
from GHGNN_architecture import GHGNN_model,count_parameters
HP={'hidden_dim':113,'lr':0.0002532501358651798,'max_epochs':250,'batch_size':32,'patience':30}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def mol_cache(smiles):
    out={}
    for mid,s in smiles.items():
        m=Chem.MolFromSmiles(s)
        node=np.array([M.atom_features(a) for a in m.GetAtoms()])
        ei=np.array(M.get_bond_pair(m)); ea=[]
        for b in m.GetBonds():ea.append(M.bond_features(b));ea.append(M.bond_features(b))
        out[mid]=dict(x=torch.tensor(node,dtype=torch.float),ei=torch.tensor(ei,dtype=torch.long),ea=torch.tensor(np.array(ea),dtype=torch.float),
            ap=APol()(m),bp=BPol()(m),tpsa=TopoPSA()(m),hba=rdMolDescriptors.CalcNumHBA(m),hbd=rdMolDescriptors.CalcNumHBD(m))
    return out
def build(rows,C):
    """Same Data objects as upstream mol2graph.sys2graph, assembled from per-molecule caches."""
    gs,gu,gt=[],[],[]
    for solv,solu,y,T in rows:
        a,b=C[solv],C[solu]
        inter=min(a['hba'],b['hbd'])+min(b['hba'],a['hbd'])
        yt=torch.tensor(y,dtype=torch.float)
        gs.append(Data(x=a['x'],edge_index=a['ei'],edge_attr=a['ea'],ap=a['ap'],bp=a['bp'],topopsa=a['tpsa'],hb=min(a['hba'],a['hbd']),inter_hb=inter,y=yt))
        gu.append(Data(x=b['x'],edge_index=b['ei'],edge_attr=b['ea'],ap=b['ap'],bp=b['bp'],topopsa=b['tpsa'],hb=min(b['hba'],b['hbd']),inter_hb=inter,y=yt))
        gt.append(Data(x=torch.tensor(T-273.15,dtype=torch.float).reshape(1)))
    return gs,gu,gt
def loader(g,bs,shuffle,gen=None,drop_last=False):
    return torch.utils.data.DataLoader(M.PairDataset_T(*g),batch_size=bs,shuffle=shuffle,drop_last=drop_last,collate_fn=M.collate_T,generator=gen)
def predict(model,dl,dev):
    model.eval();out=[]
    with torch.no_grad():
        for s,u,T in dl:out.append(model(s.to(dev),u.to(dev),T.to(dev)).reshape(-1).cpu().numpy())
    return np.concatenate(out).astype(np.float64)
def main():
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--fold',type=int,required=True);p.add_argument('--out',required=True);p.add_argument('--smoke',action='store_true')
    p.add_argument('--fixed_epochs',type=int)  # fold 0 = every record trains; stop after this many epochs; no validation/test
    a=p.parse_args();production=a.fixed_epochs is not None
    if production and a.fold!=0:raise ValueError('Production training uses --fold 0 (all records)')
    out=Path(a.out)/f'Fold_{a.fold}';out.mkdir(parents=True,exist_ok=True);D=Path(a.data)
    if (out/'completed.json').exists():print('Already complete');print('JOB done:',a.seed,a.fold);return
    t=pd.read_csv(D/'task1.csv',dtype={'canonical_record_id':str},keep_default_na=False)
    mol=pd.read_csv(D/'molecules.csv');smi=dict(zip(mol.Mol_ID,mol.SMILES))
    # GH-GNN applicability domain: its one-hot featuriser has no Al/Co/Fe/Sb (4 metal-anion ionic liquids, 0.33% of records).
    def supported(s):
        m=Chem.MolFromSmiles(s)
        try:
            for x in m.GetAtoms():M.atom_features(x)
            for x in m.GetBonds():M.bond_features(x)
            return True
        except Exception:return False
    unsupported=sorted(k for k in set(t.Solute_Mol_ID)|set(t.Solvent_Mol_ID) if not supported(smi[k]))
    t=t[~(t.Solute_Mol_ID.isin(unsupported)|t.Solvent_Mol_ID.isin(unsupported))].reset_index(drop=True)
    rd=json.loads((D/'split_manifest.json').read_text())['rounds'][str(a.fold if not production else 1)]  # round unused when production
    # a round lists whole families (original) or whole solvents (train_solvents/...)
    key,field=(('_solvents','Solvent_Mol_ID') if 'train_solvents' in rd else ('_families','Family_ID'))
    roles={r:np.flatnonzero(t[field].isin(set(rd[r+key])).values) for r in ('train','validation','test')}
    if production:roles={'train':np.arange(len(t)),'validation':np.arange(0),'test':np.arange(0)}
    hp=dict(HP)
    if a.smoke:
        hp.update(max_epochs=2,patience=2);roles['train']=roles['train'][:256];roles['validation']=roles['validation'][:128];roles['test']=roles['test'][:0]
    torch.set_num_threads(int(os.environ.get('GHGNN_THREADS','1')))
    random.seed(a.seed+a.fold);np.random.seed(a.seed+a.fold);torch.manual_seed(a.seed+a.fold)
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    used=set(t.Solute_Mol_ID)|set(t.Solvent_Mol_ID);C=mol_cache({k:smi[k] for k in used})
    rows=list(zip(t.Solvent_Mol_ID,t.Solute_Mol_ID,t.ln_gamma_inf.astype(float),t.T_K.astype(float)))
    G={r:build([rows[i] for i in ix],C) for r,ix in roles.items()}
    gen=torch.Generator();gen.manual_seed(a.seed+a.fold)
    tr=loader(G['train'],hp['batch_size'],True,gen,drop_last=True)
    tr_eval=loader(G['train'],512,False);va=loader(G['validation'],512,False) if not production else None
    model=GHGNN_model(M.n_atom_features(),M.n_bond_features(),3,hp['hidden_dim']).to(dev)
    opt=torch.optim.AdamW(model.parameters(),lr=hp['lr'])
    sch=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min',factor=0.8,patience=3,min_lr=1e-7)
    mse=torch.nn.MSELoss();state={'epoch':0,'best':None,'best_epoch':None,'bad':0,'history':[]}
    if (out/'last.pt').exists():
        # [R3] load to the host: load_state_dict copies weights and optimiser state to the parameters' device,
        # while RNG states must stay CPU ByteTensors (map_location=cuda broke set_rng_state on resume in R2)
        ck=torch.load(out/'last.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(ck['model']);opt.load_state_dict(ck['opt']);sch.load_state_dict(ck['sch']);state=ck['state']
        cpu_u8=lambda x:torch.as_tensor(x).to('cpu',torch.uint8)
        gen.set_state(cpu_u8(ck['gen']));torch.set_rng_state(cpu_u8(ck['torch_rng']));np.random.set_state(ck['np_rng'])
        if torch.cuda.is_available() and ck.get('cuda_rng') is not None:torch.cuda.set_rng_state_all([cpu_u8(x) for x in ck['cuda_rng']])
        print('resumed at epoch',state['epoch'],flush=True)
    (out/'run_manifest.json').write_text(json.dumps({'model':'GH-GNN (IL-extended), retrained from scratch','hyperparameters':hp,'production_fixed_epochs':a.fixed_epochs,'seed':a.seed,'fold':a.fold,
        'counts':{k:int(len(v)) for k,v in roles.items()},'excluded_unsupported_molecules':unsupported,'parameters':count_parameters(model),
        'torch':torch.__version__,'device':str(dev)},indent=1))
    yva=np.array([rows[i][2] for i in roles['validation']]);ytr=np.array([rows[i][2] for i in roles['train']])
    t0=time.time()
    stop_epoch=min(hp['max_epochs'],a.fixed_epochs) if production else hp['max_epochs']
    while state['epoch']<stop_epoch and (production or state['bad']<hp['patience']):
        model.train();ls=0;n=0;te=time.time()
        for s,u,T in tr:
            s,u,T=s.to(dev),u.to(dev),T.to(dev)
            y=s.y.to(torch.float32).clone()  # read labels before forward: an all-water (bond-free) batch makes upstream append a dummy graph to s in place
            pred=model(s,u,T).to(torch.float32);loss=mse(pred,y.reshape(pred.shape))
            opt.zero_grad();loss.backward();opt.step();ls+=loss.item()*s.num_graphs;n+=s.num_graphs
        mae_tr=float(np.abs(predict(model,tr_eval,dev)-ytr).mean());sch.step(mae_tr)
        mae_va=float(np.abs(predict(model,va,dev)-yva).mean()) if not production else float('nan');state['epoch']+=1
        if not production and not np.isfinite(mae_va):raise FloatingPointError('nonfinite validation')
        if not np.isfinite(mae_tr):raise FloatingPointError('nonfinite training MAE')
        if production:pass  # no epoch selection
        elif state['best'] is None or mae_va<state['best']:
            state.update(best=mae_va,best_epoch=state['epoch'],bad=0);torch.save(model.state_dict(),out/'best.pt')
        else:state['bad']+=1
        state['history'].append({'epoch':state['epoch'],'train_loss':ls/n,'train_mae':mae_tr,'val_mae':mae_va if not production else None,'lr':opt.param_groups[0]['lr'],'seconds':time.time()-te})
        torch.save({'model':model.state_dict(),'opt':opt.state_dict(),'sch':sch.state_dict(),'state':state,'gen':gen.get_state(),
                    'torch_rng':torch.get_rng_state(),'np_rng':np.random.get_state(),'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},out/'last.pt.tmp')
        os.replace(out/'last.pt.tmp',out/'last.pt')
        (out/'history.json').write_text(json.dumps(state['history'],indent=1))
        if production:print(f"Production epoch {state['epoch']}/{stop_epoch}: train_mae={mae_tr:.4f} lr={opt.param_groups[0]['lr']:.3g} {time.time()-te:.1f}s",flush=True)
        else:print(f"Fold {a.fold} epoch {state['epoch']}: train_mae={mae_tr:.4f} val_mae={mae_va:.4f} best={state['best']:.4f} {time.time()-te:.1f}s",flush=True)
    if production:
        # export the state_dict after the last epoch
        torch.save(model.state_dict(),out/'final.pt.tmp');os.replace(out/'final.pt.tmp',out/'final.pt')
        res={'production':True,'smoke_only':a.smoke,'seed':a.seed,'fixed_epochs':a.fixed_epochs,'epochs':state['epoch'],'final_train_mae':state['history'][-1]['train_mae'],
             'final_lr':opt.param_groups[0]['lr'],'final_sha256':sha(out/'final.pt'),'excluded_unsupported_molecules':unsupported,'n_train':int(len(roles['train'])),'training_seconds':time.time()-t0}
        (out/('smoke_result.json' if a.smoke else 'completed.json')).write_text(json.dumps(res,indent=1));print(json.dumps(res),flush=True);print('JOB done:',a.seed,a.fold,flush=True);return
    if a.smoke:
        (out/'smoke_result.json').write_text(json.dumps({'epochs':state['epoch'],'best':state['best'],'finite':True}));print('JOB done:',a.seed,a.fold);return
    model.load_state_dict(torch.load(out/'best.pt',map_location='cpu'));te_ix=roles['test']
    pred=predict(model,loader(G['test'],512,False),dev)
    cols=['canonical_record_id','Family_ID','Solute','Solvent','T_K','ln_gamma_inf']
    df=t.iloc[te_ix][cols].copy();df.insert(2,'Fold_ID',f'Fold_{a.fold}');df['pred_ln_gamma_inf']=pred
    df.to_csv(out/'test_observations_predictions.csv',index=False)
    res={'fold':a.fold,'seed':a.seed,'best_epoch':state['best_epoch'],'epochs_run':state['epoch'],'best_val_mae':state['best'],
         'test_mae':float(np.abs(pred-df.ln_gamma_inf.astype(float).values).mean()),'training_seconds':time.time()-t0}
    (out/'completed.json').write_text(json.dumps(res,indent=1));print(json.dumps(res),flush=True);print('JOB done:',a.seed,a.fold,flush=True)
if __name__=='__main__':main()
