"""D-MPNN inference from SMILES.

    P = DMPNNPredictor('models/dmpnn')          # folder with seed_*/{final_params.msgpack, scalers.json, config.json}
    ln_gamma = P.predict(solute_smiles, solvent_smiles, T_K)   # mean of the seed models

Graphs are built by graphs.graph(SMILES), the featuriser that wrote graphs_layer_a_v2.npz
(prepare_training_data.py --verify checks bit equality for all 981 dataset molecules).
A record's prediction does not depend on the other records in its batch (per-graph pooling, no dropout at inference).
Runs on CPU or GPU (JAX_PLATFORMS=cpu forces CPU)."""
import json
from pathlib import Path
import numpy as np
import jax
jax.config.update('jax_default_matmul_precision','highest')
from flax import serialization
from model import IDACModel
from graphs import graph,GraphBank

def bank_from_smiles(smiles):
    """In-memory GraphBank with the same layout as graphs.save_graphs."""
    items=[graph(s) for s in smiles];ap=[0];ep=[0]
    for g in items:ap.append(ap[-1]+len(g['atom']));ep.append(ep[-1]+len(g['src']))
    arr={k:np.concatenate([g[k] for g in items]) for k in ('atom','bond','src','dst','rev')}
    arr['atom_ptr']=np.asarray(ap,np.int32);arr['edge_ptr']=np.asarray(ep,np.int32)
    bank=GraphBank.__new__(GraphBank);bank.arr=arr
    return bank

class DMPNNPredictor:
    def __init__(self,model_dir):
        self.members=[]
        for d in sorted(Path(model_dir).glob('seed_*')):
            cfg=json.loads((d/'config.json').read_text(encoding='utf8'))
            scalers=json.loads((d/'scalers.json').read_text(encoding='utf8'))
            params=jax.device_put(serialization.msgpack_restore((d/'final_params.msgpack').read_bytes()))
            model=IDACModel(cfg['hidden'],cfg['depth'],cfg['dropout'],'A')
            fn=jax.jit(lambda p,b,model=model:model.apply({'params':p},b,training=False))
            self.members.append({'name':d.name,'params':params,'scalers':scalers,'fn':fn})
        if not self.members:raise FileNotFoundError('No seed_* model folders in '+str(model_dir))
        self.seeds=[m['name'] for m in self.members]

    def predict(self,solute_smiles,solvent_smiles,T_K,batch_size=512,per_seed=False):
        su=list(solute_smiles);sv=list(solvent_smiles);T=[float(x) for x in T_K];n=len(su)
        if not (len(sv)==n==len(T)):raise ValueError('solute, solvent and T_K must have the same length')
        if n==0:return ({'mean':np.zeros(0)} if per_seed else np.zeros(0))
        uniq=sorted(set(su)|set(sv));idx={s:i for i,s in enumerate(uniq)}
        bank=bank_from_smiles(uniq)
        sm=np.asarray([idx[s] for s in su],np.int32);vm=np.asarray([idx[s] for s in sv],np.int32)
        t=np.asarray([1000./x for x in T],np.float32)   # same as Dataset: 1000/T in float64, stored as float32
        out=np.zeros((len(self.members),n),np.float64)
        for lo in range(0,n,batch_size):
            hi=min(n,lo+batch_size);k=hi-lo
            pad=np.full(batch_size,lo,np.int64);pad[:k]=np.arange(lo,hi)
            mask=np.zeros(batch_size,np.float32);mask[:k]=1
            g,lookup=bank.pack(np.concatenate([sm[pad],vm[pad]]))
            for j,m in enumerate(self.members):
                sc=m['scalers']
                b={'graph':g,'solute_graph':lookup[:batch_size],'solvent_graph':lookup[batch_size:],
                   'temperature':((t[pad]-sc['t_mean'])/sc['t_std']).astype(np.float32),'record_mask':mask}
                v=np.asarray(m['fn'](m['params'],b).block_until_ready())[:k]
                if not np.isfinite(v).all():raise FloatingPointError('Nonfinite predictions')
                out[j,lo:hi]=v.astype(np.float64)*sc['y_std']+sc['y_mean']
        mean=out.mean(axis=0)
        if per_seed:return {'mean':mean,**{m['name']:out[j] for j,m in enumerate(self.members)}}
        return mean

def predict(solute_smiles,solvent_smiles,T_K,model_dir='models/dmpnn'):
    return DMPNNPredictor(model_dir).predict(solute_smiles,solvent_smiles,T_K)
