"""GH-GNN inference from SMILES.

    P = GHGNNPredictor('models/ghgnn')            # folder with seed_*/final.pt
    ln_gamma = P.predict(solute_smiles, solvent_smiles, T_K)   # mean of the seed models

Graphs and descriptors come from run_ghgnn_job.mol_cache/build, which tests/test_equivalence.py checks
bit for bit against upstream mol2graph.sys2graph. Temperature is passed in deg C inside build(), as in training.
Molecules with Al, Co, Fe or Sb are outside the featuriser (supported() is False) and raise ValueError."""
import sys
from pathlib import Path
import numpy as np,torch
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE/'ghgnn'))
from rdkit import Chem
import mol2graph as M
from GHGNN_architecture import GHGNN_model
from run_ghgnn_job import HP,mol_cache,build,loader,predict as _predict

def supported(smiles):
    m=Chem.MolFromSmiles(smiles)
    if m is None or m.GetNumAtoms()==0:return False
    try:
        for x in m.GetAtoms():M.atom_features(x)
        for x in m.GetBonds():M.bond_features(x)
        return True
    except Exception:return False

class GHGNNPredictor:
    def __init__(self,model_dir,device=None):
        self.dev=torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.members=[]
        for d in sorted(Path(model_dir).glob('seed_*')):
            m=GHGNN_model(M.n_atom_features(),M.n_bond_features(),3,HP['hidden_dim'])
            m.load_state_dict(torch.load(d/'final.pt',map_location='cpu'));m.to(self.dev).eval()
            self.members.append((d.name,m))
        if not self.members:raise FileNotFoundError('No seed_* model folders in '+str(model_dir))
        self.seeds=[n for n,_ in self.members]

    def predict(self,solute_smiles,solvent_smiles,T_K,batch_size=512,per_seed=False):
        su=list(solute_smiles);sv=list(solvent_smiles);T=[float(x) for x in T_K];n=len(su)
        if not (len(sv)==n==len(T)):raise ValueError('solute, solvent and T_K must have the same length')
        uniq=sorted(set(su)|set(sv))
        bad=[s for s in uniq if not supported(s)]
        if bad:raise ValueError('Outside the GH-GNN featuriser (Al/Co/Fe/Sb or invalid SMILES): '+'; '.join(bad[:5]))
        C=mol_cache({s:s for s in uniq})
        dl=loader(build(list(zip(sv,su,[0.0]*n,T)),C),batch_size,False)
        out=np.stack([_predict(m,dl,self.dev) for _,m in self.members])
        if out.shape[1]!=n or not np.isfinite(out).all():raise FloatingPointError('Missing or nonfinite predictions')
        mean=out.mean(axis=0)
        if per_seed:return {'mean':mean,**{name:out[j] for j,(name,_) in enumerate(self.members)}}
        return mean

def predict(solute_smiles,solvent_smiles,T_K,model_dir='models/ghgnn'):
    return GHGNNPredictor(model_dir).predict(solute_smiles,solvent_smiles,T_K)
