"""Checks the cached molecular graphs against graphs built anew from SMILES, and the coverage of the folds."""
from pathlib import Path
import argparse,json,csv,hashlib
import numpy as np
from data import Dataset
from graphs import GraphBank,graph

def verify(path='data'):
    path=Path(path)
    with (path/'molecules.csv').open(encoding='utf-8-sig',newline='') as stream:
        molecules=list(csv.DictReader(stream))
    bank=GraphBank(path/'molecule_graphs.npz').arr
    for i,m in enumerate(molecules):
        fresh=graph(m['SMILES'])
        al,ah=bank['atom_ptr'][i:i+2];el,eh=bank['edge_ptr'][i:i+2]
        for key in ('atom','bond','src','dst','rev'):
            cached=bank[key][al:ah] if key=='atom' else bank[key][el:eh]
            if not np.array_equal(fresh[key],cached):
                raise ValueError('Graph features/cache differ for: '+m['Mol_ID']+' '+key)
        rev=fresh['rev']
        assert np.array_equal(rev[rev],np.arange(len(rev)))
        assert np.array_equal(fresh['src'][rev],fresh['dst'])
    ds=Dataset(path)
    cover=[]
    for fold in range(1,6):
        s=ds.split(fold)
        roles={role:{'observations':len(ix),'pairs':len(ds.pair_indices(ix))}
               for role,ix in s.items()}
        cover.append(roles)
        if any(roles[role]['pairs']==0 for role in roles):
            raise ValueError('Original pair coverage is zero in a fold role')
    report={            'matched_conditions':ds.matched_condition_validation,
            'molecules_and_graphs_verified':len(molecules),
            'observations':len(ds.records),'pairs':len(ds.pairs),
            'fold_coverage':cover}
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--data',default='data')
    p.add_argument('--verify',action='store_true')
    args=p.parse_args()
    verify(args.data)
