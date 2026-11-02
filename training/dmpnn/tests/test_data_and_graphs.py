"""Checks of the training data and the graph encoder; needs no Flax or GPU.

Usage: python training/dmpnn/tests/test_data_and_graphs.py --data training/data/new_isomer_test
"""
import argparse,json,sys
from pathlib import Path
import numpy as np
from rdkit import Chem
HERE=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(HERE/'shared'))
from data import Dataset,GroupSampler
from graphs import graph,GraphBank
from metrics import evaluate

def main(data):
    ds=Dataset(data)
    exp=ds.info['expected_counts']
    assert len(ds.records)==exp['task1'] and len(ds.pairs)==exp['task2']
    assert len(ds.splits['outer_assignment'])==84
    assert ds.matched_condition_validation['matched_groups_checked']==exp['matched_groups']
    x=graph('C[C@H](F)Cl');y=graph('C[C@@H](F)Cl')
    assert not np.array_equal(x['atom'],y['atom']), 'Chiral stereoisomer atom features collapsed'
    ez=graph('F/C=C/F');zz=graph('F/C=C\\F')
    assert not np.array_equal(ez['bond'],zz['bond']), 'E/Z stereo bond features collapsed'
    ring5=graph('C1CCCC1');ring6=graph('C1CCCCC1')
    assert ring5['atom'].shape[0]==5 and ring6['atom'].shape[0]==6
    # Unlike old mean-only pooling, exact atom-count and sum can distinguish ring size.
    assert ring5['atom'].shape[1]==ring6['atom'].shape[1] and ring5['atom'].shape[1]>43
    atoms_charged=graph('[Na+].[Cl-]');assert len(atoms_charged['atom'])==2 and len(atoms_charged['src'])==0
    assert np.any(atoms_charged['atom'][0]!=atoms_charged['atom'][1])
    bank=GraphBank(Path(data)/'molecule_graphs.npz')
    for mid in ([0],[0,1,2],[0,1,2,4,8],[5,5,10,5]):
        batch,lookup=bank.pack(mid)
        m=int(batch['edge_mask'].sum());n=int(batch['node_mask'].sum());
        if m:
            rev=batch['rev'][:m];src=batch['src'][:m];dst=batch['dst'][:m]
            assert np.array_equal(rev[rev],np.arange(m))
            assert np.array_equal(src[rev],dst)
            assert np.all(src<n) and np.all(dst<n)
        assert len(lookup)==len(mid)
        assert int(batch['graph_mask'].sum())==len(set(mid))
    covered=[]
    for fold in range(1,6):
        spl=ds.split(fold)
        folds=set(ds.records[i]['Family_ID'] for i in spl['test'])
        assert folds==set(ds.splits['rounds'][str(fold)]['test_families'])
        assert all(len(ds.pair_indices(spl[role]))>0 for role in spl)
        covered.extend(spl['test'].tolist())
        scaler=ds.scalers(spl['train'])
        sampler=GroupSampler(ds.pairs,ds.pair_indices(spl['train']))
        rng=np.random.default_rng(271+fold)
        pairs=sampler.sample(rng,8)
        ids=spl['train'][:11]
        batch=ds.train_batch(ids,pairs,scaler,16,8)
        assert np.array_equal(batch['record_mask'][:11],np.ones(11))
        assert batch['record_mask'][len(np.unique(np.r_[ids,ds.pa[pairs],ds.pb[pairs]])):].sum()==0
        np.testing.assert_allclose(batch['absolute_y'][:11],(ds.y64[ids]-scaler['y_mean'])/scaler['y_std'],atol=2e-7)
        np.testing.assert_allclose(batch['pair_y'],ds.dy64[pairs]/scaler['y_std'],atol=2e-7)
        assert batch['record_mask'].shape==(32,)
    assert sorted(covered)==list(range(len(ds.records)))
    for source in (ds.records,ds.pairs):
        assert len(set(row['Family_ID'] for row in source))>10
    report={'status':'PASS','observations':len(ds.records),'pairs':len(ds.pairs),
            'matched_groups':ds.matched_condition_validation['matched_groups_checked'],
            'n_families':84,'molecules':len(ds.molecules),'folds':5,
            'chiral_and_EZ_graph_features':'PASS',
            'graph_cache_reverse_edges_and_padding':'PASS',
            'split_and_training_batches':'PASS'}
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return report
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',default=str(HERE.parent/'data'/'new_isomer_test'))
    args=p.parse_args();main(args.data)
