"""Small training step, joint loss, optimiser and exact resume from a checkpoint, with JAX/Flax.

Usage: python training/dmpnn/tests/test_jax_training.py [training/data/new_isomer_test]
"""
import sys,json,tempfile
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent.parent
DATA=Path(sys.argv[1]) if len(sys.argv)>1 else HERE.parent/'data'/'new_isomer_test'
sys.path.insert(0,str(HERE/'shared'))
import jax,jax.numpy as jnp
import yaml
from data import Dataset,GroupSampler
from trainer import make_state,save_checkpoint,restore_checkpoint
from losses import joint_loss

def test():
    cfg=yaml.safe_load((HERE/'configs/dmpnn.yaml').read_text(encoding='utf8'))
    cfg.update(hidden=32,depth=2,max_epochs=2,absolute_batch=8,pair_batch=4,eval_batch=8)
    ds=Dataset(DATA);fold=1
    sp=ds.split(fold);sc=ds.scalers(sp['train'])
    sampler=GroupSampler(ds.pairs,ds.pair_indices(sp['train']))
    rng=np.random.default_rng(51)
    ids=sp['train'][:8];pairs=sampler.sample(rng,4)
    b=ds.train_batch(ids,pairs,sc,8,4)
    state,step,infer=make_state(cfg,b,2)
    p=infer(state.params,b)
    assert p.shape==(16,) and jnp.isfinite(p).all()
    loss,parts=joint_loss(p,b,lam=cfg['lambda_diff'],delta=cfg['huber_delta'])
    np.testing.assert_allclose(np.array(loss),np.array(parts['absolute_loss']+cfg['lambda_diff']*parts['differential_loss']),rtol=1e-6)
    assert float(loss)>0
    nxt,values=step(state,b)
    assert int(nxt.step)==int(state.step)+1
    assert np.isfinite(float(values['loss'])) and np.isfinite(float(values['gradient_norm']))
    assert float(values['gradient_norm'])>0
    with tempfile.TemporaryDirectory() as tmp:
        cp=Path(tmp)/'last.msgpack'
        progress={'epoch':0,'cursor':8,'permutation':[int(x) for x in ids],'best_score':None}
        rid='resume-check'
        save_checkpoint(cp,nxt,progress,rng,sc,rid)
        target,_,_=make_state(cfg,b,2)
        rng2=np.random.default_rng(612)
        restored,progress2,sc2=restore_checkpoint(cp,target,rng2,rid)
        assert progress2==progress and sc2==sc
        assert rng2.bit_generator.state==rng.bit_generator.state
        params1=jax.tree_util.tree_leaves(nxt.params)
        params2=jax.tree_util.tree_leaves(restored.params)
        for a,c in zip(params1,params2):np.testing.assert_array_equal(np.asarray(a),np.asarray(c))
        # Resume from identical optimizer and RNG states must give identical next step.
        again1,_=step(nxt,b);again2,_=step(restored,b)
        for x,y in zip(jax.tree_util.tree_leaves(again1.params),jax.tree_util.tree_leaves(again2.params)):
            np.testing.assert_allclose(np.asarray(x),np.asarray(y),atol=1e-7,rtol=1e-6)
        try:restore_checkpoint(cp,target,rng2,'different-run')
        except ValueError:pass
        else:raise AssertionError('Wrong-fingerprint checkpoint accepted')
    print(json.dumps({'status':'PASS','jax_devices':[str(k) for k in jax.devices()],
                      'joint_loss':'same head, two supervision terms',
                      'training_step':'PASS','gradient':'finite nonzero',
                      'resume_parameters_optimizer_and_rng':'PASS'},ensure_ascii=False,indent=2))
if __name__=='__main__':test()
