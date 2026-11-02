"""Full absolute epochs + group-uniform differential updates; resumable JAX."""
from pathlib import Path
from functools import partial
import json,csv,math,time,hashlib,os,platform
import numpy as np
import jax,jax.numpy as jnp,optax,flax
jax.config.update('jax_default_matmul_precision','highest')
from flax import serialization
from flax.training import train_state
from model import IDACModel
from losses import joint_loss
from data import Dataset,GroupSampler
from metrics import evaluate

class State(train_state.TrainState):rng:object

def dump(path,obj):
    p=Path(path);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8');tmp.replace(p)
def code_hashes():
    names=('graphs.py','model.py','losses.py','data.py','metrics.py','trainer.py','prepare_training_data.py','requirements.txt')
    return {name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in names}
def fingerprint(cfg,ds,fold):
    return hashlib.sha256(json.dumps({'cfg':cfg,'data':ds.fingerprint,'features':ds.feature_fingerprint,'fold':fold,'code':code_hashes()},sort_keys=True).encode()).hexdigest()

def save_checkpoint(path,state,progress,host_rng,scalers,run_id):
    """Atomic single file; optimizer, JAX/host RNG, permutation and cursor saved."""
    payload={'state':serialization.to_state_dict(state),'progress_json':json.dumps(progress),'host_rng_json':json.dumps(host_rng.bit_generator.state),'scalers_json':json.dumps(scalers),'run_id':run_id}
    p=Path(path);tmp=p.with_suffix('.tmp');tmp.write_bytes(serialization.msgpack_serialize(payload));tmp.replace(p)
def restore_checkpoint(path,target,host_rng,run_id):
    payload=serialization.msgpack_restore(Path(path).read_bytes())
    if payload['run_id']!=run_id:raise ValueError('Checkpoint configuration, dataset or feature fingerprint differs')
    host_rng.bit_generator.state=json.loads(payload['host_rng_json'])
    state=serialization.from_state_dict(target,payload['state'])
    return state,json.loads(payload['progress_json']),json.loads(payload['scalers_json'])

def make_state(cfg,batch,nsteps):
    key=jax.random.PRNGKey(cfg['seed']);initkey,key=jax.random.split(key)
    model=IDACModel(cfg['hidden'],cfg['depth'],cfg['dropout'])
    params=model.init(initkey,batch,training=False)['params']
    total=max(2,nsteps*cfg['max_epochs']);warmup=min(total-1,max(1,int(total*cfg['warmup_fraction'])))
    schedule=optax.warmup_cosine_decay_schedule(0.,cfg['learning_rate'],warmup,total,end_value=cfg['learning_rate']*.05)
    optimizer=optax.chain(optax.clip_by_global_norm(cfg['clip_norm']),optax.adamw(schedule,weight_decay=cfg['weight_decay']))
    state=State.create(apply_fn=model.apply,params=params,tx=optimizer,rng=key)
    @jax.jit
    def step(state,b):
        key,dropkey=jax.random.split(state.rng)
        def objective(params):
            pred=state.apply_fn({'params':params},b,training=True,rngs={'dropout':dropkey})
            return joint_loss(pred,b,cfg['lambda_diff'],cfg['huber_delta'])
        (loss,parts),grads=jax.value_and_grad(objective,has_aux=True)(state.params)
        state=state.apply_gradients(grads=grads).replace(rng=key)
        return state,{'loss':loss,**parts,'gradient_norm':optax.global_norm(grads)}
    @jax.jit
    def infer(params,b):return model.apply({'params':params},b,training=False)
    return state,step,infer

def predict(ds,ids,state,infer,scalers,batch_size):
    out=[]
    for lo in range(0,len(ids),batch_size):
        rows=ids[lo:lo+batch_size];b=ds.record_batch(rows,scalers,batch_size)
        v=np.asarray(infer(state.params,b).block_until_ready())[:len(rows)]
        if not np.isfinite(v).all():raise FloatingPointError('Nonfinite predictions')
        out.extend(v.astype(np.float64)*scalers['y_std']+scalers['y_mean'])
    return np.asarray(out,np.float64)

def write_predictions(out,ds,ids,pred,pi,pd,fold):
    with (out/'test_observations_predictions.csv').open('w',encoding='utf-8-sig',newline='') as f:
        cols=['canonical_record_id','Family_ID','Fold_ID','Solute','Solvent','T_K','gamma_inf','ln_gamma_inf','pred_ln_gamma_inf','Source','Original_Source_File','Original_Source_Location']
        w=csv.DictWriter(f,fieldnames=cols);w.writeheader()
        for i,p in zip(ids,pred):
            r=ds.records[i];w.writerow({**{c:r.get(c,'') for c in cols},'Fold_ID':f'Fold_{fold}','pred_ln_gamma_inf':repr(float(p))})
    with (out/'test_pairs_predictions.csv').open('w',encoding='utf-8-sig',newline='') as f:
        cols=['Pair_ID','MatchedGroup_ID','Family_ID','Fold_ID','canonical_record_id_A','canonical_record_id_B','Isomer_Relation','D_AB','pred_D_AB']
        w=csv.DictWriter(f,fieldnames=cols);w.writeheader()
        for i,p in zip(pi,pd):w.writerow({**{c:ds.pairs[i].get(c,'') for c in cols},'Fold_ID':f'Fold_{fold}','pred_D_AB':repr(float(p))})

def train(cfg,data_path,out_path,fold,resume=False,smoke=False,feature_file=None,fixed_epochs=None):
    if feature_file is not None:raise ValueError('Only 2D molecular graphs are supported')
    ds=Dataset(data_path);cfg=dict(cfg)
    # fixed_epochs: fold 0 = every record trains; stop after fixed_epochs; no validation or test.
    # max_epochs is unchanged, so warmup and cosine decay follow the same per-epoch curve as the cross-validation runs.
    production=fixed_epochs is not None
    if production:
        if fold!=0:raise ValueError('Production training uses fold 0 (all records)')
        allids=np.arange(len(ds.records),dtype=np.int32)
        split={'train':allids,'validation':allids[:0],'test':allids[:0]}
        cfg.update(fixed_epochs=int(fixed_epochs),production_full_data=True)
    else:split=ds.split(fold)
    train_all=split['train']  # same set ds.split(fold)['train'] gave below; captured before smoke truncation
    comparison=cfg.get('comparison','DMPNN')  # DMPNN (lambda_diff=1) or DMPNN_activities_only (lambda_diff=0), same data and splits
    if smoke:
        cfg.update(hidden=32,depth=2,absolute_batch=8,pair_batch=4,eval_batch=16,max_epochs=2,patience=2,checkpoint_interval_steps=4)
        # Smoke samples only from non-test families; never reports test performance.
        split['train']=split['train'][:64];split['validation']=split['validation'][:32]
        # Pairs selected separately below from the original train partition.
    out=Path(out_path);out.mkdir(parents=True,exist_ok=True)
    rid=fingerprint(cfg,ds,fold)
    if (out/'completed.json').exists():
        done=json.loads((out/'completed.json').read_text(encoding='utf8'))
        if done['run_id']!=rid:raise ValueError('Output directory belongs to a different run')
        print('Run already complete.',flush=True);return done
    if (out/'last.msgpack').exists() and not resume:raise ValueError('Existing run: use --resume or a fresh output directory')
    scalers=ds.scalers(train_all)
    pair_ids=ds.pair_indices(train_all);sampler=GroupSampler(ds.pairs,pair_ids)
    nsteps=math.ceil(len(split['train'])/cfg['absolute_batch'])
    rng=np.random.default_rng(cfg['seed']+fold)
    init_b=ds.train_batch(split['train'][:cfg['absolute_batch']],sampler.sample(rng,cfg['pair_batch']),scalers,cfg['absolute_batch'],cfg['pair_batch'])
    cfg['seed']+=fold;state,step,infer=make_state(cfg,init_b,nsteps)
    progress={'epoch':0,'cursor':0,'permutation':[],'best_score':None,'best_epoch':None,'bad_epochs':0,'history':[],'epoch_loss_sum':0.,'epoch_abs_sum':0.,'epoch_pair_sum':0.,'epoch_grad_sum':0.,'epoch_steps':0}
    if resume and (out/'last.msgpack').exists():state,progress,scalers=restore_checkpoint(out/'last.msgpack',state,rng,rid)
    dump(out/'run_manifest.json',{'run_id':rid,'configuration':cfg,'code_sha256':code_hashes(),'data_fingerprint':ds.fingerprint,'feature_fingerprint':ds.feature_fingerprint,'feature_names':ds.feature_names,'comparison':comparison,'cohort':ds.cohort,'initialization':'fresh per variant/fold; resume only same complete run fingerprint','fold':fold,'split':(ds.splits['rounds'][str(fold)] if not production else 'ALL_RECORDS_PRODUCTION'),'counts':{k:len(v) for k,v in split.items()},'training_pairs':len(pair_ids),'training_matched_groups':len(sampler.groups),'steps_per_epoch':nsteps,'epoch_definition':'Every training Task1 record exactly once; each optimizer step samples groups uniformly with replacement, then one pair uniformly per group.','smoke_only':smoke,'versions':{'jax':jax.__version__,'flax':flax.__version__,'optax':optax.__version__,'numpy':np.__version__,'python':platform.python_version()},'devices':[str(d) for d in jax.devices()]})
    dump(out/'scalers.json',scalers)
    started=time.perf_counter()
    stop_epoch=min(cfg['max_epochs'],cfg['fixed_epochs']) if production else cfg['max_epochs']
    while progress['epoch']<stop_epoch and (production or progress['bad_epochs']<cfg['patience']):
        if not progress['permutation']:progress['permutation']=rng.permutation(split['train']).tolist()
        epoch_t=time.perf_counter()
        while progress['cursor']<len(progress['permutation']):
            lo=progress['cursor'];absids=progress['permutation'][lo:lo+cfg['absolute_batch']]
            ps=sampler.sample(rng,cfg['pair_batch'])
            b=ds.train_batch(absids,ps,scalers,cfg['absolute_batch'],cfg['pair_batch'])
            state,parts=step(state,b);loss=float(parts['loss'].block_until_ready())
            grad=float(parts['gradient_norm'])
            absolute=float(parts['absolute_loss']);differential=float(parts['differential_loss'])
            if not np.isfinite([loss,grad,absolute,differential]).all():
                raise FloatingPointError('Training loss, component loss or gradient is nonfinite')
            progress['cursor']+=len(absids)
            progress['epoch_loss_sum']+=loss
            progress['epoch_abs_sum']+=absolute
            progress['epoch_pair_sum']+=differential
            progress['epoch_grad_sum']+=grad
            progress['epoch_steps']+=1
            if int(state.step)%cfg['checkpoint_interval_steps']==0:save_checkpoint(out/'last.msgpack',state,progress,rng,scalers,rid)
        if production:pred=None  # no validation set
        else:pred=predict(ds,split['validation'],state,infer,scalers,cfg['eval_batch'])
        if production:report={'validation_score':None}
        elif smoke:
            # Small validation smoke includes complete endpoint predictions for selected records only.
            mae=float(np.mean(np.abs(pred-ds.y64[split['validation']])));report={'validation_score':mae,'smoke_validation_mae':mae}
        else:report,_,_=evaluate(ds,split['validation'],pred,scalers,cfg)
        score=report['validation_score'];progress['epoch']+=1
        if production:pass  # no epoch selection
        elif progress['best_score'] is None or score<progress['best_score']:
            progress.update(best_score=score,best_epoch=progress['epoch'],bad_epochs=0)
            save_checkpoint(out/'best.msgpack',state,progress,rng,scalers,rid)
        else:progress['bad_epochs']+=1
        progress['history'].append({'epoch':progress['epoch'],'step':int(state.step),'train_loss_mean':progress['epoch_loss_sum']/progress['epoch_steps'],'train_absolute_loss_mean':progress['epoch_abs_sum']/progress['epoch_steps'],'train_differential_loss_mean':progress['epoch_pair_sum']/progress['epoch_steps'],'gradient_norm_mean':progress['epoch_grad_sum']/progress['epoch_steps'],'epoch_seconds':time.perf_counter()-epoch_t,'validation':report})
        if production:print(f"Production epoch {progress['epoch']}/{stop_epoch}: train_loss={progress['history'][-1]['train_loss_mean']:.6g}, {progress['history'][-1]['epoch_seconds']:.1f}s",flush=True)
        else:print(f"Fold {fold} epoch {progress['epoch']}: validation={score:.6g}, best={progress['best_score']:.6g}, {progress['history'][-1]['epoch_seconds']:.1f}s",flush=True)
        progress.update(cursor=0,permutation=[],epoch_loss_sum=0.,epoch_abs_sum=0.,epoch_pair_sum=0.,epoch_grad_sum=0.,epoch_steps=0)
        dump(out/'history.json',progress['history']);save_checkpoint(out/'last.msgpack',state,progress,rng,scalers,rid)
    if production:
        # export the parameters after the last epoch (params only, flax msgpack) with the scalers
        tmp=out/'final_params.msgpack.tmp';tmp.write_bytes(serialization.to_bytes(state.params));tmp.replace(out/'final_params.msgpack')
        result={'run_id':rid,'production':True,'smoke_only':smoke,'seed':cfg['seed'],'fixed_epochs':cfg['fixed_epochs'],'epochs':progress['epoch'],'steps':int(state.step),
                'final_params_sha256':hashlib.sha256((out/'final_params.msgpack').read_bytes()).hexdigest(),'scalers':scalers,'training_seconds':time.perf_counter()-started}
        dump(out/('smoke_result.json' if smoke else 'completed.json'),result);return result
    if smoke:
        result={'run_id':rid,'smoke_only':True,'epochs':progress['epoch'],'steps':int(state.step),'all_losses_finite':True,'test_evaluated':False}
        dump(out/'smoke_result.json',result);return result
    # Restore selected parameters once; deterministic test predictions reused for pairs.
    best,_,_=restore_checkpoint(out/'best.msgpack',state,rng,rid)
    pred=predict(ds,split['test'],best,infer,scalers,cfg['eval_batch'])
    report,pi,pd=evaluate(ds,split['test'],pred,scalers,cfg)
    write_predictions(out,ds,split['test'],pred,pi,pd,fold)
    result={'run_id':rid,'fold':fold,'best_epoch':progress['best_epoch'],'epochs_run':progress['epoch'],'training_seconds':time.perf_counter()-started,'test':report}
    dump(out/'completed.json',result);return result
