"""One-head model metrics: absolute IDAC observations and matched-pair differences.

Both metrics are views of the SAME model's predictions, NOT two training heads.
"""
from collections import defaultdict
import numpy as np

def huber(e,delta=1.):
    ae=np.abs(e)
    return np.where(ae<=delta,.5*e*e,delta*(ae-.5*delta))

def macro(values,labels):
    bank=defaultdict(list)
    for v,k in zip(values,labels):bank[k].append(float(v))
    return float(np.mean([np.mean(x) for x in bank.values()])) if bank else None

def regression(obs,pred):
    obs=np.asarray(obs,np.float64);pred=np.asarray(pred,np.float64)
    e=pred-obs
    if not len(e):return {'n':0,'mae':None,'rmse':None,'r2':None}
    den=float(np.sum((obs-np.mean(obs))**2))
    tolerance=np.finfo(np.float64).eps*max(1.,float(np.sum(obs*obs)))
    return {'n':int(len(e)),'mae':float(np.mean(np.abs(e))),
            'rmse':float(np.sqrt(np.mean(e*e))),
            'r2':float(1-np.sum(e*e)/den) if den>tolerance else None}

def evaluate(ds,ids,pred,scalers,cfg):
    ids=np.asarray(ids,dtype=np.int32)
    pred=np.asarray(pred,np.float64)
    if len(ids)!=len(pred):raise ValueError('Record/prediction count mismatch')
    pi=ds.pair_indices(ids)
    lookup={int(i):float(v) for i,v in zip(ids,pred)}
    obs=ds.y64[ids]
    actual=ds.dy64[pi]
    pd=np.asarray([lookup[int(ds.pa[i])]-lookup[int(ds.pb[i])] for i in pi],np.float64)
    ae=np.abs(pred-obs);de=np.abs(pd-actual)
    families=[ds.records[i]['Family_ID'] for i in ids]
    pair_families=[ds.pairs[i]['Family_ID'] for i in pi]
    groups=[ds.pairs[i]['MatchedGroup_ID'] for i in pi]
    obs_metrics=regression(obs,pred)
    pair_metrics=regression(actual,pd)
    obs_metrics.update(family_macro_mae=macro(ae,families),supported_families=len(set(families)))
    zero=np.abs(actual)
    zero_mae=float(np.mean(zero)) if len(zero) else None
    zero_macro=macro(zero,pair_families)
    pair_metrics.update(family_macro_mae=macro(de,pair_families),
        group_macro_mae=macro(de,groups),
        supported_families=len(set(pair_families)),matched_groups=len(set(groups)),
        zero_diff_baseline_mae=zero_mae,
        zero_diff_family_macro_mae=zero_macro,
        mae_better_than_zero=(pair_metrics['mae']<zero_mae) if zero_mae is not None else None,
        family_macro_better_than_zero=(macro(de,pair_families)<zero_macro) if zero_macro is not None else None)
    threshold=float(cfg['sign_threshold'])
    mask=np.abs(actual)>=threshold
    pair_metrics.update(sign_threshold=threshold,sign_coverage=float(mask.mean()) if len(mask) else None,
        sign_n=int(mask.sum()),sign_accuracy=float(np.mean(np.sign(pd[mask])==np.sign(actual[mask]))) if mask.any() else None)
    abs_loss=float(np.mean(huber((pred-obs)/scalers['y_std'],cfg['huber_delta'])))
    diff_loss=macro(huber((pd-actual)/scalers['y_std'],cfg['huber_delta']),groups)
    score=abs_loss+cfg['validation_diff_weight']*(diff_loss if diff_loss is not None else 0.)
    relations={}
    for label in sorted(set(ds.pairs[i]['Isomer_Relation'] for i in pi)):
        sel=np.asarray([ds.pairs[i]['Isomer_Relation']==label for i in pi],dtype=bool)
        sse=de[sel];sactual=actual[sel]
        relatives=[pair_families[i] for i,k in enumerate(sel) if k]
        rel=regression(sactual,pd[sel]);rel.update(family_macro_mae=macro(sse,relatives),
            zero_diff_baseline_mae=float(np.abs(sactual).mean()) if len(sactual) else None)
        relations[label]=rel
    breakdown={}
    for fam in sorted(set(pair_families)):
        sel=np.asarray([k==fam for k in pair_families],dtype=bool)
        b=regression(actual[sel],pd[sel]);b['zero_diff_baseline_mae']=float(np.abs(actual[sel]).mean())
        breakdown[fam]=b
    return {'observations':obs_metrics,'isomer_pairs':pair_metrics,
            'absolute_huber':abs_loss,'group_macro_differential_huber':diff_loss,
            'validation_score':score,'relation_breakdown':relations,
            'pair_family_breakdown':breakdown},pi,pd
