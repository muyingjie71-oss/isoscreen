"""Five unchanged original family-fold rounds; one out-of-family prediction per record."""
import argparse,json,csv
from pathlib import Path
import numpy as np,yaml
from trainer import train,dump,code_hashes
from data import Dataset,csv_rows
from metrics import evaluate

def run(cfg,data_path,output_path,resume=False):
    output_path=Path(output_path)
    reports=[]
    for fold in range(1,6):
        reports.append(train(cfg,data_path,output_path/f'Fold_{fold}',fold,resume,False))
    ds=Dataset(data_path,'A')
    predictions={};allrows=[]
    for fold in range(1,6):
        rows=csv_rows(output_path/f'Fold_{fold}/test_observations_predictions.csv')
        for row in rows:
            key=row['canonical_record_id']
            if key in predictions:raise ValueError('Repeated out-of-family experiment prediction')
            predictions[key]=float(row['pred_ln_gamma_inf'])
        allrows.extend(rows)
    if set(predictions)!=set(ds.ids):raise ValueError('Incomplete out-of-family prediction coverage')
    pred=np.asarray([predictions[r['canonical_record_id']] for r in ds.records],dtype=np.float64)
    ids=np.arange(len(pred))
    combined,_,_=evaluate(ds,ids,pred,{'y_std':1.},cfg)
    for name in ('validation_score','absolute_huber','group_macro_differential_huber'):
        combined.pop(name)
    meta={'variant':'A','configuration':cfg,'code_sha256':code_hashes(),
          'data_fingerprint':ds.fingerprint,
          'split_sha256':ds.freeze['files']['split_manifest.json'],
          'seed':cfg['seed'],'outer_folds':reports,
          'pooled_out_of_family_predictions':combined,
          'n_observations_oof':len(pred),'n_pairs_oof':len(ds.pairs),
          'pair_supported_families':len(set(p['Family_ID'] for p in ds.pairs))}
    dump(output_path/'5fold_report.json',meta)
    with (output_path/'oof_observations_predictions.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(allrows[0]));w.writeheader();w.writerows(allrows)
    pairrows=[]
    for fold in range(1,6):
        pairrows.extend(csv_rows(output_path/f'Fold_{fold}/test_pairs_predictions.csv'))
    if len(pairrows)!=len(ds.pairs) or len({r['Pair_ID'] for r in pairrows})!=len(ds.pairs):
        raise ValueError('Incomplete/repeated out-of-family pair predictions')
    with (output_path/'oof_pairs_predictions.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(pairrows[0]));w.writeheader();w.writerows(pairrows)
    print(json.dumps({'folds_completed':5,'n_observations':len(pred),'n_pairs':len(pairrows),
                      'observations':combined['observations'], 'pairs':combined['isomer_pairs']},
                     ensure_ascii=False,indent=2))
    return meta

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--data',required=True)
    p.add_argument('--config',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text(encoding='utf8'))
    run(cfg,args.data,args.output,args.resume)
