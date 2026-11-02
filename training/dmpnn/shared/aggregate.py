"""Pool the five fold outputs of each (variant, seed) into OOF tables + 5fold_report.json, then a cross-seed summary."""
import argparse,json,csv,yaml,statistics
from pathlib import Path
from run_5fold import run
p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--root',required=True)
p.add_argument('--configs',nargs='+',required=True);p.add_argument('--seeds',nargs='+',type=int,required=True)
a=p.parse_args();root=Path(a.root);rows=[]
for c in a.configs:
    cfg0=yaml.safe_load(Path(c).read_text(encoding='utf8'));v=cfg0['comparison']
    for s in a.seeds:
        out=root/f'{v}_seed{s}'
        if not all((out/f'Fold_{k}'/'completed.json').exists() for k in range(1,6)):
            rows.append({'variant':v,'seed':s,'status':'incomplete'});continue
        cfg=dict(cfg0,seed=s);m=run(cfg,a.data,out,resume=True)['pooled_out_of_family_predictions']
        o,pr=m['observations'],m['isomer_pairs']
        rows.append({'variant':v,'seed':s,'status':'complete','lng_mae':o['mae'],'lng_r2':o['r2'],'D_mae':pr['mae'],'D_family_macro_mae':pr['family_macro_mae'],
                     'D_zero_mae':pr['zero_diff_baseline_mae'],'D_r2':pr['r2'],'sign_acc_|D|>=0.05':pr['sign_accuracy'],'n_obs':o['n'],'n_pairs':pr['n']})
with (root/'MATRIX_SUMMARY.csv').open('w',newline='',encoding='utf-8-sig') as f:
    keys=sorted({k for r in rows for k in r},key=lambda k:(k not in('variant','seed','status'),k))
    w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
agg={}
for v in sorted({r['variant'] for r in rows}):
    done=[r for r in rows if r['variant']==v and r['status']=='complete']
    agg[v]={'seeds_complete':len(done)}
    for k in ('lng_mae','D_mae','D_family_macro_mae','sign_acc_|D|>=0.05','D_zero_mae'):
        x=[r[k] for r in done]
        if x:agg[v][k]={'mean':statistics.fmean(x),'sd':statistics.stdev(x) if len(x)>1 else None}
(root/'MATRIX_SUMMARY.json').write_text(json.dumps({'runs':rows,'by_variant':agg},ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(agg,ensure_ascii=False,indent=2))
