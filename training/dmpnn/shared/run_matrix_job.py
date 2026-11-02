"""One job = one (variant, seed, fold). Writes <out>/Fold_<k>/ exactly as trainer.train does."""
import argparse,json,yaml
from pathlib import Path
from trainer import train
p=argparse.ArgumentParser()
p.add_argument('--data',required=True);p.add_argument('--config',required=True)
p.add_argument('--seed',type=int,required=True);p.add_argument('--fold',type=int,choices=range(1,6),required=True)
p.add_argument('--out',required=True);p.add_argument('--smoke',action='store_true')
a=p.parse_args()
cfg=yaml.safe_load(Path(a.config).read_text(encoding='utf8'));cfg['seed']=a.seed
out=Path(a.out)/f'Fold_{a.fold}'
r=train(cfg,a.data,out,a.fold,resume=(out/'last.msgpack').exists(),smoke=a.smoke)
print(json.dumps({k:v for k,v in r.items() if k!='test'},ensure_ascii=False),flush=True)
print('JOB done:',a.config,a.seed,a.fold,flush=True)
