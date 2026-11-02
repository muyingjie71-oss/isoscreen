"""One production job = one seed: every record trains, stop after --fixed_epochs, export final_params.msgpack.
Same network, loss, optimiser and learning-rate schedule (max_epochs unchanged) as the paper runs. Writes into --out directly."""
import argparse,json,yaml
from pathlib import Path
from trainer import train
p=argparse.ArgumentParser()
p.add_argument('--data',required=True);p.add_argument('--config',required=True)
p.add_argument('--seed',type=int,required=True);p.add_argument('--fixed_epochs',type=int,required=True)
p.add_argument('--out',required=True);p.add_argument('--smoke',action='store_true')
a=p.parse_args()
cfg=yaml.safe_load(Path(a.config).read_text(encoding='utf8'));cfg['seed']=a.seed
out=Path(a.out)
r=train(cfg,a.data,out,0,resume=(out/'last.msgpack').exists(),smoke=a.smoke,fixed_epochs=a.fixed_epochs)
print(json.dumps({k:v for k,v in r.items() if k!='scalers'},ensure_ascii=False),flush=True)
print('JOB done:',a.config,a.seed,'production',flush=True)
