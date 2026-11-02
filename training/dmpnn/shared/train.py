"""Train ONE chosen fold; joint loss on observations and matched isomer pairs."""
import argparse,json,yaml
from pathlib import Path
from trainer import train
if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--data',required=True)
    p.add_argument('--config',required=True)
    p.add_argument('--fold',type=int,choices=range(1,6),required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--smoke',action='store_true')
    args=p.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text(encoding='utf8'))
    result=train(cfg,args.data,Path(args.output)/('Fold_'+str(args.fold)),args.fold,args.resume,args.smoke)
    print(json.dumps(result,ensure_ascii=False,indent=2))
