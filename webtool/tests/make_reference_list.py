import sys, json
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
for p in ('training/dmpnn/shared', 'training/ghgnn', 'training/ghgnn/ghgnn'): sys.path.insert(0, str(REPO / p))
from graphs import graph
from run_ghgnn_job import mol_cache
from predict_ghgnn import supported
import warnings; warnings.filterwarnings('ignore')
S = [l.strip() for l in open(sys.argv[1]) if l.strip()]
n = 0
with open(sys.argv[2], 'w') as f:
    for s in S:
        try: g = graph(s)
        except Exception: continue
        rec = {'smiles': s, 'd': {k: v.tolist() for k, v in g.items()}}
        try:
            if supported(s):
                c = mol_cache({s: s})[s]
                rec['g'] = {'x': c['x'].tolist(), 'ea': c['ea'].tolist() if c['ea'].numel() else [], 'ei': c['ei'].tolist() if c['ei'].numel() else [[], []],
                            'ap': float(c['ap']), 'bp': float(c['bp']), 'tpsa': float(c['tpsa']), 'hba': int(c['hba']), 'hbd': int(c['hbd'])}
        except Exception as e: rec['g_error'] = str(e)[:80]
        f.write(json.dumps(rec) + '\n'); n += 1
print(n)
