"""Python reference features for the featuriser test: D-MPNN graphs.graph() and GH-GNN mol_cache for every dataset and library molecule."""
import sys, csv, json
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
for p in ('training/dmpnn/shared', 'training/ghgnn', 'training/ghgnn/ghgnn'): sys.path.insert(0, str(REPO / p))
from graphs import graph
from run_ghgnn_job import mol_cache
from predict_ghgnn import supported
S = [r['smiles'] for r in csv.DictReader(open(REPO / 'webtool/solvent_library/solvent_library.csv', encoding='utf-8-sig'))] + [r['SMILES'] for r in csv.DictReader(open(REPO / 'training/inputs/molecules_as_trained.csv', encoding='utf-8-sig'))]
S += sys.argv[2:]
S = list(dict.fromkeys(S))
with open(sys.argv[1], 'w') as f:
    for s in S:
        g = graph(s); rec = {'smiles': s, 'd': {k: v.tolist() for k, v in g.items()}}
        if supported(s):
            c = mol_cache({s: s})[s]
            rec['g'] = {'x': c['x'].tolist(), 'ea': c['ea'].tolist() if c['ea'].numel() else [], 'ei': c['ei'].tolist() if c['ei'].numel() else [[], []],
                        'ap': float(c['ap']), 'bp': float(c['bp']), 'tpsa': float(c['tpsa']), 'hba': int(c['hba']), 'hbd': int(c['hbd'])}
        f.write(json.dumps(rec) + '\n')
print(len(S))
