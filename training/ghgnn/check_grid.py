"""Fixed CPU/GPU check grid shared by both models: 5 training solvents
(3 ionic liquids and 2 molecular solvents with the most records, GH-GNN-supported elements only)
x every dataset solute x 3 temperatures. Deterministic, read from the frozen data only."""
import csv
from collections import Counter
from pathlib import Path

UNSUPPORTED = ('[Al', '[Co', '[Fe', '[Sb')
TEMPS = (298.15, 323.15, 373.15)

def _rows(p):
    with open(p, encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def grid(data_dir):
    d = Path(data_dir)
    smi = {r['Mol_ID']: r['SMILES'] for r in _rows(d / 'molecules.csv')}
    t1 = _rows(d / 'task1.csv')
    counts = Counter(r['Solvent_Mol_ID'] for r in t1)
    ok = lambda m: not any(u in smi[m] for u in UNSUPPORTED)
    ranked = sorted(counts, key=lambda m: (-counts[m], m))
    ils = [m for m in ranked if '.' in smi[m] and ok(m)][:3]
    mol = [m for m in ranked if '.' not in smi[m] and ok(m)][:2]
    solutes = sorted({r['Solute_Mol_ID'] for r in t1 if ok(r['Solute_Mol_ID'])})
    out = []
    for v in ils + mol:
        for u in solutes:
            for T in TEMPS:
                out.append({'solute_id': u, 'solvent_id': v, 'solute_smiles': smi[u], 'solvent_smiles': smi[v], 'T_K': T})
    return out

def write(path, rows, pred, seeds):
    cols = ['solute_id', 'solvent_id', 'T_K', 'pred_mean'] + [f'pred_{s}' for s in seeds]
    with open(path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f); w.writerow(cols)
        for i, r in enumerate(rows):
            w.writerow([r['solute_id'], r['solvent_id'], r['T_K'], repr(float(pred['mean'][i]))] + [repr(float(pred[s][i])) for s in seeds])

def max_diff(path_a, path_b):
    a, b = _rows(path_a), _rows(path_b)
    if len(a) != len(b) or any((x['solute_id'], x['solvent_id'], x['T_K']) != (y['solute_id'], y['solvent_id'], y['T_K']) for x, y in zip(a, b)):
        raise ValueError('Check grids differ in their rows')
    cols = [c for c in a[0] if c.startswith('pred_')]
    return max(abs(float(x[c]) - float(y[c])) for x, y in zip(a, b) for c in cols)
