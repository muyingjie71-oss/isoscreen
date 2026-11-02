"""Check the split models against the production predictors on the 5,040-point check grid (5 solvents x 336 solutes x 3 T)."""
import sys, json, csv, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import MODELS, PARAMS as PAR, TRAINING_DATA
from graphs import graph
from split_models import DMPNNEncoder, DMPNNHead, dmpnn_graph_batch, GHGNNEncoder, GHGNNHead
from run_ghgnn_job import mol_cache, HP
import mol2graph as M
from GHGNN_architecture import GHGNN_model
import check_grid

SEEDS = ['seed_20261007', 'seed_20261008', 'seed_20261009']
rows = check_grid.grid(TRAINING_DATA)
mols = sorted({r['solute_smiles'] for r in rows} | {r['solvent_smiles'] for r in rows}); ix = {s: i for i, s in enumerate(mols)}
su = np.array([ix[r['solute_smiles']] for r in rows]); sv = np.array([ix[r['solvent_smiles']] for r in rows]); T = np.array([r['T_K'] for r in rows])

def ref(path):
    with open(path) as f: return {(r['solute_id'], r['solvent_id'], float(r['T_K'])): r for r in csv.DictReader(f)}
key = [(r['solute_id'], r['solvent_id'], float(r['T_K'])) for r in rows]

# ---------------- D-MPNN
t0 = time.time()
batch = dmpnn_graph_batch([graph(s) for s in mols])
out = {}
for s in SEEDS:
    p = dict(np.load(PAR / f'dmpnn_{s}.npz')); sc = json.loads((PAR / f'dmpnn_{s}.json').read_text())['scalers']
    H = DMPNNEncoder(p)(*batch); head = DMPNNHead(p)
    tz = torch.tensor(((np.float32(1000.0 / T).astype(np.float32) - sc['t_mean']) / sc['t_std']).astype(np.float32))
    # evaluate each row: head on (1 solute, 1 solvent, 1 T); vectorised by grouping rows
    hi = H[torch.tensor(su)]; hs = H[torch.tensor(sv)]
    ht = torch.nn.functional.silu(tz[:, None] @ head.Wt + head.bt)
    base = hi @ head.W_i + hs @ head.W_s + (hi - hs) @ head.W_d + (hi * hs) @ head.W_p + head.b1
    x = torch.nn.functional.silu(base + ht @ head.W_t); x = torch.nn.functional.silu(x @ head.W2 + head.b2)
    raw = (x @ head.W3 + head.b3)[:, 0].numpy().astype(np.float64)
    out[s] = raw * sc['y_std'] + sc['y_mean']
R = ref(MODELS / 'acceptance' / 'grid_dmpnn_cpu.csv')
d = max(abs(out[s][i] - float(R[k]['pred_' + s])) for s in SEEDS for i, k in enumerate(key))
dm = max(abs(np.mean([out[s][i] for s in SEEDS]) - float(R[k]['pred_mean'])) for i, k in enumerate(key))
print(f'D-MPNN split vs production: max |diff| per seed {d:.2e}, seed mean {dm:.2e}  ({time.time()-t0:.1f}s)')

# ---------------- GH-GNN
t0 = time.time()
C = mol_cache({s: s for s in mols})
xs, eas, rows_, cols_, us, gidx, egidx = [], [], [], [], [], [], []; na = 0
for gi, s in enumerate(mols):
    c = C[s]; n = c['x'].shape[0]; e = c['ei'].shape[1] if c['ei'].numel() else 0
    xs.append(c['x']); us.append([c['ap'], c['bp'], c['tpsa']]); gidx += [gi] * n
    if e:
        eas.append(c['ea']); rows_.append(c['ei'][0] + na); cols_.append(c['ei'][1] + na); egidx += [gi] * e
    na += n
x = torch.cat(xs); ea = torch.cat(eas); row = torch.cat(rows_); col = torch.cat(cols_); u = torch.tensor(us, dtype=torch.float32)
node_inc = torch.zeros(len(mols), na); node_inc[torch.tensor(gidx), torch.arange(na)] = 1
col_inc = torch.zeros(na, len(row)); col_inc[col, torch.arange(len(row))] = 1
eg_inc = torch.zeros(len(mols), len(row)); eg_inc[torch.tensor(egidx), torch.arange(len(row))] = 1
hba = np.array([C[s]['hba'] for s in mols]); hbd = np.array([C[s]['hbd'] for s in mols])
inter = np.minimum(hba[sv], hbd[su]) + np.minimum(hba[su], hbd[sv])
out = {}
for s in SEEDS:
    m = GHGNN_model(M.n_atom_features(), M.n_bond_features(), 3, HP['hidden_dim']); m.load_state_dict(torch.load(MODELS / 'ghgnn' / s / 'final.pt', map_location='cpu')); m.eval()
    Z = GHGNNEncoder(m)(x, ea, row, col, u, node_inc, col_inc, eg_inc)
    A, B = GHGNNHead(m)(Z[torch.tensor(sv)], Z[torch.tensor(su)], torch.tensor(inter, dtype=torch.float32))
    out[s] = (A + B / torch.tensor(T, dtype=torch.float32)).numpy().astype(np.float64)
R = ref(MODELS / 'acceptance' / 'grid_ghgnn_cpu.csv')
d = max(abs(out[s][i] - float(R[k]['pred_' + s])) for s in SEEDS for i, k in enumerate(key))
dm = max(abs(np.mean([out[s][i] for s in SEEDS]) - float(R[k]['pred_mean'])) for i, k in enumerate(key))
print(f'GH-GNN split vs production: max |diff| per seed {d:.2e}, seed mean {dm:.2e}  ({time.time()-t0:.1f}s)')
