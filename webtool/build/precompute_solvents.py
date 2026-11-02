"""Encode every library solvent once with both models (3 seeds each) and write browser assets.

Outputs (in <out>):
  solvents.json            metadata in library order (+ hba, hbd for GH-GNN inter_hb)
  dmpnn_solvents.f32/.f16 encodings, little-endian, shape (3, S, 256); the site stores the .f16 copy
  ghgnn_solvents.f32/.f16 encodings, little-endian, shape (3, S, 226)
  solvents_manifest.json   number of solvents, seeds and array shapes
Usage: python precompute_solvents.py [solvent_library.csv] [output folder]
"""
import sys, json, csv
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import MODELS, PARAMS as PAR, LIBRARY, ASSETS
from graphs import graph
from split_models import DMPNNEncoder, dmpnn_graph_batch, GHGNNEncoder
from run_ghgnn_job import mol_cache, HP
import mol2graph as M
from GHGNN_architecture import GHGNN_model
torch.set_num_threads(2)
LIB = Path(sys.argv[1]) if len(sys.argv) > 1 else LIBRARY / 'solvent_library.csv'
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else ASSETS; OUT.mkdir(parents=True, exist_ok=True)
SEEDS = ['seed_20261007', 'seed_20261008', 'seed_20261009']
with open(LIB, encoding='utf-8') as f: lib = list(csv.DictReader(f))
smiles = [r['smiles'] for r in lib]; S = len(smiles)

def ghgnn_batch(C, mols):
    xs, eas, rows_, cols_, us, gidx, egidx = [], [], [], [], [], [], []; na = 0
    for gi, s in enumerate(mols):
        c = C[s]; n = c['x'].shape[0]; e = c['ei'].shape[1] if c['ei'].numel() else 0
        xs.append(c['x']); us.append([c['ap'], c['bp'], c['tpsa']]); gidx += [gi] * n
        if e: eas.append(c['ea']); rows_.append(c['ei'][0] + na); cols_.append(c['ei'][1] + na); egidx += [gi] * e
        na += n
    x = torch.cat(xs); ne = sum(len(r) for r in rows_)
    ea = torch.cat(eas) if eas else torch.zeros(0, 9); row = torch.cat(rows_) if rows_ else torch.zeros(0, dtype=torch.long); col = torch.cat(cols_) if cols_ else torch.zeros(0, dtype=torch.long)
    u = torch.tensor(us, dtype=torch.float32)
    node_inc = torch.zeros(len(mols), na); node_inc[torch.tensor(gidx), torch.arange(na)] = 1
    col_inc = torch.zeros(na, ne); col_inc[col, torch.arange(ne)] = 1
    eg_inc = torch.zeros(len(mols), ne)
    if ne: eg_inc[torch.tensor(egidx), torch.arange(ne)] = 1
    return x, ea, row, col, u, node_inc, col_inc, eg_inc

CH = 48
dm = np.zeros((3, S, 256), np.float32); gh = np.zeros((3, S, 226), np.float32)
enc_d = [DMPNNEncoder(dict(np.load(PAR / f'dmpnn_{s}.npz'))) for s in SEEDS]
ghm = []
for s in SEEDS:
    m = GHGNN_model(M.n_atom_features(), M.n_bond_features(), 3, HP['hidden_dim']); m.load_state_dict(torch.load(MODELS / 'ghgnn' / s / 'final.pt', map_location='cpu')); m.eval(); ghm.append(GHGNNEncoder(m))
hba = np.zeros(S, int); hbd = np.zeros(S, int)
for lo in range(0, S, CH):
    chunk = smiles[lo:lo + CH]
    b = dmpnn_graph_batch([graph(s) for s in chunk])
    for k, e in enumerate(enc_d): dm[k, lo:lo + len(chunk)] = e(*b).numpy()
    C = mol_cache({s: s for s in chunk}); gb = ghgnn_batch(C, chunk)
    for k, e in enumerate(ghm): gh[k, lo:lo + len(chunk)] = e(*gb).numpy()
    hba[lo:lo + len(chunk)] = [C[s]['hba'] for s in chunk]; hbd[lo:lo + len(chunk)] = [C[s]['hbd'] for s in chunk]
    if lo % (CH * 20) == 0: print(lo, '/', S, flush=True)
assert np.isfinite(dm).all() and np.isfinite(gh).all()
(OUT / 'dmpnn_solvents.f32').write_bytes(dm.astype('<f4').tobytes()); (OUT / 'ghgnn_solvents.f32').write_bytes(gh.astype('<f4').tobytes())
(OUT / 'dmpnn_solvents.f16').write_bytes(dm.astype('<f2').tobytes()); (OUT / 'ghgnn_solvents.f16').write_bytes(gh.astype('<f2').tobytes())   # stored by the site
keep = ['solvent_id', 'name', 'name_short', 'kind', 'cation_class', 'status', 'domain', 'max_sim', 'nearest_training_solvent', 'n_records', 'smiles', 'cas', 'source']
meta = [{**{k: r[k] for k in keep}, 'hba': int(hba[i]), 'hbd': int(hbd[i])} for i, r in enumerate(lib)]
(OUT / 'solvents.json').write_text(json.dumps(meta, ensure_ascii=False, separators=(',', ':')))
(OUT / 'solvents_manifest.json').write_text(json.dumps({'n_solvents': S, 'seeds': SEEDS, 'dmpnn_shape': [3, S, 256], 'ghgnn_shape': [3, S, 226], 'dtype': 'float32 little-endian'}, indent=1))
print('done', S)
