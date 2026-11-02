#!/usr/bin/env python
"""Data S2: numerical data underlying Figures 1-6, as one spreadsheet (AIChE J data sharing policy).
The summary values of Figures 3, 4 and 6 are taken by running the data part of the figure scripts, so they are identical
to what is plotted.
Usage: python make_data_s2.py <this folder> <out.xlsx> [Data S1 .xlsx, default ../data/isomer_selectivity_dataset.xlsx]"""
import sys, io, contextlib, math
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(sys.argv[1]).resolve(); OUTF = Path(sys.argv[2]).resolve()
DATA_S1 = Path(sys.argv[3]).resolve() if len(sys.argv) > 3 else ROOT.parent / 'data' / 'isomer_selectivity_dataset.xlsx'   # for molecule names
sys.path.insert(0, str(ROOT / 'scripts'))
from load_all import load


def data_part(script):
    """execute a figure script up to its figure section and return its namespace"""
    src = (ROOT / script).read_text(encoding='utf-8').split('# ---------------- figure ----------------')[0]
    ns = {'__name__': 'datapart', '__file__': str(ROOT / script)}
    argv = sys.argv; sys.argv = [script, str(ROOT), '/tmp/_data_s2_unused']
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(src, script, 'exec'), ns)
    sys.argv = argv
    return ns


SEEDS = ('20261007', '20261008', '20261009')
d = load(common_domain=False); obs, pairs, M = d['obs'], d['pairs'], d['models']
both = pd.read_csv(ROOT.parent / 'validation' / 'oof_predictions_ln_gamma.csv', dtype={'canonical_record_id': str})
COLS = []   # (source column, label)
for i, s in enumerate(SEEDS, 1):
    COLS.append((f'newiso_DMPNN_{s}', f'D-MPNN, new isomers, seed {i}'))
for i, s in enumerate(SEEDS, 1):
    COLS.append((f'newiso_DMPNNact_{s}', f'D-MPNN activities only, new isomers, seed {i}'))
for i, s in enumerate(SEEDS, 1):
    COLS.append((f'newiso_GHGNN_{s}', f'GH-GNN, new isomers, seed {i}'))
COLS.append(('newiso_ECFP', 'ECFP + LightGBM, new isomers'))
for i, s in enumerate(SEEDS, 1):
    COLS.append((f'newsolv_DMPNN_{s}', f'D-MPNN, new solvents, seed {i}'))
for i, s in enumerate(SEEDS, 1):
    COLS.append((f'newsolv_GHGNN_{s}', f'GH-GNN, new solvents, seed {i}'))
obs = obs.merge(both[['canonical_record_id'] + [c for c, _ in COLS if c.startswith('newsolv')]], on='canonical_record_id', how='left')
rec = pd.DataFrame({'Record ID': obs.canonical_record_id, 'Measured ln γ∞': obs.ln_gamma_inf})
for c, lab in COLS:
    rec['Predicted ln γ∞, ' + lab] = obs[c]
ix = obs.set_index('canonical_record_id')
pr = pd.DataFrame({'Pair ID': pairs.Pair_ID, 'Record ID of A': pairs.canonical_record_id_A, 'Record ID of B': pairs.canonical_record_id_B,
                   'Type of isomerism': pairs.relation_en, 'Measured ln S∞ = ln γ∞A − ln γ∞B': pairs.D})
for c, lab in COLS:
    pr['Predicted ln S∞, ' + lab] = ix.loc[pairs.canonical_record_id_A, c].values - ix.loc[pairs.canonical_record_id_B, c].values

# Figure 1
f1a = pd.read_csv(ROOT / 'plot_data/fig1/fig1a_isomer_pairs.csv')
f1a = f1a.rename(columns={'type': 'Type', 'A': 'Isomer A', 'A_smiles': 'SMILES A', 'B': 'Isomer B', 'B_smiles': 'SMILES B',
                          'Tb_A_C': 'Normal boiling point A (°C)', 'Tb_B_C': 'Normal boiling point B (°C)', 'dTb_K': 'Difference (K)'})
f1c = pd.read_csv(ROOT / 'plot_data/fig1/fig1c_histogram_counts.csv')
f1c = f1c.rename(columns={'bin_lo': 'Selectivity from', 'bin_hi': 'Selectivity to', 'count': 'Isomer pairs', 'zone': 'Range'})

# Figure 3b, c
ns3 = data_part('fig3.py')
rows = []
for c in ns3['cases']:
    panel = 'b' if c['model'] == 'GHGNN' else 'c'
    for k in range(c['n']):
        rows.append({'Panel': panel, 'Selectivity defined as': f"γ∞({c['numer']}) / γ∞({c['denom']})",
                     'Measured selectivity': c['ex'][k], 'Predicted selectivity, D-MPNN (mean of seeds)': c['pr']['JOINT'][k],
                     'Predicted selectivity, GH-GNN (mean of seeds)': c['pr']['GHGNN'][k]})
f3 = pd.DataFrame(rows)

# Figure 4a, d
ns4 = data_part('fig4.py')
f4a = ns4['rep'].rename(columns={'pair': 'Isomer pair (Mol IDs)', 'solvent': 'Solvent (Mol ID)', 'T': 'T (K)',
                                 'dA': 'Deviation of isomer A (ln γ∞)', 'dB': 'Deviation of isomer B (ln γ∞)',
                                 'dD': 'Deviation of selectivity (ln S∞)'})
NAME = {'EXP': 'Experiments (two studies)', 'JOINT': 'D-MPNN, new isomers', 'GHGNN': 'GH-GNN, new isomers',
        'SO_JOINT': 'D-MPNN, new solvents', 'SO_GHGNN': 'GH-GNN, new solvents'}
f4d = pd.DataFrame([{'Group': NAME[m], 'Typical error of a single activity coefficient (fold)': a,
                     'Typical error of the selectivity (fold)': b, 'Cancelled share of the error (log scale)': 1 - math.log(b) / math.log(a),
                     'Isomer error correlation r': ns4['r'][m]}
                    for m, (a, b) in ns4['typ'].items()])

# Figure 6a, b
ns6 = data_part('fig6.py')
q = lambda v: dict(zip(['10th percentile', '25th percentile', 'Median', '75th percentile', '90th percentile'],
                       np.percentile(v, [10, 25, 50, 75, 90])))
MN = {'JOINT': 'D-MPNN', 'GHGNN': 'GH-GNN'}
f6a = pd.DataFrame([{'Model': MN[m], 'Predicted selectivity': lab, 'Predictions (three seeds)': len(v), **q(v),
                     'Hit rate': hr, 'Share of all predictions': sh}
                    for m in ('JOINT', 'GHGNN') for lab, (v, hr, sh) in zip(ns6['BLAB'], ns6['cal'][m])])
f6b = pd.DataFrame([{'Model': MN[m], 'Type of isomerism': ns6['TLAB'][r], 'Predictions ≥ 1.12 (three seeds)': len(ns6['typ'][(r, m)][0]),
                     **q(ns6['typ'][(r, m)][0]), 'Hit rate': ns6['typ'][(r, m)][1]}
                    for r in ns6['TYPES'] for m in ('JOINT', 'GHGNN')])

# Figure 6c and ranking statistics (Section 4, Table S5)
ARGV = sys.argv; sys.argv = ['ranking_stats.py', str(ROOT)]
import importlib.util
spec = importlib.util.spec_from_file_location('ranking_stats', ROOT / 'ranking_stats.py'); rs = importlib.util.module_from_spec(spec)
with contextlib.redirect_stdout(io.StringIO()):
    spec.loader.exec_module(rs); u = rs.table()
mol = pd.read_excel(DATA_S1, sheet_name='Molecules')
name = dict(zip(mol['Mol ID'], mol['Name']))
f6c = pd.DataFrame({'Isomer pair': [' / '.join(name.get(k, k) for k in key.split('|')) for key in u.pair_key],
                    'T (K)': u.T_K, 'Solvents measured': u.n_solvents,
                    'Tests to find the best solvent, D-MPNN, new isomers': u.tests_j,
                    'Tests to find the best solvent, GH-GNN, new isomers': u.tests_g,
                    'Tests to find the best solvent, D-MPNN, new solvents': u.tests_nj,
                    'Tests to find the best solvent, GH-GNN, new solvents': u.tests_ng,
                    'Tests to find the best solvent, random order (expected)': u.random,
                    'Spearman ρ, D-MPNN, new isomers': u.rho_j, 'Spearman ρ, GH-GNN, new isomers': u.rho_g,
                    'Spearman ρ, D-MPNN, new solvents': u.rho_nj, 'Spearman ρ, GH-GNN, new solvents': u.rho_ng,
                    'Best / second-best measured selectivity': u.best_over_second})
show = ns6['cases'].iloc[[1, 4, 0, 3]]                      # the four examples plotted in Figure 6c
ex = dict(zip(show.key, show.label))
f6c.insert(1, 'Example in Figure 6c', [ex.get(k, '') for k in u.pair_key])
f6c = f6c.sort_values('Solvents measured', ascending=False)

contents = [
    ('Contents', 'This list'),
    ('Activity coefficients', 'Every measured ln γ∞ with the out-of-fold prediction of every model and training seed in both tests '
     '(new isomer families held out, new solvents held out). Record IDs match Data S1. Underlies Figures 2a and 4b–d and the single-activity '
     'columns of Tables 2 and S5. Figure 2a and Figures 4b, c plot seed 1, and all reported metrics are means over the three seeds. '
     'GH-GNN has no prediction for 225 records outside its domain, and all figures use the 68,538 records predicted by every model.'),
    ('Isomer pairs', 'Every isomer pair with its measured and predicted ln S∞ (isomer A over isomer B, as in Data S1), for every model, seed '
     'and test. Underlies Figures 1c, 2b, 3a, 5 and 6a, b and Tables 1, 2 and S5. Pairs with a measured selectivity below 1.026 are the '
     'experimentally unresolved pairs of Figures 3a and 5c. Figure 2b and Figure 5a plot seed 1.'),
    ('Fig 1a', 'Normal boiling points of the five example pairs (NIST Chemistry WebBook).'),
    ('Fig 1c', 'Number of isomer pairs per selectivity bin.'),
    ('Fig 3b-c', 'Measurements of m-xylene/p-xylene (panel b) and 1-butene/isobutene (panel c) with the predicted selectivities (mean of three seeds).'),
    ('Fig 4a', 'Deviation between the two studies of each of the 120 cross-study replicates.'),
    ('Fig 4d', 'Typical errors, cancelled share and isomer error correlation for each group.'),
    ('Fig 6a', 'Measured selectivity in the predicted direction, by predicted selectivity: box statistics and hit rate.'),
    ('Fig 6b', 'The same for predicted selectivities of at least 1.12, by type of isomerism.'),
    ('Fig 6c', 'For the 48 isomer pairs measured in at least 30 solvents: tests to find the best solvent and Spearman ρ between predicted '
     'and measured selectivity across solvents, for both tests. Underlies Figure 6c, Section 4 and Table S5.'),
]
SHEETS = [('Contents', pd.DataFrame(contents, columns=['Sheet', 'Content'])), ('Activity coefficients', rec), ('Isomer pairs', pr),
          ('Fig 1a', f1a), ('Fig 1c', f1c), ('Fig 3b-c', f3), ('Fig 4a', f4a), ('Fig 4d', f4d), ('Fig 6a', f6a), ('Fig 6b', f6b), ('Fig 6c', f6c)]
with pd.ExcelWriter(OUTF, engine='openpyxl') as xw:
    for nm, df in SHEETS:
        df = df.copy()
        for c in df.columns:
            if df[c].dtype.kind == 'f':
                df[c] = df[c].round(5)
        df.to_excel(xw, sheet_name=nm, index=False)
        ws = xw.sheets[nm]; ws.freeze_panes = 'A2'
        for j, c in enumerate(df.columns, 1):
            ws.column_dimensions[ws.cell(1, j).column_letter].width = 90 if (nm == 'Contents' and j == 2) else min(max(12, len(str(c)) * 0.9), 48)
print({nm: df.shape for nm, df in SHEETS})
print(f6a.round(3).to_string(index=False)); print(f6b.round(3).to_string(index=False)); print(f4d.round(3).to_string(index=False))
