#!/usr/bin/env python
"""Figure 4. Errors of the two isomers of a pair: experiments (a), D-MPNN (b) and GH-GNN (c) on new isomer families, and the
share of error that cancels for new isomers and new solvents (d). New-solvent predictions come from
validation/oof_predictions_ln_gamma.csv (columns newsolv_*).
Usage: python fig4.py <this folder> <out_dir>"""
import sys, os, math
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
ROOT = Path(sys.argv[1]).resolve(); OUT = Path(sys.argv[2]).resolve(); OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / 'scripts')); os.chdir(ROOT)
from load_all import load
W, H, MM = 183.0, 117.0, 25.4
INK, PUR, GRY = '#2C2C2A', '#534AB7', '#5F5E5A'
plt.rcParams.update({'font.family': ['Arial', 'Liberation Sans', 'DejaVu Sans'], 'svg.fonttype': 'none', 'pdf.fonttype': 42,
    'axes.linewidth': 0.5, 'xtick.major.width': 0.5, 'ytick.major.width': 0.5, 'xtick.major.size': 2.0, 'ytick.major.size': 2.0,
    'xtick.labelsize': 6, 'ytick.labelsize': 6, 'axes.labelsize': 7, 'xtick.major.pad': 1.5, 'ytick.major.pad': 1.5,
    'axes.labelpad': 1.5, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.unicode_minus': True, 'mathtext.fontset': 'custom', 'mathtext.rm': 'Liberation Sans', 'mathtext.it': 'Liberation Sans:italic', 'mathtext.default': 'regular'})

# ---------------- data ----------------
d = load(); obs, M, pairs = d['obs'], d['models'], d['pairs']
SO = {'SO_GHGNN': [], 'SO_JOINT': []}
# new-solvent test (solvents held out): out-of-fold ln gamma for 3 seeds of each model
both = pd.read_csv(ROOT.parent / 'validation' / 'oof_predictions_ln_gamma.csv', dtype={'canonical_record_id': str})
for tag, pre in (('SO_GHGNN', 'newsolv_GHGNN_'), ('SO_JOINT', 'newsolv_DMPNN_')):
    for col in sorted(c for c in both.columns if c.startswith(pre)):
        name = tag + '_' + col.split('_')[-1]
        obs = obs.merge(both[['canonical_record_id', col]].rename(columns={col: name}), on='canonical_record_id', how='left'); SO[tag].append(name)
assert all(len(v) == 3 for v in SO.values()) and not obs[SO['SO_GHGNN'] + SO['SO_JOINT']].isna().any().any()
M = dict(M); M.update(SO)
rep = pd.read_csv(ROOT / 'outputs/fig4a_experiment_replicate_errors.csv')
idx = obs.set_index('canonical_record_id')
err, r = {}, {}
for m in ('JOINT', 'GHGNN'):
    rs = []
    for c in M[m]:
        e = idx[c] - idx.ln_gamma_inf
        eA = e.reindex(pairs.canonical_record_id_A).values; eB = e.reindex(pairs.canonical_record_id_B).values
        rs.append(np.corrcoef(eA, eB)[0, 1])
        if c == M[m][0]: err[m] = (eA, eB)
    r[m] = np.mean(rs)
r['EXP'] = np.corrcoef(rep.dA, rep.dB)[0, 1]
print({k: round(v, 3) for k, v in r.items()})
res = pd.read_csv(ROOT / 'outputs/overall_metrics_by_run.csv'); res = res[~res.model.str.endswith('_ens')]
medD = res.groupby('model').D_median_abs_err.mean()
y = obs.ln_gamma_inf.values
typ = {'EXP': (math.exp(np.median(np.abs(np.r_[rep.dA, rep.dB]))), math.exp(np.median(np.abs(rep.dD)))),
       'JOINT': (math.exp(np.mean([np.median(np.abs(obs[c] - y)) for c in M['JOINT']])), math.exp(medD['JOINT'])),
       'GHGNN': (math.exp(np.mean([np.median(np.abs(obs[c] - y)) for c in M['GHGNN']])), math.exp(medD['GHGNN']))}
ix = obs.set_index('canonical_record_id'); ia = pairs.canonical_record_id_A.values; ib = pairs.canonical_record_id_B.values
for m in ('SO_JOINT', 'SO_GHGNN'):
    sel = [np.median(np.abs((ix.loc[ia, c].values - ix.loc[ib, c].values) - pairs.D.values)) for c in M[m]]
    typ[m] = (math.exp(np.mean([np.median(np.abs(obs[c] - y)) for c in M[m]])), math.exp(np.mean(sel)))
    rs = []
    for c in M[m]:
        e = ix[c] - ix.ln_gamma_inf; rs.append(np.corrcoef(e.reindex(ia).values, e.reindex(ib).values)[0, 1])
    r[m] = np.mean(rs)
print({k: tuple(round(x, 3) for x in v) for k, v in typ.items()}, {k: round(v, 3) for k, v in r.items()})
LIM = 1.5
for k, (a, b) in [('EXP', (rep.dA.values, rep.dB.values))] + [(m, err[m]) for m in err]:
    print(k, 'points outside +-1.5:', int(((np.abs(a) > LIM) | (np.abs(b) > LIM)).sum()), 'of', len(a))

# ---------------- figure ----------------
fig = plt.figure(figsize=(W / MM, H / MM))
ov = fig.add_axes([0, 0, 1, 1]); ov.set_xlim(0, W); ov.set_ylim(H, 0); ov.axis('off'); ov.set_zorder(-1)
def ax_mm(x, y_, w, h):
    a = fig.add_axes([x / W, 1 - (y_ + h) / H, w / W, h / H]); a.patch.set_alpha(0); return a
def head(x, letter, title, y=4.0):
    ov.text(x, y, letter, fontsize=8, fontweight='bold', va='baseline', color=INK)
    ov.text(x + 4.0, y, title, fontsize=8, fontweight='bold', va='baseline', color=INK)
TK = [-1, 0, 1]
def fmt(ax, ylab):
    ax.set_xlim(-LIM, LIM); ax.set_ylim(-LIM, LIM); ax.set_xticks(TK); ax.set_yticks(TK)
    ax.set_aspect('equal')
    ax.plot([-LIM, LIM], [-LIM, LIM], color=INK, lw=0.5, ls=(0, (3, 2)), zorder=3)
    ax.set_xlabel(r'Deviation, isomer A (ln $\gamma^{\infty}$)')
    if ylab: ax.set_ylabel(r'Deviation, isomer B (ln $\gamma^{\infty}$)')
SZ, Y0 = 44.0, 10.0
XS = (12.0, 73.0, 134.0)
NAME = {'EXP': 'Experiment', 'JOINT': 'D-MPNN', 'GHGNN': 'GH-GNN', 'SO_JOINT': 'D-MPNN', 'SO_GHGNN': 'GH-GNN'}
SUB = {'EXP': 'two studies', 'JOINT': 'new isomers', 'GHGNN': 'new isomers', 'SO_JOINT': 'new solvents', 'SO_GHGNN': 'new solvents'}
COL = {'EXP': INK, 'JOINT': PUR, 'GHGNN': GRY, 'SO_JOINT': PUR, 'SO_GHGNN': GRY}
LT = {'EXP': '#9A9994', 'JOINT': '#AFA9EC', 'GHGNN': '#BDBCB8', 'SO_JOINT': '#AFA9EC', 'SO_GHGNN': '#BDBCB8'}
head(0.0, 'a', 'Experiments err together')
head(XS[1] - 11.0, 'b', 'D-MPNN errs separately')
head(XS[2] - 11.0, 'c', 'So does GH-GNN')
ax = ax_mm(XS[0], Y0, SZ, SZ); fmt(ax, True)
ax.scatter(rep.dA, rep.dB, s=5, color=INK, alpha=0.75, lw=0, zorder=2)
ax.text(-1.38, 1.38, f'Experiment\nr = {r["EXP"]:.2f}', fontsize=7, va='top', color=INK, linespacing=1.15, fontweight='bold')
CM = {'JOINT': LinearSegmentedColormap.from_list('p', ['#EEEDFE', '#AFA9EC', PUR, '#26215C']),
      'GHGNN': LinearSegmentedColormap.from_list('g', ['#F1F1EF', '#BDBCB8', GRY, '#1F1F1D'])}
for m, x0 in (('JOINT', XS[1]), ('GHGNN', XS[2])):
    ax = ax_mm(x0, Y0, SZ, SZ); fmt(ax, False)
    ax.hexbin(*err[m], gridsize=44, extent=(-LIM, LIM, -LIM, LIM), bins='log', cmap=CM[m], mincnt=1, linewidths=0.0, zorder=1)
    ax.text(-1.38, 1.38, f'{NAME[m]}\nr = {r[m]:.2f}', fontsize=7, va='top', color=COL[m], linespacing=1.15, fontweight='bold')

# ---- d: waterfall, single-isomer error -> cancelled part -> selectivity error ----
Y1 = 66.0
head(0.0, 'd', 'Model errors cancel only for isomers seen in training', y=Y1)
ax = ax_mm(XS[0], Y1 + 7.0, W - XS[0] - 4.0, 29.0)
ax.patch.set_alpha(0)
BW = 0.62
xt, xl = [], []
GROUPS = ('EXP', 'JOINT', 'GHGNN', 'SO_JOINT', 'SO_GHGNN')
for g, m in enumerate(GROUPS):
    a_, b_ = typ[m]; x0 = g * 4.0 + (0.6 if g >= 3 else 0.0)
    cancel = 1 - math.log(b_) / math.log(a_)
    print(m, 'cancelled share of log error', round(cancel, 3))
    ax.bar(x0, a_ - 1, BW, bottom=1, color=LT[m], lw=0, zorder=2)
    ax.bar(x0 + 1, a_ - b_, BW, bottom=b_, facecolor='white', edgecolor=COL[m], hatch='//////', lw=0.5, zorder=2)
    ax.bar(x0 + 2, b_ - 1, BW, bottom=1, color=COL[m], lw=0, zorder=2)
    ax.plot([x0 + BW / 2, x0 + 1 - BW / 2], [a_, a_], color=INK, lw=0.4, zorder=1)
    ax.plot([x0 + 1 + BW / 2, x0 + 2 - BW / 2], [b_, b_], color=INK, lw=0.4, zorder=1)
    ax.text(x0, a_ + 0.003, f'{a_:.2f}', ha='center', va='bottom', fontsize=6, color=INK)
    ax.text(x0 + 1, a_ + 0.003, f'{100 * cancel:.0f}%', ha='center', va='bottom', fontsize=6, color=COL[m], fontweight='bold')
    ax.text(x0 + 2, b_ + 0.003, f'{b_:.2f}', ha='center', va='bottom', fontsize=6, color=INK)
    xt += [x0, x0 + 1, x0 + 2]; xl += ['Each\nisomer', 'Cancelled', 'Selec-\ntivity']
    ax.text(x0 + 1, 0.938, NAME[m], ha='center', va='top', fontsize=7, fontweight='bold', color=COL[m], clip_on=False)
    ax.text(x0 + 1, 0.913, SUB[m], ha='center', va='top', fontsize=6, color=COL[m], clip_on=False)
ax.set_xticks(xt); ax.set_xticklabels(['Each\nisomer', 'Can-\ncelled', 'Selec-\ntivity'] * len(GROUPS), fontsize=6, linespacing=1.0)
ax.tick_params(axis='x', length=0)
ax.set_xlim(-0.6, 19.2); ax.set_ylim(1.0, 1.27)
ax.axvline(11.3, color='#D3D1C7', lw=0.5, zorder=0)
ax.set_yticks([1.0, 1.05, 1.1, 1.15, 1.2, 1.25]); ax.set_yticklabels(['1.00', '1.05', '1.10', '1.15', '1.20', '1.25'])
ax.set_ylabel('Typical error (fold)')
for ext, kw in (('svg', {}), ('pdf', {}), ('png', {'dpi': 600})):
    fig.savefig(OUT / f'Fig4.{ext}', **kw)
print('saved')
