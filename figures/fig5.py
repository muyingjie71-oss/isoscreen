#!/usr/bin/env python
"""Figure 5. The same D-MPNN trained on activities only or on activities + selectivity: selectivity (a), error correlation (b)
and spurious selectivity (c).
Usage: python fig5.py <this folder> <out_dir>"""
import sys, os
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
ROOT = Path(sys.argv[1]).resolve(); OUT = Path(sys.argv[2]).resolve(); OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / 'scripts')); os.chdir(ROOT)
from load_all import load
W, H, MM = 183.0, 114.0, 25.4
INK, PUR, LP, LPT = '#2C2C2A', '#534AB7', '#AFA9EC', '#8C84D9'
COL = {'ABS': LP, 'JOINT': PUR}; TXT = {'ABS': LPT, 'JOINT': PUR}
NAME = {'ABS': 'Activities only', 'JOINT': 'Activities + selectivity'}
T_EXP, T_MOD = 1.026, 1.122
plt.rcParams.update({'font.family': ['Arial', 'Liberation Sans', 'DejaVu Sans'], 'svg.fonttype': 'none', 'pdf.fonttype': 42,
    'axes.linewidth': 0.5, 'xtick.major.width': 0.5, 'ytick.major.width': 0.5, 'xtick.major.size': 2.0, 'ytick.major.size': 2.0,
    'xtick.labelsize': 6, 'ytick.labelsize': 6, 'axes.labelsize': 7, 'xtick.major.pad': 1.5, 'ytick.major.pad': 1.5,
    'axes.labelpad': 1.5, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.unicode_minus': True,
    'mathtext.fontset': 'custom', 'mathtext.rm': 'Liberation Sans', 'mathtext.default': 'regular'})

# ---------------- data ----------------
d = load(); p, M, obs = d['pairs'], d['models'], d['obs']
Dt = p.D.values
r2 = lambda t, q: 1 - np.sum((t - q) ** 2) / np.sum((t - t.mean()) ** 2)
R2 = {k: np.mean([r2(Dt, p['D_' + c].values) for c in M[k]]) for k in ('ABS', 'JOINT')}
null = p[p.S_inf < T_EXP]
FS = {k: np.mean([(np.abs(null['D_' + c]) > np.log(T_MOD)).mean() for c in M[k]]) for k in ('ABS', 'JOINT')}
idx = obs.set_index('canonical_record_id'); err, R = {}, {}
for k in ('ABS', 'JOINT'):
    ea, eb, rs = [], [], []
    for c in M[k]:
        e = idx[c] - idx.ln_gamma_inf
        a_ = e.reindex(p.canonical_record_id_A).values; b_ = e.reindex(p.canonical_record_id_B).values
        ea.append(a_); eb.append(b_); rs.append(np.corrcoef(a_, b_)[0, 1])
    err[k] = (np.concatenate(ea), np.concatenate(eb)); R[k] = np.mean(rs)
print('R2', {k: round(v, 3) for k, v in R2.items()}, ' invented', {k: round(v, 3) for k, v in FS.items()}, ' r', {k: round(v, 3) for k, v in R.items()})
assert abs(R2['JOINT'] - 0.415) < 0.005 and abs(FS['JOINT'] - 0.377) < 0.005
assert round(R['ABS'], 2) == round(R['JOINT'], 2)

# ---------------- figure ----------------
fig = plt.figure(figsize=(W / MM, H / MM))
ov = fig.add_axes([0, 0, 1, 1]); ov.set_xlim(0, W); ov.set_ylim(H, 0); ov.axis('off'); ov.set_zorder(-1)
def ax_mm(x, y, w, h):
    a = fig.add_axes([x / W, 1 - (y + h) / H, w / W, h / H]); a.patch.set_alpha(0); return a
def head(x, letter, title):
    ov.text(x, 3.6, letter, fontsize=8, fontweight='bold', va='baseline', color=INK)
    ov.text(x + 4.0, 3.6, title, fontsize=8, fontweight='bold', va='baseline', color=INK)
CM = LinearSegmentedColormap.from_list('p', ['#EEEDFE', '#AFA9EC', PUR, '#26215C'])
def head(x, letter, title, y=3.6):
    ov.text(x, y, letter, fontsize=8, fontweight='bold', va='baseline', color=INK)
    ov.text(x + 4.0, y, title, fontsize=8, fontweight='bold', va='baseline', color=INK)
Y0, SZ = 13.0, 38.0
STK = [0.5, 0.67, 1.0, 1.5, 2.0]
L = np.log(2.0); LIM = 1.5
# ---- a: selectivity parity, one hexbin per training ----
head(0, 'a', 'Selectivity stays hard to predict')
for j, (k, x0) in enumerate((('ABS', 11.0), ('JOINT', 52.5))):
    ax = ax_mm(x0, Y0, SZ, SZ)
    q = np.concatenate([p['D_' + c].values for c in M[k]]) if False else p['D_' + M[k][0]].values
    ax.hexbin(Dt, q, gridsize=34, extent=(-L, L, -L, L), bins='log', cmap=CM, mincnt=1, linewidths=0, zorder=1)
    ax.plot([-L, L], [-L, L], color=INK, lw=0.5, ls=(0, (3, 2)), zorder=2)
    ax.set_xlim(-L, L); ax.set_ylim(-L, L); ax.set_aspect('equal')
    ax.set_xticks(np.log(STK)); ax.set_xticklabels(['0.5', '0.67', '1', '1.5', '2' if j else ''])
    ax.set_yticks(np.log(STK)); ax.set_yticklabels(['0.5', '0.67', '1', '1.5', '2'] if j == 0 else [])
    ax.set_xlabel('Measured selectivity')
    if j == 0: ax.set_ylabel('Predicted selectivity')
    ax.text(-L, L * 1.06, f'{NAME[k]}:  R$^{{2}}$ = {R2[k]:.2f}', fontsize=7, color=TXT[k], va='bottom', fontweight='bold')
# ---- b: errors of the two isomers, one hexbin per training ----
head(91.0, 'b', 'Errors still do not cancel')
for j, (k, x0) in enumerate((('ABS', 103.0), ('JOINT', 143.5))):
    ax = ax_mm(x0, Y0, SZ, SZ)
    ea, eb = err[k]; n1 = len(p)
    ax.hexbin(ea[:n1], eb[:n1], gridsize=34, extent=(-LIM, LIM, -LIM, LIM), bins='log', cmap=CM, mincnt=1, linewidths=0, zorder=1)
    ax.plot([-LIM, LIM], [-LIM, LIM], color=INK, lw=0.5, ls=(0, (3, 2)), zorder=2)
    ax.set_xlim(-LIM, LIM); ax.set_ylim(-LIM, LIM); ax.set_aspect('equal')
    ax.set_xticks([-1, 0, 1]); ax.set_yticks([-1, 0, 1])
    if j: ax.set_yticklabels([])
    ax.set_xlabel(r'Deviation, isomer A (ln $\gamma^{\infty}$)', fontsize=6.5)
    if j == 0: ax.set_ylabel(r'Deviation, isomer B (ln $\gamma^{\infty}$)', fontsize=6.5)
    ax.text(-LIM, LIM * 1.06, f'{NAME[k]}:  r = {R[k]:.2f}', fontsize=7, color=TXT[k], va='bottom', fontweight='bold')
# ---- c: predicted selectivity for pairs experiments cannot separate (as Fig. 3a) ----
Y1 = 67.0
head(0, 'c', 'Spurious selectivity does not drop', y=Y1)
ax = ax_mm(11.0, Y1 + 6.0, 170.0, 32.0)
edges = np.arange(np.log(1 / 3), np.log(3) + 1e-9, 0.02); ymax = 0
for k, lw, z in (('ABS', 1.4, 2), ('JOINT', 0.8, 3)):
    Dn = null[['D_' + c for c in M[k]]].values
    cnt = np.histogram(Dn.ravel(), edges)[0] / Dn.shape[1]; ymax = max(ymax, cnt.max())
    ax.stairs(cnt, np.exp(edges), color=COL[k], lw=lw, zorder=z)
ax.set_xscale('log')
ax.axvspan(1 / T_EXP, T_EXP, ymax=0.88, color=INK, lw=0, zorder=1)
for t in (T_MOD, 1 / T_MOD): ax.axvline(t, ymax=0.88, color=PUR, lw=0.5, ls=(0, (3, 2)), zorder=1)
ax.set_xlim(1 / 3, 3); ax.set_ylim(0, ymax * 1.2)
ax.set_xticks([1 / 3, 0.5, 0.67, 1, 1.5, 2, 3]); ax.set_xticklabels(['1/3', '0.5', '0.67', '1', '1.5', '2', '3']); ax.minorticks_off()
ax.text(1.0, ymax * 1.2, 'Measured', fontsize=6, color=INK, ha='center', va='top')
ax.set_xlabel('Predicted selectivity (A/B), pairs that experiments cannot separate'); ax.set_ylabel('Isomer pairs')
ax.text(2.06, ymax * 0.95, f'{NAME["ABS"]}\n{100 * FS["ABS"]:.0f}% beyond 1.12', fontsize=6.5, color=TXT['ABS'], va='top', ha='center', multialignment='center', linespacing=1.15, fontweight='bold')
ax.text(2.06, ymax * 0.62, f'{NAME["JOINT"]}\n{100 * FS["JOINT"]:.0f}% beyond 1.12', fontsize=6.5, color=TXT['JOINT'], va='top', ha='center', multialignment='center', linespacing=1.15, fontweight='bold')
for ext, kw in (('svg', {}), ('pdf', {}), ('png', {'dpi': 600})):
    fig.savefig(OUT / f'Fig5.{ext}', **kw)
print('saved')
