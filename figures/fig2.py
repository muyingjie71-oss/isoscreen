#!/usr/bin/env python
"""Figure 2. Measured versus predicted ln gamma_inf (a) and selectivity (b) for D-MPNN, GH-GNN and ECFP + LightGBM
on new isomer families.
Usage: python fig2.py <this folder> <out_dir>"""
import sys, math
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
ROOT = Path(sys.argv[1]); OUT = Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / 'scripts')); from load_all import load
W, H, MM = 180.0, 130.0, 25.4
INK = '#2C2C2A'; PUR = '#534AB7'; LIGHTP = '#AFA9EC'; LGREY = '#C9CFCC'
plt.rcParams.update({'font.family': ['Arial', 'Liberation Sans', 'DejaVu Sans'], 'svg.fonttype': 'none', 'pdf.fonttype': 42,
    'axes.unicode_minus': True, 'axes.linewidth': 0.8, 'xtick.major.width': 0.75, 'ytick.major.width': 0.75, 'xtick.major.size': 2.4, 'ytick.major.size': 2.4,
    'xtick.labelsize': 7.2, 'ytick.labelsize': 7.2, 'axes.labelsize': 7.8, 'xtick.major.pad': 1.8, 'ytick.major.pad': 1.8, 'axes.labelpad': 2.0})
fig = plt.figure(figsize=(W / MM, H / MM))
ov = fig.add_axes([0, 0, 1, 1]); ov.set_xlim(0, W); ov.set_ylim(H, 0); ov.axis('off')
def axmm(x, y, w, h):
    a = fig.add_axes([x / W, 1 - (y + h) / H, w / W, h / H]); a.set_zorder(2); a.patch.set_alpha(0)
    for s in ('top', 'right'): a.spines[s].set_visible(False)
    return a
def panel(x, l, t):
    ov.text(x, 3.5, l, fontsize=10.5, fontweight='bold', va='center'); ov.text(x + 4.6, 3.6, t, fontsize=8.0, fontweight='bold', va='center', color=INK)
d = load(); obs, pairs = d['obs'], d['pairs']; MODS = d['models']
y = obs.ln_gamma_inf.values; Dt = pairs.D.values; S = np.exp(pairs.abs_D.values)
r2 = lambda t, p: 1 - np.sum((t - p) ** 2) / np.sum((t - t.mean()) ** 2)
res = pd.read_csv(ROOT / 'outputs/overall_metrics_by_run.csv'); thr = res[res.model == 'JOINT'].D_median_abs_err.mean(); SM, SE = math.exp(thr), math.exp(0.0257)
cm = LinearSegmentedColormap.from_list('p', ['#EEEDFE', LIGHTP, PUR, '#26215C'])
SZ = 44.0; XS = (14.0, 67.0, 120.0); YR = (11.0, 74.0)
HX = []
def parity(ax, x, yv, lo, hi, tk):
    HX.append(ax.hexbin(x, yv, gridsize=42, extent=(lo, hi, lo, hi), cmap=cm, mincnt=1, linewidths=0, zorder=1))
    ax.plot([lo, hi], [lo, hi], color=INK, lw=0.7, ls=(0, (3, 2)), zorder=2); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_xticks(tk); ax.set_yticks(tk)
L = math.log(2.0); tl = ['0.5', '0.67', '1.0', '1.5', '2.0']
panel(XS[0] - 14, 'a', 'Each activity is predicted well')
panel_row2 = lambda x, l, t: (ov.text(x, YR[1] - 7.0, l, fontsize=10.5, fontweight='bold', va='center'),
                              ov.text(x + 4.6, YR[1] - 6.9, t, fontsize=8.0, fontweight='bold', va='center', color=INK))
panel_row2(XS[0] - 14, 'b', 'Selectivity is predicted poorly')
for k, (key, nm) in enumerate((('JOINT', 'D-MPNN'), ('GHGNN', 'GH-GNN'), ('ECFP', 'ECFP + LightGBM'))):
    cols = MODS[key]; Dp = [pairs['D_' + c].values for c in cols]
    R2a = np.mean([r2(y, obs[c].values) for c in cols]); R2b = np.mean([r2(Dt, p) for p in Dp]); print(nm, 'R2 lnγ %.3f  R2 sel %.3f' % (R2a, R2b))
    ax = axmm(XS[k], YR[0], SZ, SZ); parity(ax, y, obs[cols[0]].values, -3, 7, [-2, 0, 2, 4, 6])
    ax.set_xlabel('Measured ln γ$^{\\infty}$')
    if k == 0: ax.set_ylabel('Predicted ln γ$^{\\infty}$')
    ax.text(0.04, 0.96, f'{nm}\nR$^{{2}}$ = {R2a:.2f}', transform=ax.transAxes, fontsize=7.4, va='top', fontweight='bold')
    ax = axmm(XS[k], YR[1], SZ, SZ); parity(ax, Dt, Dp[0], -L, L, [-L, -math.log(1.5), 0, math.log(1.5), L]); ax.set_xticklabels(tl); ax.set_yticklabels(tl)
    ax.set_xlabel('Measured selectivity (A/B)')
    if k == 0: ax.set_ylabel('Predicted selectivity (A/B)')
    ax.text(0.04, 0.96, f'{nm}\nR$^{{2}}$ = {R2b:.2f}'.replace('= -', '= −'), transform=ax.transAxes, fontsize=7.4, va='top', fontweight='bold')
vmax = max(h_.get_array().max() for h_ in HX); nrm = LogNorm(vmin=1, vmax=vmax)
for h_ in HX: h_.set_norm(nrm)
cax = fig.add_axes([(XS[2] + SZ + 3.0) / W, 1 - (YR[1] + SZ) / H, 2.2 / W, (YR[1] + SZ - YR[0]) / H])
cb = fig.colorbar(HX[0], cax=cax); cb.outline.set_linewidth(0.5); cb.solids.set_rasterized(False); cb.solids.set_edgecolor('face'); cb.solids.set_linewidth(0.2)
cb.set_ticks([1, 10, 100, 1000]); cb.set_ticklabels(['1', '10', '100', '1,000']); cb.ax.minorticks_off(); cb.ax.tick_params(labelsize=7.2, width=0.6, length=2.0)
cb.set_label('Data points per hexagon', fontsize=7.4, labelpad=1)
for ext, kw in (('svg', {}), ('pdf', {}), ('png', {'dpi': 600})): fig.savefig(OUT / f'Fig2.{ext}', **kw)
