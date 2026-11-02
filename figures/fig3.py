#!/usr/bin/env python
"""Figure 3. Models predict selectivity where experiments find none: predicted selectivity of pairs that experiments
cannot separate (a) and two cases, xylenes (b) and butenes (c).
Usage: python fig3.py <this folder> <out_dir>"""
import sys, os
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import rdDepictor
ROOT = Path(sys.argv[1]).resolve(); OUT = Path(sys.argv[2]).resolve(); OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / 'scripts')); os.chdir(ROOT)
from load_all import load

W, H, MM = 183.0, 66.0, 25.4
INK = '#2C2C2A'; PUR = '#534AB7'; GRY = '#5F5E5A'; ECF = '#B4B2A9'
COL = {'JOINT': PUR, 'GHGNN': GRY, 'ECFP': ECF}
TXT = {'JOINT': PUR, 'GHGNN': GRY, 'ECFP': '#8E8C82'}
NAME = {'JOINT': 'D-MPNN', 'GHGNN': 'GH-GNN', 'ECFP': 'ECFP + LightGBM'}
T_EXP, T_MOD = 1.026, 1.122
plt.rcParams.update({'font.family': ['Arial', 'Liberation Sans', 'DejaVu Sans'], 'svg.fonttype': 'none', 'pdf.fonttype': 42,
    'axes.linewidth': 0.5, 'xtick.major.width': 0.5, 'ytick.major.width': 0.5, 'xtick.minor.width': 0.4, 'ytick.minor.width': 0.4,
    'xtick.major.size': 2.0, 'ytick.major.size': 2.0, 'xtick.minor.size': 0, 'ytick.minor.size': 0,
    'xtick.labelsize': 6, 'ytick.labelsize': 6, 'axes.labelsize': 7, 'xtick.major.pad': 1.5, 'ytick.major.pad': 1.5,
    'axes.labelpad': 1.5, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.unicode_minus': True})

# ---------------- data ----------------
d = load(); p, M, smi = d['pairs'], d['models'], d['mol']
def Dcols(m): return ['D_' + c for c in M[m]]
null = p[p.S_inf < T_EXP]
share = {}
for m in ('JOINT', 'GHGNN', 'ECFP'):
    S = np.exp(np.abs(null[Dcols(m)].values))
    share[m] = (S > T_MOD).mean(0).mean()
    print(m, 'outside 1/1.12-1.12 (seed mean)', round(share[m], 3), ' beyond 1/3-3', round((S > 3).mean(), 3))
assert abs(share['JOINT'] - 0.377) < 0.005 and abs(share['GHGNN'] - 0.471) < 0.005 and abs(share['ECFP'] - 0.793) < 0.005

# case rule: among pairs with >= 50 records and median measured selectivity < 1.03,
# for each graph model take the most-measured pair whose median predicted selectivity exceeds 1.12
g = p.groupby('pair_key')
summ = g.agg(N=('S_inf', 'size'), med=('S_inf', 'median'))
cand = summ[(summ.N >= 50) & (summ.med < 1.03)].sort_values('N', ascending=False)
case = {}
for m in ('GHGNN', 'JOINT'):
    for key in cand.index:
        df = p[p.pair_key == key]
        if np.median(np.exp(np.abs(df[Dcols(m)].values))) > T_MOD:
            case[m] = key; break
can = lambda s: Chem.MolToSmiles(Chem.MolFromSmiles(s))
NM = {can(k): v for k, v in {'Cc1cccc(C)c1': 'm-xylene', 'Cc1ccc(C)cc1': 'p-xylene', 'C=CCC': '1-butene', 'C=C(C)C': 'isobutene'}.items()}
cases = []
for m in ('GHGNN', 'JOINT'):
    df = p[p.pair_key == case[m]].copy()
    names = {k: NM[can(smi[k])] for k in case[m].split('|')}
    ids = sorted(names)                       # fixed molecule order
    rowsgn = np.where(df.mol_A.values == ids[0], 1.0, -1.0)    # ln(gamma_ids0 / gamma_ids1)
    lex = df.D.values * rowsgn
    lpr = {k: df[Dcols(k)].values.mean(1) * rowsgn for k in ('JOINT', 'GHGNN')}
    if np.median(lpr[m]) < 0:                 # numerator = the isomer the inventing model puts higher
        ids = ids[::-1]; lex = -lex; lpr = {k: -v for k, v in lpr.items()}
    ex = np.exp(lex); pr = {k: np.exp(v) for k, v in lpr.items()}
    numer_id, denom_id = ids; numer, denom = names[numer_id], names[denom_id]
    print(f'case {m}: {numer}/{denom} n={len(df)} measured med {np.median(ex):.3f}  D-MPNN {np.median(pr["JOINT"]):.3f}  GH-GNN {np.median(pr["GHGNN"]):.3f}')
    cases.append(dict(model=m, numer=numer, denom=denom, ids=(numer_id, denom_id), n=len(df), ex=ex, pr=pr))
assert cases[0]['n'] == 1322 and {cases[0]['numer'], cases[0]['denom']} == {'m-xylene', 'p-xylene'}
assert {cases[1]['numer'], cases[1]['denom']} == {'1-butene', 'isobutene'}

# ---------------- figure ----------------
fig = plt.figure(figsize=(W / MM, H / MM))
ov = fig.add_axes([0, 0, 1, 1]); ov.set_xlim(0, W); ov.set_ylim(H, 0); ov.axis('off'); ov.set_zorder(-1)
def ax_mm(x, y, w, h):
    a = fig.add_axes([x / W, 1 - (y + h) / H, w / W, h / H]); a.patch.set_alpha(0); return a
def head(x, y, letter, title):
    ov.text(x, y, letter, fontsize=8, fontweight='bold', va='baseline', color=INK)
    ov.text(x + 4.0, y, title, fontsize=8, fontweight='bold', va='baseline', color=INK)

def draw_mol(axo, smi_, cx, cy, maxw, maxh, bl=1.9, lw=0.5, color=INK):
    m = Chem.MolFromSmiles(smi_)
    rdDepictor.SetPreferCoordGen(False); rdDepictor.Compute2DCoords(m); Chem.Kekulize(m, clearAromaticFlags=True)
    c = m.GetConformer()
    P = np.array([[c.GetAtomPosition(i).x, -c.GetAtomPosition(i).y] for i in range(m.GetNumAtoms())])
    P -= (P.min(0) + P.max(0)) / 2
    w, h = np.ptp(P, 0) + 1e-6
    s = min(bl / 1.5, maxw / w, maxh / h); P = P * s + np.array([cx, cy])
    rings = [list(r) for r in m.GetRingInfo().AtomRings()]
    for b in m.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx(); p_, q_ = P[i], P[j]
        axo.plot([p_[0], q_[0]], [p_[1], q_[1]], color=color, lw=lw, solid_capstyle='round')
        if b.GetBondTypeAsDouble() == 2:
            v = q_ - p_; u = v / np.hypot(*v); nrm = np.array([-u[1], u[0]])
            ring = next((r for r in rings if i in r and j in r), None)
            if ring is not None:
                cen = P[ring].mean(0); sg = 1 if np.dot(cen - (p_ + q_) / 2, nrm) > 0 else -1
            else:
                sg = -1 if nrm[1] > 0 else 1
            off = nrm * sg * 0.34 * s / (bl / 1.5)
            a1, a2 = p_ + v * 0.15 + off, p_ + v * 0.85 + off
            axo.plot([a1[0], a2[0]], [a1[1], a2[1]], color=color, lw=lw, solid_capstyle='round')

LOGTICKS = [0.5, 0.67, 1.0, 1.5, 2.0]
def logaxis(ax, axis, lim, ticks):
    f = getattr(ax, f'set_{axis}scale'); f('log')
    getattr(ax, f'set_{axis}lim')(*lim)
    getattr(ax, f'set_{axis}ticks')(ticks); getattr(ax, f'set_{axis}ticklabels')([f'{t:g}' for t in ticks])
    getattr(ax, f'{axis}axis').set_minor_locator(matplotlib.ticker.NullLocator())

TOP = 12.0; PH = 42.0
# ---- a: histogram of predicted selectivity, inseparable pairs ----
head(0, 4.0, 'a', 'Models predict selectivity where none is measured')
ax = ax_mm(11.0, TOP, 58.0, PH)
lo, hi = np.log(1 / 3), np.log(3)
edges = np.arange(lo, hi + 1e-9, 0.02)
ymax = 0
for m in ('ECFP', 'GHGNN', 'JOINT'):
    D = null[Dcols(m)].values
    cnt = np.histogram(D.ravel(), edges)[0] / D.shape[1]
    ymax = max(ymax, cnt.max())
    ax.stairs(cnt, np.exp(edges), color=COL[m], lw=0.9 if m != 'ECFP' else 1.0, zorder=3 if m == 'JOINT' else 2)
ax.axvspan(1 / T_EXP, T_EXP, ymax=0.9, color=INK, lw=0, zorder=1)
for t in (T_MOD, 1 / T_MOD):
    ax.axvline(t, ymax=0.9, color=PUR, lw=0.5, ls=(0, (3, 2)), zorder=1)
logaxis(ax, 'x', (1 / 3, 3), [0.33, 0.5, 0.67, 1.0, 1.5, 2.0, 3.0])
ax.set_xticklabels(['1/3', '0.5', '0.67', '1', '1.5', '2', '3'])
ax.set_ylim(0, ymax * 1.2)
ax.set_xlabel('Predicted selectivity (A/B)'); ax.set_ylabel('Isomer pairs')
ax.text(1.0, ymax * 1.2, 'Measured', fontsize=6, color=INK, ha='center', va='top')
# direct labels: model name + share, left of the peak
lab_y = {'JOINT': 1.13, 'GHGNN': 0.89, 'ECFP': 0.63}
for m in ('JOINT', 'GHGNN', 'ECFP'):
    ax.text(0.48, ymax * lab_y[m], f'{NAME[m]}\n{100 * share[m]:.0f}% beyond 1.12', fontsize=6, color=TXT[m], ha='center', va='top', linespacing=1.15, multialignment='center')

# ---- b, c: parity cases ----
LIM = (0.5, 2.0)
for k, (cs, letter, x0) in enumerate(zip(cases, 'bc', (88.0, 138.0))):
    title = {'GHGNN': 'Xylenes: GH-GNN invents one', 'JOINT': 'Butenes: D-MPNN invents one'}[cs['model']]
    head(x0 - 10.5, 4.0, letter, title)
    ax = ax_mm(x0, TOP, PH, PH)
    ax.plot(LIM, LIM, color=INK, lw=0.5, ls=(0, (3, 2)), zorder=1)
    for t in (T_MOD, 1 / T_MOD):
        ax.axhline(t, color=PUR, lw=0.5, ls=(0, (3, 2)), zorder=1)
    order = ('GHGNN', 'JOINT') if cs['model'] == 'GHGNN' else ('GHGNN', 'JOINT')
    for m in order:
        ax.plot(cs['ex'], cs['pr'][m], 'o', ms=1.3 if cs['n'] > 500 else 1.9, mew=0, color=COL[m], alpha=0.55 if cs['n'] > 500 else 0.8, zorder=2)
    logaxis(ax, 'x', LIM, LOGTICKS); logaxis(ax, 'y', LIM, LOGTICKS)
    nm, dn = cs['numer'], cs['denom']
    ax.set_xlabel('Measured selectivity')
    ax.set_ylabel(f'Predicted selectivity', fontsize=7)
    # model labels, top-left
    for j, m in enumerate(('GHGNN', 'JOINT')):
        ax.text(0.04, 0.96 - 0.075 * j, NAME[m], transform=ax.transAxes, fontsize=6.5, color=TXT[m], ha='left', va='top', fontweight='bold')
    # structures, top-left
    for j, (mid_, nme) in enumerate(zip(cs['ids'], (nm, dn))):
        cx = x0 + 28.0 + j * 10.0
        draw_mol(ov, smi[mid_], cx, TOP + PH - 9.6, 7.5, 4.6)
        ov.text(cx, TOP + PH - 6.0, nme, fontsize=6, ha='center', va='top', color=INK)
    ov.text(x0 + 33.0, TOP + PH - 9.4, '/', fontsize=8, ha='center', va='center', color=INK)

for ext, kw in (('svg', {}), ('pdf', {}), ('png', {'dpi': 600})):
    fig.savefig(OUT / f'Fig3.{ext}', **kw)
print('saved')
