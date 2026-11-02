#!/usr/bin/env python
"""Figure 6. How far predicted selectivities can be trusted: by predicted selectivity (a), by isomer type (b), and the number of
solvents tested before the best one is found when solvents are ranked by a model (c).
Panel c orients every isomer pair the same way before averaging over studies, and excludes the three cyclohexane pairs of
Arumugam et al. 2020 (SI S1.4).
Usage: python fig6.py <this folder> <out_dir>"""
import sys, os
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from rdkit import Chem
from rdkit.Chem import rdDepictor
ROOT = Path(sys.argv[1]).resolve(); OUT = Path(sys.argv[2]).resolve(); OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / 'scripts')); os.chdir(ROOT)
from load_all import load
W, H, MM = 183.0, 124.0, 25.4
INK, PUR, GRY, BAND = '#2C2C2A', '#534AB7', '#5F5E5A', '#D3D1C7'
COL = {'JOINT': PUR, 'GHGNN': GRY}; NAME = {'JOINT': 'D-MPNN', 'GHGNN': 'GH-GNN'}
LT = {'JOINT': '#AFA9EC', 'GHGNN': '#BDBCB8'}
T_EXP, T_MOD, T_RIGHT = 1.026, 1.122, float(np.exp(0.05))
plt.rcParams.update({'font.family': ['Arial', 'Liberation Sans', 'DejaVu Sans'], 'svg.fonttype': 'none', 'pdf.fonttype': 42,
    'axes.linewidth': 0.5, 'xtick.major.width': 0.5, 'ytick.major.width': 0.5, 'xtick.major.size': 2.0, 'ytick.major.size': 2.0,
    'xtick.labelsize': 6, 'ytick.labelsize': 6, 'axes.labelsize': 7, 'xtick.major.pad': 1.5, 'ytick.major.pad': 1.5,
    'axes.labelpad': 1.5, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.unicode_minus': True})

# ---------------- data ----------------
d = load(); p, M, smi = d['pairs'], d['models'], d['mol']
Dt = p.D.values
def oriented(mask, m):
    """per seed: predicted selectivity (folded) and measured selectivity in the predicted direction"""
    out = []
    for c in M[m]:
        q = p['D_' + c].values[mask]; out.append((np.exp(np.abs(q)), np.exp(Dt[mask] * np.sign(q))))
    return out
ALL = np.ones(len(p), bool)
BINS = [1.0, 1.05, 1.12, 1.25, 1.65, np.inf]; BLAB = ['1–1.05', '1.05–1.12', '1.12–1.25', '1.25–1.65', '>1.65']
cal = {}
for m in ('JOINT', 'GHGNN'):
    per = oriented(ALL, m); cal[m] = []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        vals, right, share = [], [], []
        for ps, ms in per:
            k = (ps >= lo) & (ps < hi); vals.append(ms[k]); right.append((ms[k] >= T_RIGHT).mean()); share.append(k.mean())
        cal[m].append((np.concatenate(vals), np.mean(right), np.mean(share)))
    print(m, 'right %', [round(100 * r, 1) for _, r, _ in cal[m]], ' share %', [round(100 * s, 1) for *_, s in cal[m]])
assert abs(cal['JOINT'][-1][1] - 0.84) < 0.015      # trust table: 84% right above 1.65

TYPES = ['connectivity (metamer)', 'functional group', 'ring-chain', 'positional', 'skeletal (chain branching)', 'skeletal (ring system)']
TLAB = {'connectivity (metamer)': 'Connectivity', 'functional group': 'Functional group', 'ring-chain': 'Ring–chain',
        'positional': 'Position', 'skeletal (chain branching)': 'Chain branching', 'skeletal (ring system)': 'Ring system'}
REPS = {'skeletal (chain branching)': ('CCCCCCCC', 'CC(C)CC(C)(C)C'), 'positional': ('Cc1ccccc1C', 'Cc1ccc(C)cc1'), 'ring-chain': ('C1CCCCC1', 'C=CCCCC'),
        'connectivity (metamer)': ('CCOC(C)(C)C', 'CC(C)OC(C)C'), 'functional group': ('CCCCO', 'CCOCC'), 'skeletal (ring system)': ('C1CCCCCC1', 'CC1CCCCC1')}
typ = {}
for r in TYPES:
    mk = (p.relation_en == r).values
    for m in ('JOINT', 'GHGNN'):
        vals, right = [], []
        for ps, ms in oriented(mk, m):
            k = ps >= T_MOD; vals.append(ms[k]); right.append((ms[k] >= T_RIGHT).mean())
        typ[(r, m)] = (np.concatenate(vals), np.mean(right))
    print(f'{r:28s} right% D-MPNN {100 * typ[(r, "JOINT")][1]:.0f}  GH-GNN {100 * typ[(r, "GHGNN")][1]:.0f}')

# c: solvent picks. groups = isomer pair x temperature (rounded K) with >= 30 solvents; per solvent average over studies.
p['Tr'] = p.T_K.round(0)
# v10: cyclohexane in Arumugam et al. (Fluid Phase Equilib 2020;505:112362) is inconsistent with all other data
# (gamma(n-hexane)/gamma(cyclohexane) = 10.5, against a median of 1.59 and a 99th percentile of 3.42 over 2,001
# same-study comparisons). Pairs with this record are kept in the dataset but excluded from the solvent ranking.
FLAG = (p.Study_ID == 'BIB:["fluidphaseequilib","505","112362"]') & (p.Isomer_A.str.lower().eq('cyclohexane') | p.Isomer_B.str.lower().eq('cyclohexane'))
print('pairs excluded from ranking:', int(FLAG.sum()))
p_all = p; p = p[~FLAG].copy()
# studies list the two isomers in either order (D = ln gA - ln gB): orient all records of a pair as in pair_key before averaging
sgn = np.where(p.mol_A == p.pair_key.str.split('|').str[0], 1.0, -1.0)
p['D'] = p.D * sgn
p['qJ'] = p[['D_' + c for c in M['JOINT']]].mean(axis=1) * sgn; p['qG'] = p[['D_' + c for c in M['GHGNN']]].mean(axis=1) * sgn
rows = []; TABLES = {}
for (key, T), g in p.groupby(['pair_key', 'Tr']):
    t = g.groupby('Solvent_Mol_ID').agg(m=('D', 'mean'), j=('qJ', 'mean'), gg=('qG', 'mean'))
    if len(t) < 30: continue
    S = np.exp(t.m.abs())
    pick = lambda col: float(np.median(np.exp(t.m.loc[t[col].abs().nlargest(5).index].abs())))
    TABLES[(key, T)] = t
    rows.append(dict(key=key, T=T, ns=len(t), rel=g.relation_en.iloc[0], none=float(np.median(S)), G=pick('gg'), J=pick('j'),
                     best=float(np.median(np.sort(S.values)[-5:]))))
cg = pd.DataFrame(rows).sort_values('ns', ascending=False)
u = cg.drop_duplicates('key')                                  # one temperature (most solvents) per isomer pair
for col in ('G', 'J'):
    gain = (u[col] - u.none) / (u.best - u.none)
    print(col, 'pairs', len(u), 'beats random', round((u[col] > u.none).mean(), 2), 'median share of attainable gain', round(gain.median(), 2))
# cases: per isomer type, the most-measured pair that is hard to separate (median solvent selectivity < 1.5); functional-group pairs excluded
cases = u[(u.none < 1.5) & (u.rel != 'functional group')].drop_duplicates('rel').copy()
can = lambda s: Chem.MolToSmiles(Chem.MolFromSmiles(s))
NM = {can(k): v for k, v in {'C=CCCCC': '1-hexene', 'C1CCCCC1': 'cyclohexane', 'Cc1ccccc1C': 'o-xylene', 'Cc1cccc(C)c1': 'm-xylene',
      'CCc1ccccc1': 'ethylbenzene', 'CC1CCCC1': 'methylcyclopentane', 'CCOC(C)(C)C': 'ethyl tert-butyl ether', 'CCCOCCC': 'dipropyl ether'}.items()}
rk = {'j': [], 'gg': [], 'rand': []}
for (key, T), t in TABLES.items():
    if key not in set(u.key) or T != float(u.loc[u.key == key, 'T'].iloc[0]): continue
    S_ = np.exp(t.m.abs().values); best = np.argmax(S_)
    for col in ('j', 'gg'):
        order = np.argsort(-t[col].abs().values, kind='stable'); rk[col].append(int(np.where(order == best)[0][0]) + 1)
    rk['rand'].append((len(S_) + 1) / 2)
print('solvents tested until the best one (median over 48 pairs): D-MPNN', np.median(rk['j']), ' GH-GNN', np.median(rk['gg']), ' random', np.median(rk['rand']))
print('  share of pairs where model finds best before random expectation: D-MPNN', np.mean(np.array(rk['j']) < np.array(rk['rand'])).round(2), ' GH-GNN', np.mean(np.array(rk['gg']) < np.array(rk['rand'])).round(2))
cases['label'] = [' / '.join(NM[can(smi[k])] for k in key.split('|')) for key in cases.key]
print(cases[['label', 'T', 'ns', 'rel', 'none', 'G', 'J', 'best']].round(3).to_string(index=False))
assert len(cases) == 5

# ---------------- figure ----------------
fig = plt.figure(figsize=(W / MM, H / MM))
ov = fig.add_axes([0, 0, 1, 1]); ov.set_xlim(0, W); ov.set_ylim(H, 0); ov.axis('off'); ov.set_zorder(-1)
def ax_mm(x, y, w, h):
    a = fig.add_axes([x / W, 1 - (y + h) / H, w / W, h / H]); a.patch.set_alpha(0); return a
def head(x, y, letter, title):
    ov.text(x, y, letter, fontsize=8, fontweight='bold', va='baseline', color=INK)
    ov.text(x + 4.0, y, title, fontsize=8, fontweight='bold', va='baseline', color=INK)
def box(ax, pos, v, width, col, light, vertical=True):
    q10, q25, q50, q75, q90 = np.percentile(v, [10, 25, 50, 75, 90])
    if vertical:
        ax.add_patch(Rectangle((pos - width / 2, q25), width, q75 - q25, facecolor=light, edgecolor=col, lw=0.5, zorder=2))
        ax.plot([pos, pos], [q10, q25], color=col, lw=0.5, zorder=2); ax.plot([pos, pos], [q75, q90], color=col, lw=0.5, zorder=2)
        ax.plot([pos - width / 2, pos + width / 2], [q50, q50], color=col, lw=1.1, zorder=3)
    else:
        ax.add_patch(Rectangle((q25, pos - width / 2), q75 - q25, width, facecolor=light, edgecolor=col, lw=0.5, zorder=2))
        ax.plot([q10, q25], [pos, pos], color=col, lw=0.5, zorder=2); ax.plot([q75, q90], [pos, pos], color=col, lw=0.5, zorder=2)
        ax.plot([q50, q50], [pos - width / 2, pos + width / 2], color=col, lw=1.1, zorder=3)
def draw_mol(smi_, cx, cy, maxw, maxh, bl=1.5, lw=0.45, color=INK):
    m = Chem.MolFromSmiles(smi_); rdDepictor.SetPreferCoordGen(False); rdDepictor.Compute2DCoords(m); Chem.Kekulize(m, clearAromaticFlags=True)
    c = m.GetConformer(); P = np.array([[c.GetAtomPosition(i).x, -c.GetAtomPosition(i).y] for i in range(m.GetNumAtoms())])
    P -= (P.min(0) + P.max(0)) / 2; w, h = np.ptp(P, 0) + 1e-6
    s = min(bl / 1.5, maxw / w, maxh / h); P = P * s + np.array([cx, cy])
    rings = [list(r) for r in m.GetRingInfo().AtomRings()]
    lab = {a.GetIdx(): a.GetSymbol() + ('H' if a.GetTotalNumHs() == 1 else '') for a in m.GetAtoms() if a.GetSymbol() != 'C'}
    for b in m.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx(); p_, q_ = P[i].copy(), P[j].copy(); v = q_ - p_; u_ = v / np.hypot(*v)
        if i in lab: p_ = p_ + u_ * 0.9
        if j in lab: q_ = q_ - u_ * 0.9
        ov.plot([p_[0], q_[0]], [p_[1], q_[1]], color=color, lw=lw, solid_capstyle='round')
        if b.GetBondTypeAsDouble() == 2:
            nrm = np.array([-u_[1], u_[0]]); ring = next((r for r in rings if i in r and j in r), None)
            sg = (1 if np.dot(P[ring].mean(0) - (P[i] + P[j]) / 2, nrm) > 0 else -1) if ring else 1
            off = nrm * sg * 0.32; a1, a2 = P[i] + v * 0.15 + off, P[i] + v * 0.85 + off
            ov.plot([a1[0], a2[0]], [a1[1], a2[1]], color=color, lw=lw, solid_capstyle='round')
    for i, t in lab.items():
        ov.text(P[i][0], P[i][1], t, fontsize=5, color='#D0312D' if t.startswith('O') else color, ha='center', va='center')
YT = [0.67, 1.0, 1.5, 2.0, 3.0]; YL = ['0.67', '1', '1.5', '2', '3']; YR = (0.6, 3.2)

# ---- a: calibration ----
head(0, 3.6, 'a', 'The larger the predicted selectivity, the more it can be trusted')
ax = ax_mm(12.0, 12.0, 78.0, 44.0); ax.set_yscale('log')
ax.axhspan(1 / T_EXP, T_EXP, color=BAND, lw=0, zorder=0)
ax.axhline(1.0, color=INK, lw=0.4, zorder=1)
for i in range(5):
    for m, off in (('JOINT', -0.18), ('GHGNN', 0.18)):
        v, right, _ = cal[m][i]
        box(ax, i + off, v, 0.3, COL[m], LT[m])
        ax.text(i + off, 3.25, f'{100 * right:.0f}', fontsize=6, color=COL[m], ha='center', va='bottom', clip_on=False, fontstyle='italic')
ax.set_ylim(0.6, 3.0); ax.set_yticks(YT); ax.set_yticklabels(YL); ax.minorticks_off()
ax.set_xlim(-0.55, 4.55); ax.set_xticks(range(5)); ax.set_xticklabels(BLAB); ax.tick_params(axis='x', length=0)
ax.set_xlabel('Selectivity predicted by the model'); ax.set_ylabel('Measured selectivity\n(in the predicted direction)', linespacing=1.1)
ax.text(-0.62, 3.25, '% right', fontsize=6, color=INK, ha='right', va='bottom', clip_on=False)
ax.text(0.02, 0.98, 'D-MPNN', transform=ax.transAxes, fontsize=6.5, color=PUR, fontweight='bold', va='top')
ax.text(0.02, 0.91, 'GH-GNN', transform=ax.transAxes, fontsize=6.5, color=GRY, fontweight='bold', va='top')

# ---- b: by isomer type (confident predictions) ----
BX = 97.0
head(BX, 3.6, 'b', 'Reliability depends on the isomer type')
RT, RH = 12.0, 44.0 / 6
bx = ax_mm(BX + 46.0, RT, 30.0, RH * 6); bx.set_xscale('log')
bx.axvspan(1 / T_EXP, T_EXP, color=BAND, lw=0, zorder=0); bx.axvline(1.0, color=INK, lw=0.4, zorder=1)
for i, r in enumerate(TYPES):
    yc = RT + (i + 0.5) * RH
    ov.text(BX, yc, TLAB[r], fontsize=6.5, va='center', color=INK)
    draw_mol(REPS[r][0], BX + 27.5, yc, 9.0, RH - 1.6, bl=1.35); ov.text(BX + 34.0, yc, 'vs', fontsize=5.5, color=INK, ha='center', va='center')
    draw_mol(REPS[r][1], BX + 40.0, yc, 9.0, RH - 1.6, bl=1.35)
    for m, off in (('JOINT', -0.18), ('GHGNN', 0.18)):
        v, right = typ[(r, m)]
        box(bx, i + off, v, 0.3, COL[m], LT[m], vertical=False)
        bx.text(6.4, i + off, f'{100 * right:.0f}', fontsize=6, color=COL[m], va='center', ha='left', clip_on=False, fontstyle='italic')
bx.set_ylim(5.6, -0.6); bx.set_yticks([]); bx.spines['left'].set_visible(False)
bx.set_xlim(0.6, 6.0); bx.set_xticks([0.67, 1, 1.5, 2, 3, 6]); bx.set_xticklabels(['0.67', '1', '1.5', '2', '3', '6']); bx.minorticks_off()
bx.set_xlabel('Measured selectivity')
ov.text(BX + 46.0 + 30.0 + 1.2, RT - 1.3, '% right', fontsize=6, color=INK, va='bottom')

# ---- c: how many solvents must be tested (in model order) before the best one is found ----
Y2 = 70.0
head(0, Y2, 'c', 'Ranking solvents with a model finds the best one with fewer tests')
def rank_of_best(t, col):
    S_ = np.exp(t.m.abs().values); order = np.argsort(-t[col].abs().values, kind='stable')
    return int(np.where(order == np.argmax(S_))[0][0]) + 1
show = cases.iloc[[1, 4, 0, 3]]
SHORT = {'ethyl tert-butyl ether / dipropyl ether': 'ETBE /\ndipropyl ether', 'methylcyclopentane / cyclohexane': 'methylcyclopentane /\ncyclohexane',
         'o-xylene / m-xylene': 'o-xylene /\nm-xylene', '1-hexene / cyclohexane': '1-hexene /\ncyclohexane'}
grp = []
for _, r in show.iterrows():
    t = TABLES[(r.key, r['T'])]
    grp.append((SHORT[r.label] + f'\n({r.ns} solvents)', rank_of_best(t, 'j'), rank_of_best(t, 'gg'), (len(t) + 1) / 2))
grp.append((f'All {len(rk["j"])} pairs\n(median)', float(np.median(rk['j'])), float(np.median(rk['gg'])), float(np.median(rk['rand']))))
for g_ in grp: print('c', g_[0].replace('\n', ' '), g_[1:])
cx = ax_mm(12.0, Y2 + 9.0, 160.0, 30.0)
SER = [('D-MPNN ranking', PUR), ('GH-GNN ranking', GRY), ('Random order', '#CFCDC4')]
bw = 0.24
for gi, (lab_, *vals) in enumerate(grp):
    for si, ((nm, c_), v) in enumerate(zip(SER, vals)):
        x_ = gi + (si - 1) * (bw + 0.02)
        cx.bar(x_, v, bw, color=c_, lw=0, zorder=2)
        cx.text(x_, v + 1.5, (f'{v:.1f}' if si < 2 else f'{v:.0f}') if v != int(v) else f'{int(v)}', fontsize=6, color=INK, ha='center', va='bottom')
cx.axvline(len(grp) - 1.5, color='#D9D8D3', lw=0.5, zorder=0)
cx.set_xlim(-0.55, len(grp) - 0.45); cx.set_xticks(range(len(grp))); cx.set_xticklabels([g_[0] for g_ in grp], fontsize=6.5, linespacing=1.15)
cx.tick_params(axis='x', length=0)
ymax_c = max(max(g_[1:]) for g_ in grp)
cx.set_ylim(0, ymax_c * 1.12); cx.set_ylabel('Solvents tested before\nfinding the best one', linespacing=1.1)
for i_, (nm, c_) in enumerate(SER):
    x_ = 100.0 + i_ * 27.0; y_ = Y2 + 4.6
    ov.add_patch(Rectangle((x_, y_ - 1.2), 2.4, 2.4, color=c_, lw=0))
    ov.text(x_ + 3.4, y_, nm, fontsize=6.5, color=c_ if c_ != '#CFCDC4' else '#8E8C82', va='center')

for ext, kw in (('svg', {}), ('pdf', {}), ('png', {'dpi': 600})):
    fig.savefig(OUT / f'Fig6.{ext}', **kw)
print('saved')
