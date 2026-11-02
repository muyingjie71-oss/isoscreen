#!/usr/bin/env python
"""Figure 1. Isomer pairs, how selectivity is obtained, and how many pairs lie below the experimental and model resolution.
Reads plot_data/fig1. Writes SVG, then PDF and PNG through headless Chromium (playwright).
Usage: python fig1.py <this folder> <out_dir>"""
import sys, json, math
from pathlib import Path
import numpy as np, pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, rdDepictor

ROOT, OUT = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve(); OUT.mkdir(parents=True, exist_ok=True)
PD = ROOT / 'plot_data' / 'fig1'
NAME, W, H = 'Fig1', 183.0, 114.5
PAD = 2.4; SCL = (W - 2 * PAD) / W; WF, HF = W, round(H * SCL + 2 * PAD + 0.4, 1)
Z_A, Z_B, Z_C = '#EAF1F7', '#F3F3EF', '#F4F6F8'
INK, GREY, MID, HAIR = '#2C2C2A', '#5F5E5A', '#9A9A96', '#D9D9D5'
PUR, PUR_L = '#534AB7', '#B9B4EC'
Z0, Z1, Z2 = '#2C2C2A', '#AFA9EC', '#C9CFCC'          # zone greys: neither / experiments only / both
CPK = {'C': '#3A3A38', 'H': '#FFFFFF', 'N': '#3B62B0', 'O': '#D0312D', 'S': '#D9A82B'}
CPK_EDGE = {'C': '#1E1E1D', 'H': '#6E6E6B', 'N': '#25407A', 'O': '#8E1F1C', 'S': '#8A6A17'}
RAD = {'H': 0.22, 'C': 0.42, 'N': 0.40, 'O': 0.40, 'S': 0.52}  # Å, drawing radii
PT = 0.3528 / SCL   # keep true point sizes after the zone inset scale
LW, HL = 0.18, 0.1
S6, S7, S8 = 6.0, 7.0, 8.0
BL = 2.1

num = json.loads((PD / 'fig1_numbers.json').read_text())
pairs = pd.read_csv(PD / 'fig1c_all_pairs.csv')
EXP_RES, MOD_RES = num['exp_resolution'], num['model_resolution_DMPNN']
S = pairs['S_measured'].values
fr = [(S < EXP_RES).mean(), ((S >= EXP_RES) & (S < MOD_RES)).mean(), (S >= MOD_RES).mean()]
pct = [round(100 * f) for f in fr]
assert pct == [17, 38, 45], pct

E = []
def T(x, y, s, pt=S7, c=INK, anchor='start', w='normal', extra=''):
    E.append(f'<text x="{x:.2f}" y="{y:.2f}" font-size="{pt * PT:.3f}" fill="{c}" text-anchor="{anchor}" '
             f'font-weight="{w}" dominant-baseline="central" {extra}>{s}</text>')
def ln(x1, y1, x2, y2, c=INK, w=LW, extra='', out=None):
    (E if out is None else out).append(f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{c}" stroke-width="{w}" stroke-linecap="round" {extra}/>')
def rect(x, y, w, h, stroke=INK, sw=LW, fill='none'):
    E.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
def circ(x, y, r, fill, stroke, w, out=None):
    (E if out is None else out).append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{r:.3f}" fill="{fill}" stroke="{stroke}" stroke-width="{w}"/>')
def arr(x1, y1, x2, y2, c=INK, dash=''):
    mk = {INK: 'aI', PUR: 'aP', MID: 'aM'}[c]
    E.append(f'<path d="M{x1:.2f} {y1:.2f} L{x2:.2f} {y2:.2f}" stroke="{c}" stroke-width="{LW}" fill="none" {dash} marker-end="url(#{mk})"/>')
def sub(base, s, pt):
    return f'{base}<tspan font-size="{pt * 0.7 * PT:.3f}" dy="{pt * 0.24 * PT:.3f}">{s}</tspan><tspan dy="{-pt * 0.24 * PT:.3f}">​</tspan>'
def head(x, y, letter, title):
    T(x, y, letter, S8, INK, w='bold'); T(x + 4.2, y, title, S7, INK, w='bold')
DEFS = ''.join(f'<marker id="{k}" viewBox="0 0 6 6" refX="5.2" refY="3" markerWidth="1.25" markerHeight="1.25" markerUnits="userSpaceOnUse" orient="auto">'
               f'<path d="M0 0 L6 3 L0 6 z" fill="{c}"/></marker>' for k, c in (('aI', INK), ('aP', PUR), ('aM', MID)))

# ---------- 2D skeletal formula, heteroatom labels in CPK colour
def mol2d(smi, cx, cy, maxw, maxh, bl=BL, fs=S6):
    m = Chem.MolFromSmiles(smi); rdDepictor.SetPreferCoordGen(False); rdDepictor.Compute2DCoords(m)
    Chem.Kekulize(m, clearAromaticFlags=True); conf = m.GetConformer()
    P = np.array([[conf.GetAtomPosition(i).x, -conf.GetAtomPosition(i).y] for i in range(m.GetNumAtoms())])
    P -= (P.min(0) + P.max(0)) / 2
    w, h = np.ptp(P, 0) + 1e-6; sc = min(bl / 1.5, (maxw - 0.6) / w, (maxh - 0.6) / h); P = P * sc + [cx, cy]
    rings = [list(r) for r in m.GetRingInfo().AtomRings()]; lab = {}
    for a in m.GetAtoms():
        if a.GetSymbol() != 'C':
            s = a.GetSymbol() + ('H' if a.GetTotalNumHs() == 1 else ''); q = a.GetFormalCharge()
            if q: s += f'<tspan font-size="{fs * PT * 0.75:.3f}" dy="{-fs * PT * 0.4:.3f}">{"+" if q > 0 else "−"}</tspan>'
            lab[a.GetIdx()] = (s, CPK[a.GetSymbol()])
    cut = fs * PT * 0.55
    for b in m.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx(); p_, q_ = P[i].copy(), P[j].copy(); v = q_ - p_
        u = v / np.hypot(*v); nr = np.array([-u[1], u[0]])
        p2, q2 = (p_ + u * cut if i in lab else p_), (q_ - u * cut if j in lab else q_)
        o = b.GetBondTypeAsDouble(); ring = next((r for r in rings if i in r and j in r), None)
        if o in (1, 3) or (o == 2 and ring is not None): ln(*p2, *q2)
        if o == 3:
            off = nr * sc * 0.18
            for sg in (1, -1): ln(*(p2 + sg * off), *(q2 + sg * off))
        if o == 2 and ring is not None:
            sg = 1 if np.dot(P[ring].mean(0) - (p_ + q_) / 2, nr) > 0 else -1; off = nr * sg * sc * 0.24
            ln(*(p_ + (q_ - p_) * 0.15 + off), *(p_ + (q_ - p_) * 0.85 + off))
        elif o == 2:
            off = nr * sc * 0.13
            for sg in (1, -1): ln(*(p2 + sg * off), *(q2 + sg * off))
    for i, (t, c) in lab.items(): T(P[i][0], P[i][1], t, fs, c, 'middle')

# ---------- 3D ball-and-stick (flat CPK fills, painter's order, no gradients)
def mol3d(smi, cx, cy, maxw, maxh, tilt=(18, -25, 0), seed=7, smax=2.6):
    m = Chem.AddHs(Chem.MolFromSmiles(smi))
    AllChem.EmbedMolecule(m, randomSeed=seed); AllChem.MMFFOptimizeMolecule(m, maxIters=2000)
    X = m.GetConformer().GetPositions(); X -= X.mean(0)
    _, _, vt = np.linalg.svd(X, full_matrices=False); X = X @ vt.T          # long axis -> x
    ax, ay, az = (math.radians(a) for a in tilt)
    Rx = np.array([[1, 0, 0], [0, math.cos(ax), -math.sin(ax)], [0, math.sin(ax), math.cos(ax)]])
    Ry = np.array([[math.cos(ay), 0, math.sin(ay)], [0, 1, 0], [-math.sin(ay), 0, math.cos(ay)]])
    Rz = np.array([[math.cos(az), -math.sin(az), 0], [math.sin(az), math.cos(az), 0], [0, 0, 1]])
    X = X @ (Rz @ Ry @ Rx).T
    sym = [a.GetSymbol() for a in m.GetAtoms()]
    xy = np.c_[X[:, 0], -X[:, 1]]; r = np.array([RAD[t] for t in sym])
    lo, hi = (xy - r[:, None]).min(0), (xy + r[:, None]).max(0)
    sc = min(smax, maxw / (hi - lo)[0], maxh / (hi - lo)[1])
    xy = (xy - (lo + hi) / 2) * sc + [cx, cy]
    z = X[:, 2]; persp = 1 + 0.08 * (z - z.mean()) / (np.ptp(z) + 1e-6)
    items = [((z[b.GetBeginAtomIdx()] + z[b.GetEndAtomIdx()]) / 2 - 0.05, 'b', b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in m.GetBonds()]
    items += [(z[i], 'a', i, None) for i in range(len(sym))]
    for _, kind, i, j in sorted(items, key=lambda t: t[0]):
        if kind == 'b': ln(*xy[i], *xy[j], '#A3A3A0', 0.26 * sc / 2.2)
        else: circ(*xy[i], RAD[sym[i]] * sc * persp[i], CPK[sym[i]], CPK_EDGE[sym[i]], 0.07)
    return xy

# ================= a
head(0, 3.2, 'a', 'Isomers boil a few K apart')
rows = (('CCCCCC', 'CCC(C)CC', 5.5), ('Cc1cccc(C)c1', 'Cc1ccccc1C', 5.3), ('C=CCCCC', 'CC1CCCC1', 8.3),
        ('CCOC(C)=O', 'CCC(=O)OC', 2.3), ('CC1CCCCC1', 'CCC1CCCC1', 2.6))
xA, xB, xT, TW = 10.5, 29.5, 47.0, 49.0
T(xA, 9.4, 'Isomer A', S6, INK, 'middle'); T(xB, 9.4, 'Isomer B', S6, INK, 'middle'); T(xT, 9.4, sub('ΔT', 'b', S6) + ' (K)', S6, INK, 'end')
y0, rh, gp = 11.6, 9.6, 0.9
for k, (sa, sb, dT) in enumerate(rows):
    ya = y0 + rh * k; yc = ya + (rh - gp) / 2
    E.append(f'<rect x="0" y="{ya:.2f}" width="{TW}" height="{rh - gp:.2f}" fill="white"/>')
    mol2d(sa, xA, yc, 17, 7.6); mol2d(sb, xB, yc, 17, 7.6); T(xT, yc, f'{dT:.1f}', S7, INK, 'end')

# ================= b
BX = 57.0
head(BX, 3.2, 'b', 'Selectivity is measured, or predicted from structure')
PX0, PX1, PY0, PY1 = BX, BX + 44.0, 8.6, 58.4
E.append(f'<rect x="{PX0}" y="{PY0}" width="{PX1 - PX0}" height="{PY1 - PY0}" fill="white"/>')
ys = (15.4, 27.4, 40.6, 53.0); c2, c3 = BX + 18.0, BX + 33.0
for yy, t in zip(ys, ('Isomer A', 'Isomer B', 'Solvent', 'Temperature')): T(BX + 1.6, yy, t, S6, INK)
mol2d('CCCCCC', c2, ys[0], 10, 6); mol3d('CCCCCC', c3, ys[0], 16, 8.5, tilt=(15, 0, 0))
mol2d('CCC(C)CC', c2, ys[1], 10, 7); mol3d('CCC(C)CC', c3, ys[1], 14, 9.5, tilt=(-30, 30, 0))
mol2d('CCn1cc[n+](C)c1', c2 + 0.4, ys[2] - 1.8, 14, 7.4, bl=2.1); mol2d('N#C[S-]', c2 + 0.4, ys[2] + 4.2, 9, 2.5, bl=2.9)
cat = mol3d('CCn1cc[n+](C)c1', c3 - 2.4, ys[2] - 0.2, 13.0, 9.5, tilt=(30, 0, 0))
an = mol3d('N#C[S-]', c3 + 7.4, ys[2] + 0.6, 5.2, 3, tilt=(0, 0, 0))
T(cat[:, 0].max() + 0.5, cat[:, 1].min() + 0.6, '+', S6, INK, 'middle'); T(an[:, 0].mean(), an[:, 1].min() - 1.6, '−', S6, INK, 'middle')
T(c2, ys[3], '<tspan font-style="italic">T</tspan>', S7, INK, 'middle')
for yy in ((ys[0] + ys[1]) / 2 + 0.6, (ys[1] + ys[2]) / 2 + 0.4, (ys[2] + ys[3]) / 2 + 0.8):
    ln(BX + 1.6, yy, PX1 - 1.6, yy, HAIR, HL)
bk = PX1 + 2.6
ln(bk - 1.4, 11.0, bk, 11.0); ln(bk - 1.4, 56.0, bk, 56.0); ln(bk, 11.0, bk, 56.0)
yE, yM = 17.5, 43.5; tr = bk + 3.5
ln(bk, 33.5, tr, 33.5); ln(tr, yE, tr, yM)
X0, X1 = tr + 3.6, tr + 37.6
arr(tr, yE, X0 - 0.3, yE); arr(tr, yM, X0 - 0.3, yM)
cx = (X0 + X1) / 2
rect(X0, 10.5, X1 - X0, 14.0, fill='white')
T(cx, 14.7, 'Experiment', S7, INK, 'middle', 'bold'); T(cx, 19.8, f"{num['n_studies']} studies, {num['n_observations']:,} values", S6, INK, 'middle')
MY0, MY1 = 32.5, 54.5
rect(X0, MY0, X1 - X0, MY1 - MY0, PUR, fill='white')
T(cx, MY0 + 3.2, 'Model', S7, PUR, 'middle', 'bold')
# network sketch: molecular graph -> graph features + solvent & T -> network -> output
gy = MY0 + 9.6; gx = X0 + 4.4; gr = 1.9
hexp = [(gx + gr * math.cos(math.radians(30 + 60 * k)), gy + gr * math.sin(math.radians(30 + 60 * k))) for k in range(6)]
for k in range(6): ln(*hexp[k], *hexp[(k + 1) % 6], PUR, 0.15)
for (x, y) in hexp: circ(x, y, 0.5, 'white', PUR, 0.15)
arr(gx + gr + 0.9, gy, gx + gr + 2.9, gy, PUR)
BLU = ('#1F3B66', '#3B62B0', '#7FA6D6', '#B9CDE8', '#E4ECF6')
fx0 = gx + gr + 3.4; sw_ = 0.62
for k, ci in enumerate((1, 3, 0, 2, 4, 1, 3, 2)):
    E.append(f'<rect x="{fx0 + k * sw_:.2f}" y="{gy - 3.0:.2f}" width="{sw_ - 0.12:.2f}" height="6.0" fill="{BLU[ci]}"/>')
T(fx0 + 4 * sw_, gy + 4.6, 'Graph', S6 * 0.85, INK, 'middle'); T(fx0 + 4 * sw_, gy + 6.6, 'features', S6 * 0.85, INK, 'middle')
HM = ('#3B62B0', '#7FA6D6', '#D5DEEA', '#F2C9A0', '#E39A5B', '#C0652A')
hx0 = fx0 + 8 * sw_ + 1.6; cs = 1.5
grid = ((0, 3, 1, 4), (2, 5, 0, 3), (4, 1, 3, 2), (1, 2, 5, 0))
for r_, row in enumerate(grid):
    for c_, v in enumerate(row):
        E.append(f'<rect x="{hx0 + c_ * cs:.2f}" y="{gy - 3.0 + r_ * cs:.2f}" width="{cs - 0.1:.2f}" height="{cs - 0.1:.2f}" fill="{HM[v]}"/>')
T(hx0 + 2 * cs, gy + 4.6, 'Solvent', S6 * 0.85, INK, 'middle'); T(hx0 + 2 * cs, gy + 6.6, '&amp; T', S6 * 0.85, INK, 'middle')
nx0 = hx0 + 4 * cs + 1.0
arr(nx0, gy, nx0 + 1.9, gy, PUR)
lx = [nx0 + 3.0, nx0 + 6.0, nx0 + 9.0]; ln_ = [4, 3, 1]; sp = 1.7
layers = [[(lx[i], gy + (k - (n - 1) / 2) * sp) for k in range(n)] for i, n in enumerate(ln_)]
for a_, b_ in zip(layers[:-1], layers[1:]):
    for p in a_:
        for q in b_: ln(*p, *q, PUR_L, 0.08)
for L_ in layers:
    for p in L_: circ(*p, 0.5, 'white', PUR, 0.15)
T(cx, MY1 - 2.6, 'D-MPNN, GH-GNN', S6, INK, 'middle', extra='font-style="italic"')
arr(cx, 24.8, cx, MY0 - 0.3, INK); T(cx + 1.5, 28.9, 'trains', S6, INK)
OX = X1 + 4.0
for yy, col in ((yE, INK), (yM, PUR)):
    arr(X1 + 0.3, yy, OX - 0.4, yy, col); T(OX, yy, sub('γ', 'A', S7) + ', ' + sub('γ', 'B', S7), S7, col)
BR = OX + 8.6; ym = (yE + yM) / 2
ln(BR - 1.4, yE, BR, yE); ln(BR - 1.4, yM, BR, yM); ln(BR, yE, BR, yM); arr(BR, ym, BR + 3.0, ym)
SX = BR + 3.6; SW = W - 0.4 - SX
rect(SX, ym - 6.0, SW, 12.0, INK, 0.3, fill='white')
T(SX + SW / 2, ym - 2.4, 'Selectivity', S7, INK, 'middle', 'bold'); T(SX + SW / 2, ym + 2.4, sub('γ', 'A', S7) + '/' + sub('γ', 'B', S7), S7, INK, 'middle')

# ================= c
CY = 64.0
head(0, CY, 'c', 'Over half of isomer pairs differ by less than models resolve')
px0, px1, py0, py1 = 10.5, W - 3.0, 106.0, 82.0
smax, cmax = 1.50, 2400.0
Xs = lambda v: px0 + (v - 1.0) / (smax - 1.0) * (px1 - px0)
Yc = lambda c: py0 - c / cmax * (py0 - py1)
edges = sorted(set(np.round(np.arange(1.0, smax + 1e-9, 0.01), 4).tolist()) | {EXP_RES, MOD_RES})
hist, _ = np.histogram(S, bins=edges)
for lo, hi, n in zip(edges[:-1], edges[1:], hist):
    hgt = n * 0.01 / (hi - lo)                      # pairs per 0.01 of selectivity
    col = Z0 if hi <= EXP_RES + 1e-12 else (Z1 if hi <= MOD_RES + 1e-12 else Z2)
    gap = 0.05 if hi - lo > 0.009 else 0.0
    l_ = Xs(lo) + (gap if lo not in (EXP_RES, MOD_RES) else 0); r_ = Xs(hi) - (gap if hi not in (EXP_RES, MOD_RES) else 0)
    E.append(f'<rect x="{l_:.3f}" y="{Yc(hgt):.3f}" width="{r_ - l_:.3f}" height="{py0 - Yc(hgt):.3f}" fill="{col}"/>')
ln(px0, py0, px1, py0); ln(px0, py0, px0, py1 - 1.0)
for t in (1.0, 1.1, 1.2, 1.3, 1.4, 1.5):
    ln(Xs(t), py0, Xs(t), py0 + 1.0); T(Xs(t), py0 + 3.2, f'{t:.1f}', S7, INK, 'middle')
for t in (0, 1000, 2000):
    ln(px0 - 1.0, Yc(t), px0, Yc(t)); T(px0 - 1.7, Yc(t), f'{t // 1000}', S7, INK, 'end')
T((px0 + px1) / 2, py0 + 7.0, 'Measured selectivity', S7, INK, 'middle')
T(2.4, (py0 + py1) / 2, 'Isomer pairs (thousands)', S7, INK, 'middle', extra=f'transform="rotate(-90 2.4 {(py0 + py1) / 2:.2f})"')
zy = 75.0
ln(Xs(EXP_RES), py0, Xs(EXP_RES), zy, INK, 0.25)
ln(Xs(MOD_RES), py0, Xs(MOD_RES), zy, PUR, 0.25, 'stroke-dasharray="1.2 0.8"')
T(Xs(EXP_RES) + 1.2, 78.6, f'Experiments {EXP_RES:.2f}', S6, INK)
T(Xs(MOD_RES) + 1.2, 78.6, f'D-MPNN {MOD_RES:.2f}', S6, PUR)
for (a_, b_), lab, pc_ in (((1.0, EXP_RES), 'Neither', pct[0]), ((EXP_RES, MOD_RES), 'Only experiments', pct[1]), ((MOD_RES, smax), 'Both', pct[2])):
    xa, xb = Xs(a_) + 0.35, Xs(b_) - 0.35
    ln(xa, zy, xb, zy); ln(xa, zy, xa, zy + 1.0); ln(xb, zy, xb, zy + 1.0)
    T((xa + xb) / 2, zy - 2.0, f'{pc_}%', S7, INK, 'middle', 'bold')
    T((xa + xb) / 2, zy - 5.2, lab, S6, INK, 'middle')

p_ = PAD / SCL
ZONES = (f'<rect x="{-p_:.2f}" y="{-p_:.2f}" width="{52.2 + p_:.2f}" height="{61.0 + p_:.2f}" fill="{Z_A}"/>'
         f'<rect x="53.4" y="{-p_:.2f}" width="{W - 53.4 + p_:.2f}" height="{61.0 + p_:.2f}" fill="{Z_B}"/>'
         f'<rect x="{-p_:.2f}" y="62.3" width="{W + 2 * p_:.2f}" height="{(HF - PAD) / SCL - 62.3:.2f}" fill="{Z_C}"/>')
W_, H_ = W, H; W, H = WF, HF
svg_ = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}mm" height="{H}mm" viewBox="0 0 {W} {H}" '
        f'font-family="Arial, \'Liberation Sans\', Helvetica, sans-serif"><defs>{DEFS}</defs><rect width="{W}" height="{H}" fill="white"/>'
        f'<g transform="translate({PAD} {PAD}) scale({SCL:.5f})">' + ZONES + ''.join(E) + '</g></svg>')
(OUT / f'{NAME}.svg').write_text(svg_, encoding='utf-8')
html = (f'<!doctype html><html><head><meta charset="utf-8"><title>Figure 1</title><style>@page{{size:{W}mm {H}mm;margin:0}}'
        f'html,body{{margin:0;background:white}}svg{{display:block}}</style></head><body>{svg_}</body></html>')
(OUT / f'{NAME}.html').write_text(html, encoding='utf-8')
from playwright.sync_api import sync_playwright
with sync_playwright() as pw:
    br = pw.chromium.launch(); pg = br.new_page(viewport={'width': int(W / 25.4 * 96) + 1, 'height': int(H / 25.4 * 96) + 1}, device_scale_factor=600 / 96)
    pg.goto((OUT / f'{NAME}.html').resolve().as_uri()); pg.pdf(path=str(OUT / f'{NAME}.pdf'), width=f'{W}mm', height=f'{H}mm', print_background=True)
    pg.locator('svg').screenshot(path=str(OUT / f'{NAME}.png')); br.close()
print(pct, [round(f, 4) for f in fr])
