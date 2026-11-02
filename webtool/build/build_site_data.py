"""Write the data and model files the web tool loads (site/data, site/model) from data/, validation/ and the build assets.
Usage: python build_site_data.py [site folder]"""
import sys, csv, json, math, shutil, collections, re
from pathlib import Path
import numpy as np, pandas as pd
from rdkit import Chem, RDLogger; RDLogger.DisableLog('rdApp.*')
from chemicals.identifiers import search_chemical
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import DATA as D, VALIDATION as V, ASSETS as A, SITE
SITE = Path(sys.argv[1]) if len(sys.argv) > 1 else SITE
OUT = SITE / 'data'; MOD = SITE / 'model'; OUT.mkdir(parents=True, exist_ok=True); (MOD / 'onnx').mkdir(parents=True, exist_ok=True)
def rows(p):
    with open(p, encoding='utf-8-sig', newline='') as f: return list(csv.DictReader(f))
dump = lambda p, o: Path(p).write_text(json.dumps(o, ensure_ascii=False, separators=(',', ':')))
REL = {'metameric': 'connectivity (metamer)'}   # isomer type labels used by the site
ARUMUGAM = 'BIB:["fluidphaseequilib","505","112362"]'

# ------------------------------------------------------------------ model files
for f in sorted((A / 'onnx32').glob('*.onnx')): shutil.copyfile(f, MOD / 'onnx' / (f.name + '.bin'))
wm = json.loads((A / 'weights_js.json').read_text()); w32 = np.fromfile(A / 'weights_js.f32', '<f4')
enc_idx, enet_idx, enc, enet, oe, on = {}, {}, [], [], 0, 0
for name, (off, shape) in wm['index'].items():
    n = int(np.prod(shape)); a = w32[off:off + n]
    if 'enet2/W' in name: enet_idx[name] = [on, shape]; enet.append(a); on += n
    else: enc_idx[name] = [oe, shape]; enc.append(a); oe += n
np.concatenate(enc).astype('<f4').tofile(MOD / 'enc.f32.bin'); np.concatenate(enet).astype('<f2').tofile(MOD / 'enet.f16.bin')
mm = {'seeds': wm['seeds'], 'dmpnn': wm['dmpnn'], 'ghgnn': wm['ghgnn'], 'enc_index': enc_idx, 'enet_index': enet_idx,
      'precision': 'ONNX heads and encoders float32; GH-GNN edge network and solvent encodings stored as float16'}
dump(MOD / 'meta.json', mm)

# ------------------------------------------------------------------ solvents
sv = json.loads((A / 'solvents.json').read_text())
for n in ('dmpnn', 'ghgnn'): shutil.copyfile(A / f'{n}_solvents.f16', OUT / f'emb_{n}.f16.bin')
KIND = {'ionic liquid': 'IL', 'molecular solvent': 'M'}
SRC = {'training data': 'dataset', 'ion combination (training cations x training anions)': 'ion combination'}
solv = [{'id': s['solvent_id'], 'name': s['name'], 'short': s['name_short'], 'kind': KIND[s['kind']], 'cls': s['cation_class'], 'measured': s['status'] == 'measured',
         'domain': s['domain'], 'sim': float(s['max_sim']), 'nrec': int(s['n_records']), 'smiles': s['smiles'], 'cas': s['cas'],
         'src': SRC.get(s['source'], 'CHEM21' if s['source'].startswith('CHEM21') else 'Langner 2022'), 'hba': s['hba'], 'hbd': s['hbd']} for s in sv]
dump(OUT / 'solvents.json', solv)
sidx = {s['id']: i for i, s in enumerate(solv)}

# ------------------------------------------------------------------ solutes (dataset isomers) with synonyms
mol = {r['Mol_ID']: r for r in rows(D / 'molecules.csv')}
t1 = rows(D / 'activity_coefficients.csv'); names = collections.defaultdict(collections.Counter); trange = collections.defaultdict(list)
for r in t1: names[r['Solute_Mol_ID']][r['Solute']] += 1; trange[r['Solute_Mol_ID']].append(float(r['T_K']))
fam = {r['Solute_Mol_ID']: r['Family_ID'] for r in t1}
solutes = []
for mid in sorted(names):
    nm = names[mid].most_common(1)[0][0]; syn = []
    try:
        c = search_chemical('InChIKey=' + mol[mid]['InChIKey'])
        ok = lambda x: x and len(x) <= 32 and ', ' not in x and not re.search(r'\d{2,}-\d{2}-\d|\(\d+ci\)|^[a-z]{2,4}-[a-z]{0,2}\d|[;:]', x, re.I) and re.fullmatch(r"[A-Za-z0-9 ,()'\-+\[\]]+", x)
        syn = [x for x in dict.fromkeys([c.common_name] + list(c.synonyms or [])) if ok(x)][:6]
    except Exception: pass
    syn = [x for x in dict.fromkeys([nm] + [n for n, _ in names[mid].most_common(4)] + syn) if x]
    solutes.append({'id': mid, 'name': nm, 'syn': syn[1:], 'smiles': mol[mid]['SMILES'], 'inchikey': mol[mid]['InChIKey'], 'family': fam[mid]})
dump(OUT / 'solutes.json', solutes)

# ------------------------------------------------------------------ measured pairs (task2), oriented by sorted Mol_ID
t2 = rows(D / 'isomer_pairs.csv'); rec = {r['canonical_record_id']: r for r in t1}
rel_cd = {}
for r in rows(V / 'pairs_common_domain.csv'): rel_cd[r['pair_key']] = r['relation_en']
pairs = {}
JOURNAL = {'jmolliq': 'J. Mol. Liq.', 'jchemengdata': 'J. Chem. Eng. Data', 'jchemengdab': 'J. Chem. Eng. Data', 'jsolchem': 'J. Solution Chem.', 'jchemthermodyn': 'J. Chem. Thermodyn.',
 'fluidphaseequilib': 'Fluid Phase Equilib.', 'jphyschemb': 'J. Phys. Chem. B', 'jphyschem': 'J. Phys. Chem.', 'indengchemres': 'Ind. Eng. Chem. Res.', 'rscadv': 'RSC Adv.',
 'jchromatogr': 'J. Chromatogr.', 'jchromatogra': 'J. Chromatogr. A', 'chemengj': 'Chem. Eng. J.', 'acssustainablechemeng': 'ACS Sustainable Chem. Eng.', 'molecules': 'Molecules',
 'intjthermophys': 'Int. J. Thermophys.', 'chemengcommun': 'Chem. Eng. Commun.', 'frontiersofchemicalengineeringinchina': 'Front. Chem. Eng. China',
 'japplchembiotechnol': 'J. Appl. Chem. Biotechnol.', 'safrjchemeng': 'S. Afr. J. Chem. Eng.', 'intjmolsci': 'Int. J. Mol. Sci.', 'jfluorinechem': 'J. Fluorine Chem.',
 'analchem': 'Anal. Chem.', 'journaldechimiephysiqueetdephysicochimiebiologique': 'J. Chim. Phys.', 'indengchemprocessdesdev': 'Ind. Eng. Chem. Process Des. Dev.',
 'jcolloidinterfacesci': 'J. Colloid Interface Sci.', 'revroumchim': 'Rev. Roum. Chim.', 'indengchemfundam': 'Ind. Eng. Chem. Fundam.', 'jliqchromatogr': 'J. Liq. Chromatogr.',
 'transfaradaysoc': 'Trans. Faraday Soc.', 'chemengtechnol': 'Chem. Eng. Technol.', 'latinamericanappliedresearch': 'Lat. Am. Appl. Res.', 'kholodtekhtekhnol': 'Kholod. Tekh. Tekhnol.'}
def study_label(s):
    if s.startswith('DOI:'): return s[4:], 'https://doi.org/' + s[4:]
    if s.startswith('BIB:'):
        j, v, p = json.loads(s[4:]); return f'{JOURNAL.get(j, j)} {v}, {p}', None
    if s.startswith('TITLE:'): return s[6:][:120], None
    return s, None
for r in t2:
    a, b = rec[r['canonical_record_id_A']]['Solute_Mol_ID'], rec[r['canonical_record_id_B']]['Solute_Mol_ID']
    d = float(r['D_AB'])
    if a > b: a, b, d = b, a, -d
    key = a + '|' + b
    p = pairs.setdefault(key, {'a': a, 'b': b, 'rel': rel_cd.get(key) or REL.get(r['Isomer_Relation'], r['Isomer_Relation']), 'family': r['Family_ID'], 'm': []})
    lab, url = study_label(r['Study_ID'])
    note = 1 if (r['Study_ID'] == ARUMUGAM and ('cyclohexane' in (mol[a]['SMILES'], mol[b]['SMILES']) or 'C1CCCCC1' in (mol[a]['SMILES'], mol[b]['SMILES']))) else 0
    p['m'].append([sidx.get(r['Solvent_Mol_ID'], r['Solvent']), round(float(r['T_K']), 2), round(d, 5), lab, url or '', note])
plist = []
for key, p in sorted(pairs.items()):
    m = p['m']; S = [math.exp(abs(x[2])) for x in m]
    plist.append({'key': key, 'a': p['a'], 'b': p['b'], 'rel': p['rel'], 'family': p['family'], 'n_meas': len(m), 'n_solv': len({str(x[0]) for x in m}),
                  'Tmin': min(x[1] for x in m), 'Tmax': max(x[1] for x in m), 'Smax': round(max(S), 4), 'm': m})
dump(OUT / 'pairs.json', plist)

# ------------------------------------------------------------------ validation numbers (validation/, recomputed where needed)
cal = rows(V / 'calibration_by_predicted_S.csv'); summ = rows(V / 'model_summary_by_regime.csv'); typ = rows(V / 'hit_rate_by_isomer_type.csv')
o = pd.read_csv(V / 'oof_predictions_ln_gamma.csv'); pc = pd.read_csv(V / 'pairs_common_domain.csv')
cols = lambda pre: [c for c in o.columns if c.startswith(pre)]
gam = {}
for reg, pre in (('new_isomers', 'newiso_'), ('known_isomers_new_solvents', 'newsolv_')):
    q = (o[cols(pre + 'DMPNN_')].mean(1) + o[cols(pre + 'GHGNN_')].mean(1)) / 2
    gam[reg] = round(float(np.exp(np.median(np.abs(q - o.ln_gamma_inf)))) - 1, 4)
# resolution by similarity of the held-out solvent (consensus, solvent-held-out test)
m = dict(zip(o.canonical_record_id, range(len(o)))); ia = pc.canonical_record_id_A.map(m).values; ib = pc.canonical_record_id_B.map(m).values
qd = np.mean([o[c].values[ia] - o[c].values[ib] for c in cols('newsolv_DMPNN_')], 0); qg = np.mean([o[c].values[ia] - o[c].values[ib] for c in cols('newsolv_GHGNN_')], 0)
err = np.abs((qd + qg) / 2 - pc.D.values)
sim = pd.read_csv(V / 'test_solvent_similarity_solvent_out.csv').set_index('Solvent_Mol_ID')
ms = pc.Solvent_Mol_ID.map(sim.max_sim).values; il = pc.Solvent_Mol_ID.map(sim.is_IL).values
dom = []
for lab, lo, hi in (('in domain', 0.5, 9), ('near domain', 0.3, 0.5), ('outside domain', -1, 0.3)):
    for kind, mask in (('molecular solvents', il == False), ('ionic liquids', il == True)):
        k = (ms >= lo) & (ms < hi) & mask
        dom.append({'domain': lab, 'sim': f'{lo:g}–{hi:g}' if hi < 9 else f'≥ {lo:g}', 'kind': kind, 'n_pairs': int(k.sum()), 'resolution': round(float(np.exp(np.median(err[k]))), 3) if k.sum() else None})
val = {'calibration': [r for r in cal if r['model'] == 'AVG'], 'calibration_all': cal, 'summary': summ, 'types': typ, 'typical_gamma_error': gam, 'domain': dom}
dump(OUT / 'validation.json', val)
print(json.dumps({'solvents': len(solv), 'solutes': len(solutes), 'pairs': len(plist), 'meas': sum(len(p['m']) for p in plist), 'gamma_err': gam, 'domain': dom}, indent=1))
for f in sorted(list(OUT.glob('*')) + list(MOD.rglob('*'))):
    if f.is_file(): print(f'{f.stat().st_size/1e6:7.2f} MB  {f.relative_to(SITE)}')
