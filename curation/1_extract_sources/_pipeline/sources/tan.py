"""
Tan et al. 2022 supporting-information PDF -> standardized 02_IDAC.xlsx

Run inside one source folder:

05_Tan/
├─ 00_Source/
│  ├─ Tan_2022_Supporting_Information.pdf
│  ├─ Tan_2022_Article.pdf
│  └─ Original_corresponding_pages/   # optional audit copies, pages 16-192
├─ tan_build_simple.py
├─ extract_idac_isomers_classified.py
└─ 02_IDAC.xlsx

Dependencies:
    python -m pip install pymupdf xlsxwriter

This script does NOT use OCR.
It reads the born-digital PDF text layer and performs hard validation before
writing a formal workbook.

Important PDF issue handled here:
- Table S2 is visually cramped: long solvent/solute names and Ref text are
  clipped/overlapped in the PDF layout.
- Therefore Table S2 printed names are NOT trusted.
- Names are recovered by Table S2 solute_id/solvent_id + PubChem CID and linked
  back to Table S1(a,b), which contain the clean identity lists.

Scientific value rule:
- Table S2 'Experimental value' is ln(gamma_inf), not gamma_inf (negative values
  occur; the source Square Error equals (Prediction - Experimental value)^2;
  the article states ln(gamma_inf) is used as the model quantity).
- Output '无限稀释活度系数' = exp(Experimental value).
- NCF Prediction/Square Error are used only for parser alignment audits; neither
  value is exported to IDAC or isomer workbooks.
"""
from __future__ import annotations
from functools import lru_cache
import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from collections import Counter
try:
    import pymupdf as fitz
except ImportError as exc:
    raise SystemExit('Missing PyMuPDF. Install: python -m pip install pymupdf xlsxwriter') from exc
ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = ROOT / '00_Source' / 'Tan_2022_Supporting_Information.pdf'
DEFAULT_OUTPUT = ROOT / '02_IDAC.xlsx'
EXPECTED_ROWS = 21048
EXPECTED_SOLUTES = 295
EXPECTED_SOLVENTS = 407
EXPECTED_PAIRS = 7668
EXPECTED_TMIN = 249.84
EXPECTED_TMAX = 555.6
HEADERS = ['IDAC记录ID', 'DOI', '文献标题', '期刊', '发表年份', '原始JSON文件', '数据块编号', '溶质', '溶质分子式', '溶质InChI', '溶质InChIKey', '溶剂组分数', '溶剂', '溶剂InChI', '溶剂InChIKey', '温度_K', '压力_kPa', '无限稀释活度系数', '活度系数不确定度', '实验方法', '相态', '标准态', '无限稀释判定方式', '稀释变量', '稀释组分', '稀释值', '其他固定条件', '实验目的', 'IDAC判定等级', '数据质量状态', '建议用于建模', '质量备注']
FLOAT_RE = re.compile('[-+]?(?:\\d+(?:\\.\\d*)?|\\.\\d+)(?:[Ee][-+]?\\d+)?')
S1_RE = re.compile('^\\s*(.*?)\\s+(\\d+)\\s+(\\S+)\\s+([0-9.]+)\\s+(\\d+)\\s*$')

def y_groups(page):
    spans = []
    d = page.get_text('dict')
    for b in d.get('blocks', []):
        for line in b.get('lines', []):
            spans.extend(line.get('spans', []))
    groups = {}
    for s in spans:
        y = round(float(s['bbox'][1]), 2)
        groups.setdefault(y, []).append(s)
    return [(y, sorted(v, key=lambda s: s['bbox'][0])) for y, v in sorted(groups.items())]
SOLUTE_NAME_PATCHES = {53: 'cis-1,2-dimethylcyclohexane', 216: 'cis-1,4-dimethylcyclohexane', 116: '1,1,2-Trichloro-1,2,2-trifluoroethane', 288: '(R)-1-methyl-4-(1-methylethenyl)cyclohexene', 112: '1,1,1-trichlorotrifluoroethane', 181: '1,1,1-trifluoro-2-bromo-2-chloroethane', 100: '1,1,2,2-tetrachloro-1,2-difluoroethane'}
SOLVENT_NAME_PATCHES = {319: '2-Methylenepentanedinitrile', 202: 'Triethylene glycol dibutyl ether', 203: 'Diethylene glycol dibutyl ether', 207: 'Tripropylene glycol dimethyl ether', 91: 'Cis-decahydronaphthalene', 323: '2-(Diethylamino)ethylamine', 204: 'Tetraethylene glycol dimethyl ether', 242: '1,1,1,3,3,3-Hexafluoro-2-propanol', 68: 'Butyl phthalyl butyl glycolate', 273: 'cis,cis,cis-9,12,15-octadecatrienoic acid', 274: '(Z,Z)-9,12-octadecadienoic acid', 129: '2-thiophene carboxaldehyde', 214: '1,1,2-Trichloro-1,2,2-trifluoroethane', 406: '(R)-1-methyl-4-(1-methylethenyl)cyclohexene', 28: 'Tetrahydrofurfuryl alcohol', 41: 'N,N-Dimethyltetradecanamide', 44: 'Dibutyl tetrachlorophthalate', 53: 'Dipropyl tetrachlorophthalate', 75: 'N-methylmethane sulfonamide', 94: 'hexamethylphosphoramide', 117: 'Butyl ethylhexyl phthalate', 131: 'Ethyl 2-hydroxyethyl sulfide', 217: 'N,N-diethyldodecanamide', 218: '1,1,1,2,2,3,3,4,4,5,5,6,6-Tridecafluorooctadecane', 219: '1,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8-Heptadecafluoroicosane', 220: '1,1,1,2,2,3,3,4,4-Nonafluorodocosane', 233: 'tetrahydroxyethylethylenediamine', 332: '1,3-dimethyl-2-imidazolidinone', 249: '1,5-Dimethylpyrrolidinone', 250: '19,24-dioctadecyldotetracontane', 252: 'tetra(cyanoethoxy)pentaerythritol', 285: '2-amino-4-methyl thiazole', 297: 'N,N-dimethylpropionamide', 335: 'octamethylcyclotetrasiloxane', 313: '1,1,2,2-tetrachloro-1,2-difluoroethane'}
KNOWN_FAMILIES = {'Nitriles', 'Ketones', 'Halogenated', 'Ethers', 'Aromatics', 'Alkanes', 'Alcohols', 'Others', 'Amines', 'Esters', 'Acids', 'Alkenes', 'Alkynes', 'Amides', 'Aldehydes', 'esters'}

def parse_s1(doc, kind):
    pages = range(2, 8) if kind == 'solute' else range(8, 16)
    expected = EXPECTED_SOLUTES if kind == 'solute' else EXPECTED_SOLVENTS
    patches = SOLUTE_NAME_PATCHES if kind == 'solute' else SOLVENT_NAME_PATCHES
    rows = {}
    for pno in pages:
        for y, spans in y_groups(doc[pno - 1]):
            texts = [x['text'].strip() for x in spans if x['text'].strip()]
            if not texts:
                continue
            whole = ' '.join(texts)
            if 'Table S1' in whole or ('Name' in whole and ('Pubchem' in whole or 'MolWeight' in whole)):
                continue
            id_candidates = []
            mw_candidates = []
            for sp in spans:
                x = float(sp['bbox'][0])
                t = sp['text'].strip()
                if re.fullmatch('\\d+', t):
                    if kind == 'solute' and x >= 470 or (kind == 'solvent' and x >= 440):
                        id_candidates.append((x, t))
                if re.fullmatch('[-+]?\\d+(?:\\.\\d+)?', t):
                    if kind == 'solute' and 390 <= x < 470 or (kind == 'solvent' and 330 <= x < 440):
                        mw_candidates.append((x, t))
            if not id_candidates or not mw_candidates:
                continue
            idx = int(id_candidates[-1][1])
            mw = float(mw_candidates[-1][1])
            cid = None
            family = None
            name_parts = []
            for sp in spans:
                x = float(sp['bbox'][0])
                t = sp['text'].strip()
                if not t:
                    continue
                if t == str(idx) and (kind == 'solute' and x >= 470 or (kind == 'solvent' and x >= 440)):
                    continue
                if re.fullmatch('[-+]?\\d+(?:\\.\\d+)?', t) and abs(float(t) - mw) < 1e-10 and (kind == 'solute' and 390 <= x < 470 or (kind == 'solvent' and 330 <= x < 440)):
                    continue
                m = re.fullmatch('(\\d+)\\s+([A-Za-z]+)', t)
                if m and m.group(2) in KNOWN_FAMILIES:
                    cid = int(m.group(1))
                    family = m.group(2)
                    continue
                if t in KNOWN_FAMILIES:
                    family = t
                    continue
                if re.fullmatch('\\d+', t):
                    if kind == 'solute' and 220 <= x < 390 or (kind == 'solvent' and 150 <= x < 330):
                        if cid is None:
                            cid = int(t)
                        continue
                if kind == 'solute' and x < 280 or (kind == 'solvent' and x < 225):
                    name_parts.append(t)
            name = ' '.join(name_parts).strip()
            if cid is None:
                m = re.search('(\\d+)\\s+(' + '|'.join(map(re.escape, KNOWN_FAMILIES)) + ')\\b', whole)
                if m:
                    cid = int(m.group(1))
                    family = family or m.group(2)
                    name = whole[:m.start()].strip()
            if cid is not None:
                name = re.sub(re.escape(str(cid)) + '\\s*$', '', name).strip()
            if idx in patches:
                name = patches[idx]
            if not name or cid is None or (not family):
                raise RuntimeError(f"Table S1 {kind} parse failure page={pno}, y={y}, id={idx}, spans={[(round(float(x['bbox'][0]), 1), x['text']) for x in spans]}")
            rows[idx] = {'name': name, 'cid': cid, 'family': family, 'mw': mw, 'id': idx}
    if len(rows) != expected or set(rows) != set(range(expected)):
        missing = sorted(set(range(expected)) - set(rows))
        raise RuntimeError(f'Table S1 {kind} parse failed: got {len(rows)}, expected {expected}; missing IDs={missing[:20]}')
    return rows

def parse_s2_group(y, spans, page_no):
    vals = {}
    for s in spans:
        x0 = float(s['bbox'][0])
        txt = s['text'].strip()
        if not txt:
            continue
        if 110 <= x0 < 152:
            vals.setdefault('exp', []).append(txt)
        elif 152 <= x0 < 179:
            vals.setdefault('temp', []).append(txt)
        elif 205 <= x0 < 231:
            vals.setdefault('solute_cid', []).append(txt)
        elif 255 <= x0 < 290:
            vals.setdefault('solvent_cid', []).append(txt)
        elif 290 <= x0 < 335:
            vals.setdefault('year_ref', []).append(txt)
        elif 345 <= x0 < 365:
            vals.setdefault('solvent_id', []).append(txt)
        elif 365 <= x0 < 418:
            vals.setdefault('solute_id_family', []).append(txt)
        elif 418 <= x0 < 458:
            vals.setdefault('solvent_family', []).append(txt)
        elif 455 <= x0 < 520:
            vals.setdefault('pred_err', []).append(txt)
    if 'exp' not in vals:
        return None
    try:
        exp = float(' '.join(vals['exp']))
        temp = float(' '.join(vals.get('temp', [])))
        scid = int(' '.join(vals.get('solute_cid', [])))
        vcid = int(' '.join(vals.get('solvent_cid', [])))
        yrref = ' '.join(vals.get('year_ref', []))
        m = re.match('(\\d{4})\\s*(.*)', yrref)
        assert m
        year = int(m.group(1))
        ref = m.group(2).strip()
        vid = int(' '.join(vals.get('solvent_id', [])))
        sif = ' '.join(vals.get('solute_id_family', []))
        m2 = re.match('(\\d+)\\s+(.*)', sif)
        assert m2
        sid = int(m2.group(1))
        sfam = m2.group(2).strip()
        vfam = ' '.join(vals.get('solvent_family', [])).strip()
        nums = FLOAT_RE.findall(' '.join(vals.get('pred_err', [])))
        assert len(nums) == 2
        pred = float(nums[0])
        sq = float(nums[1])
        return {'page': page_no, 'ln_gamma': exp, 'temp': temp, 'scid': scid, 'vcid': vcid, 'year': year, 'ref': ref, 'vid': vid, 'sid': sid, 'sfam': sfam, 'vfam': vfam, 'pred': pred, 'sq': sq}
    except Exception as exc:
        raise RuntimeError(f'Table S2 parse failure on PDF page {page_no}, y={y}: {vals}') from exc

def parse_s2(doc):
    rows = []
    for pno in range(16, 193):
        for y, spans in y_groups(doc[pno - 1]):
            r = parse_s2_group(y, spans, pno)
            if r is not None:
                rows.append(r)
    if len(rows) != EXPECTED_ROWS:
        raise RuntimeError(f'Table S2 row count {len(rows)} != expected {EXPECTED_ROWS}')
    return rows

def validate(rows, solutes, solvents):
    for i, r in enumerate(rows, 1):
        so = solutes[r['sid']]
        sv = solvents[r['vid']]
        if so['cid'] != r['scid'] or sv['cid'] != r['vcid']:
            raise RuntimeError(f'CID mismatch at S2 record {i}')
        if so['family'].casefold() != r['sfam'].casefold() or sv['family'].casefold() != r['vfam'].casefold():
            raise RuntimeError(f'family mismatch at S2 record {i}')
        if abs((r['pred'] - r['ln_gamma']) ** 2 - r['sq']) > 5e-06:
            raise RuntimeError(f'square-error mismatch at S2 record {i}')
    pairs = len({(r['sid'], r['vid']) for r in rows})
    if pairs != EXPECTED_PAIRS:
        raise RuntimeError(f'pair count {pairs} != {EXPECTED_PAIRS}')
    if min((r['temp'] for r in rows)) != EXPECTED_TMIN or max((r['temp'] for r in rows)) != EXPECTED_TMAX:
        raise RuntimeError('temperature range mismatch')
    if len({(r['sid'], r['vid'], r['temp'], r['ln_gamma']) for r in rows}) != len(rows):
        raise RuntimeError('exact duplicate records found')

@lru_cache(maxsize=None)
def cid_properties(cid, mw):
    path = ROOT / 'tan_cid_structures.json'
    data = json.loads(path.read_text(encoding='utf-8'))
    p = data['properties'].get(str(cid))
    if p is None or abs(float(p['MolecularWeight']) - mw) > 0.15:
        raise RuntimeError(f'Missing/invalid source-CID structure: CID {cid}')
    return p

def output_row(i, r, solutes, solvents):
    so = solutes[r['sid']]
    sv = solvents[r['vid']]
    sp = cid_properties(so['cid'], so['mw'])
    vp = cid_properties(sv['cid'], sv['mw'])
    other = f"lnγ={r['ln_gamma']:.10g};scid={r['scid']};vcid={r['vcid']};sid={r['sid']};vid={r['vid']};ref={r['ref']}"
    return [f'TAN_IDAC_{i:06d}', '', '', '', r['year'], f"00_Source/Tan_2022_Supporting_Information.pdf#page={r['page']}", f"TableS2_p{r['page']:04d}_row{i:06d}", so['name'], sp['MolecularFormula'], sp['InChI'], sp['InChIKey'], 1, sv['name'], vp['InChI'], vp['InChIKey'], r['temp'], '', math.exp(r['ln_gamma']), '', '', '', '', 'SOURCE_IDAC_LN_TO_GAMMA', '', '', '', other, 'Tan2022_NCF', 'A_SOURCE_IDAC', 'OK', '是', 'S2_NAMES_FROM_S1_ID_CID;STRUCTURE_FROM_EXACT_SOURCE_PUBCHEM_CID;REF_CLIPPED']
