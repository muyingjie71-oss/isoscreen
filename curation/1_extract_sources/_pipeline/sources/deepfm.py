"""
DeepFM supplementary IDAC workbook -> standardized 02_IDAC.xlsx

Folder contract (run inside 04_DeepFM):

04_DeepFM/
├─ 00_Source/
│  └─ DeepFM_IDAC_Source.xlsx
├─ deepfm_build_simple.py
├─ extract_idac_isomers_classified.py
├─ 02_IDAC.xlsx          # created by this script
└─ 03_ISOMERS.xlsx       # created later by the classifier

Source interpretation
---------------------
The original supplementary workbook contains:
- Table S1: experimental IDAC dataset + DeepFM predictions
- Table S2: literature citation metadata keyed by Ref_id
- Table S3: solvent (IL/DES) information keyed by IL/DES_id
- Table S4: solute information keyed by Solute_id
- Tables S5-S7: model-input/model-performance information

Only the EXPERIMENTAL IDAC values in Table S1 are used scientifically here:
- H = IDAC
- I = Ln_IDAC (used only as an internal consistency check)
- J/K prediction columns are deliberately NOT imported as experimental data.

The script joins S1 to S2/S3/S4, derives InChI/InChIKey/formula from the
SOURCE-PROVIDED SMILES using RDKit, and writes the same 32-column standardized
IDAC schema used by the other sources.

No external web lookup is performed.
No experimental rows are deduplicated.

Dependencies:
    python -m pip install XlsxWriter rdkit

Run:
    python .\\deepfm_build_simple.py

Validation only:
    python .\\deepfm_build_simple.py --validate-only
"""
from __future__ import annotations
import argparse
import math
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path
import xml.etree.ElementTree as ET
from typing import Any, Dict, Iterable, Iterator, Tuple
try:
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog('rdApp.warning')
    from rdkit.Chem import rdMolDescriptors
    from rdkit.Chem import inchi as rd_inchi
except ImportError as exc:
    raise SystemExit('Missing RDKit. Install with: python -m pip install rdkit') from exc
SCRIPT_VERSION = '1.0.0'
ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = ROOT / '00_Source' / 'DeepFM_IDAC_Source.xlsx'
DEFAULT_OUTPUT = ROOT / '02_IDAC.xlsx'
NS_MAIN = 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'
NS_REL_DOC = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
NS_REL_PKG = 'http://schemas.openxmlformats.org/package/2006/relationships'
NS = {'m': NS_MAIN}
IDAC_HEADERS = ['IDAC记录ID', 'DOI', '文献标题', '期刊', '发表年份', '原始JSON文件', '数据块编号', '溶质', '溶质分子式', '溶质InChI', '溶质InChIKey', '溶剂组分数', '溶剂', '溶剂InChI', '溶剂InChIKey', '温度_K', '压力_kPa', '无限稀释活度系数', '活度系数不确定度', '实验方法', '相态', '标准态', '无限稀释判定方式', '稀释变量', '稀释组分', '稀释值', '其他固定条件', '实验目的', 'IDAC判定等级', '数据质量状态', '建议用于建模', '质量备注']
FIELD_HELP = {'IDAC记录ID': '稳定行级主键：DEEPFM::S1ROW<Excel row>。', 'DOI': 'DeepFM 补充表 Table S2 未给 DOI，因此留空；不外部猜测。', '文献标题': '由 Table S1.Ref_id 连接 Table S2.name。', '期刊': '由 Table S2.Journal 读取。', '发表年份': '由 Table S2.year 读取。', '原始JSON文件': '统一字段名沿用其他来源；本来源实际为 Excel，因此记录来源工作簿、sheet 和原始行。', '数据块编号': 'DeepFM Table S1 的 Ref_id，用于回溯对应文献。', '溶质': '由 Table S1.Solute_id 连接 Table S4.Solute；并与 Table S1.Solute_Name 核对。', '溶质分子式': '由 Table S4 源 SMILES 通过 RDKit 计算；不是外部数据库补值。', '溶质InChI': '由 Table S4 源 SMILES 通过 RDKit 生成。', '溶质InChIKey': '由 Table S4 源 SMILES 通过 RDKit 生成。', '溶剂组分数': 'IL 作为一个溶剂实体记 1；DES 由两类组成物构成，记 2。', '溶剂': '由 Table S3.Full name 读取。', '溶剂InChI': '由 Table S3 源 SMILES（含离子/混合物片段）通过 RDKit 生成的整体 InChI。', '溶剂InChIKey': '由 Table S3 源 SMILES 通过 RDKit 生成的整体 InChIKey。', '温度_K': 'Table S1.T/K。', '压力_kPa': 'DeepFM Table S1 未提供压力，留空。', '无限稀释活度系数': 'Table S1.IDAC（实验值）。', '活度系数不确定度': 'DeepFM Table S1 未提供逐点实验不确定度，留空。', '实验方法': '补充表没有统一逐行实验方法字段，留空；不从文献标题推断。', '相态': '源表未给逐行相态字段，留空。', '标准态': '源表未给，留空。', '无限稀释判定方式': '来源本身明确将 Table S1 列 H 标为 IDAC；列 I=Ln_IDAC 用于核对 ln(IDAC)。', '稀释变量': '源表已直接给 IDAC，但未给逐点稀释组成变量，留空。', '稀释组分': '同上，留空。', '稀释值': '同上，留空。', '其他固定条件': '保存 DeepFM Ref_id / IL_id / Solute_id、IL/DES 类型、缩写和源 SMILES。', '实验目的': '源表未给逐点实验目的，留空。', 'IDAC判定等级': 'A_来源明确IDAC。', '数据质量状态': '通过主键连接、IDAC>0、Ln_IDAC=ln(IDAC) 审计的记录为 OK。', '建议用于建模': '通过上述审计则为 是。', '质量备注': '保存 DeepFM 的 Sample Partitioning；预测列 J/K 明确不进入实验 IDAC 总表。'}

def excel_col_to_int(ref: str) -> int:
    m = re.match('([A-Z]+)', ref)
    if not m:
        raise ValueError(ref)
    n = 0
    for ch in m.group(1):
        n = n * 26 + (ord(ch) - 64)
    return n

def clean_text(x: Any) -> str:
    if x is None:
        return ''
    return re.sub('\\s+', ' ', str(x)).strip()

def as_int(x: Any) -> int:
    return int(float(str(x).strip()))

def as_float(x: Any) -> float:
    v = float(str(x).strip())
    if not math.isfinite(v):
        raise ValueError(f'non-finite numeric value: {x!r}')
    return v

def shared_strings(z: zipfile.ZipFile) -> list[str]:
    if 'xl/sharedStrings.xml' not in z.namelist():
        return []
    root = ET.fromstring(z.read('xl/sharedStrings.xml'))
    out = []
    for si in root.findall('m:si', NS):
        out.append(''.join((t.text or '' for t in si.iter(f'{{{NS_MAIN}}}t'))))
    return out

def sheet_paths(z: zipfile.ZipFile) -> dict[str, str]:
    wb = ET.fromstring(z.read('xl/workbook.xml'))
    rels = ET.fromstring(z.read('xl/_rels/workbook.xml.rels'))
    relmap = {r.attrib['Id']: r.attrib['Target'] for r in rels}
    out = {}
    sheets = wb.find('m:sheets', NS)
    if sheets is None:
        return out
    for s in sheets:
        rid = s.attrib[f'{{{NS_REL_DOC}}}id']
        target = relmap[rid]
        if target.startswith('/'):
            target = target.lstrip('/')
        elif not target.startswith('xl/'):
            target = 'xl/' + target
        out[s.attrib['name']] = target
    return out

def iter_sheet_rows(xlsx: Path, sheet_name: str) -> Iterator[tuple[int, dict[int, Any]]]:
    with zipfile.ZipFile(xlsx) as z:
        ss = shared_strings(z)
        paths = sheet_paths(z)
        if sheet_name not in paths:
            raise RuntimeError(f'Missing worksheet: {sheet_name}')
        with z.open(paths[sheet_name]) as fh:
            for event, elem in ET.iterparse(fh, events=('end',)):
                if elem.tag != f'{{{NS_MAIN}}}row':
                    continue
                rnum = int(elem.attrib.get('r', '0'))
                cells: dict[int, Any] = {}
                for c in elem.findall('m:c', NS):
                    ref = c.attrib.get('r', '')
                    ci = excel_col_to_int(ref)
                    typ = c.attrib.get('t', '')
                    v = c.find('m:v', NS)
                    val: Any = ''
                    if typ == 's' and v is not None and (v.text is not None):
                        val = ss[int(v.text)]
                    elif typ == 'inlineStr':
                        t = c.find('.//m:t', NS)
                        val = t.text if t is not None and t.text is not None else ''
                    elif typ == 'b' and v is not None:
                        val = bool(int(v.text or '0'))
                    elif v is not None and v.text is not None:
                        val = v.text
                    cells[ci] = val
                yield (rnum, cells)
                elem.clear()

def load_table(xlsx: Path, sheet: str, header_row: int) -> list[dict[str, Any]]:
    headers: list[str] | None = None
    rows = []
    for rnum, cells in iter_sheet_rows(xlsx, sheet):
        if rnum < header_row:
            continue
        if rnum == header_row:
            max_col = max(cells) if cells else 0
            headers = [clean_text(cells.get(i, '')) for i in range(1, max_col + 1)]
            continue
        if headers is None:
            raise RuntimeError(f'Header row not found in {sheet}')
        row = {headers[i - 1]: cells.get(i, '') for i in range(1, len(headers) + 1) if headers[i - 1]}
        if any((v not in (None, '') for v in row.values())):
            row['__excel_row__'] = rnum
            rows.append(row)
    return rows

def iter_table_s1(xlsx: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    headers: list[str] | None = None
    for rnum, cells in iter_sheet_rows(xlsx, 'Table S1'):
        if rnum < 2:
            continue
        if rnum == 2:
            max_col = max(cells) if cells else 0
            headers = [clean_text(cells.get(i, '')) for i in range(1, max_col + 1)]
            required = ['Ref_id', 'IL_id', 'Solute_id', 'T/K', 'IL_Structure', 'Solute_Name', 'Sample Partitioning', 'IDAC', 'Ln_IDAC', 'Predicted_Ln_IDAC', 'Predicted_IDAC']
            missing = [h for h in required if h not in headers]
            if missing:
                raise RuntimeError(f'Table S1 missing expected columns: {missing}')
            continue
        if headers is None:
            raise RuntimeError('Table S1 header row not found')
        row = {headers[i - 1]: cells.get(i, '') for i in range(1, len(headers) + 1) if headers[i - 1]}
        if any((v not in (None, '') for v in row.values())):
            yield (rnum, row)

def structure_from_source_smiles(smiles: str, label: str) -> dict[str, str]:
    smi = clean_text(smiles)
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        raise RuntimeError(f'RDKit cannot parse source SMILES for {label}: {smi}')
    try:
        formula = rdMolDescriptors.CalcMolFormula(mol)
        inc = rd_inchi.MolToInchi(mol)
        ik = rd_inchi.MolToInchiKey(mol)
    except Exception as exc:
        raise RuntimeError(f'RDKit structure conversion failed for {label}: {exc}') from exc
    if not formula or not inc or (not ik):
        raise RuntimeError(f'Incomplete RDKit structure identifiers for {label}: {smi}')
    return {'formula': formula, 'inchi': inc, 'inchikey': ik, 'smiles': smi}
