"""
Brouwer IDAC source workbook -> one normalized long-format IDAC workbook.

INPUT
-----
Brouwer_IDAC_Source.xlsx
    Raw Data       : original wide experimental table
    Data Treatment : derived/fitted table from the paper (NOT used to create raw IDAC rows)
    Corrections    : verified corrections / audit records

OUTPUT
------
05_Brouwer_IDAC.xlsx
    字段说明
    IDAC_001       : one row = one experimental gamma^infinity point; same 32-column schema as ThermoML IDAC

Design principles
-----------------
1) The Brouwer Excel is the ONLY scientific source used in this step.
2) Raw Data is unfolded from wide format into long format.
3) Corrections are applied without silently inventing information; correction details are folded into 质量备注.
4) No PubChem/Crossref/web enrichment is performed here.
5) Structure identifiers not present in Brouwer (formula/InChI/InChIKey) are left blank.
   They are deliberately deferred to the later isomer-classification step.
6) Data Treatment is NOT treated as additional experimental data, avoiding duplication.
7) Decimal commas in source numeric cells are normalized (e.g. 14,6 -> 14.6); source cell locations remain traceable via IDAC记录ID / 数据块编号.
8) The number of output experimental rows MUST equal the total declared in Raw Data!E3.
   Otherwise the formal output is not finalized.

Default project layout
----------------------
Surrogate_Modelling/
    00_Source_Data/
        03_Brouwer/
            Brouwer_IDAC_Source.xlsx
    brouwer_build_simple.py
    05_Brouwer_IDAC.xlsx

Run
---
    python brouwer_build_simple.py

Explicit paths
--------------
    python brouwer_build_simple.py         --source ./00_Source_Data/03_Brouwer/Brouwer_IDAC_Source.xlsx         --output ./05_Brouwer_IDAC.xlsx

Audit extraction only (no Excel written)
----------------------------------------
    python brouwer_build_simple.py --validate-only

Dependency
----------
    python -m pip install XlsxWriter

The source XLSX is read directly from its Open XML parts using only Python's standard
library. This avoids modifying the source workbook and keeps the build dependency small.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import math
import re
import sys
import time
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple
import xml.etree.ElementTree as ET
SOURCE_DB = 'Brouwer2021'
SOURCE_SHEET = 'Raw Data'
CORRECTIONS_SHEET = 'Corrections'
DATA_TREATMENT_SHEET = 'Data Treatment'
DEFAULT_OUTPUT = str(Path(__file__).resolve().parent / '02_IDAC.xlsx')
NS_MAIN = 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'
NS_REL = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
NS_PKG_REL = 'http://schemas.openxmlformats.org/package/2006/relationships'
NS = {'m': NS_MAIN, 'r': NS_REL, 'p': NS_PKG_REL}
SOLVENT_TYPE = {'1': 'Molecular Solvent', '2': 'Ionic Liquid', '3': 'Deep Eutectic Solvent'}
IDAC_HEADERS = ['IDAC记录ID', 'DOI', '文献标题', '期刊', '发表年份', '原始JSON文件', '数据块编号', '溶质', '溶质分子式', '溶质InChI', '溶质InChIKey', '溶剂组分数', '溶剂', '溶剂InChI', '溶剂InChIKey', '温度_K', '压力_kPa', '无限稀释活度系数', '活度系数不确定度', '实验方法', '相态', '标准态', '无限稀释判定方式', '稀释变量', '稀释组分', '稀释值', '其他固定条件', '实验目的', 'IDAC判定等级', '数据质量状态', '建议用于建模', '质量备注']
FIELD_HELP = {'IDAC记录ID': 'Brouwer 数据点稳定主键；包含 Raw Data 中原始 gamma∞ 单元格坐标。', 'DOI': '仅在 Brouwer 来源工作簿或 Corrections 中明确给出时填写；不联网猜测。', '文献标题': 'Brouwer Raw Data 未单独结构化提供，当前阶段留空。', '期刊': 'Brouwer Raw Data 未单独结构化提供，当前阶段留空。', '发表年份': 'Raw Data B 列。', '原始JSON文件': '为与 ThermoML IDAC 保持完全相同的字段名，本列记录 Brouwer 来源工作簿及 Raw Data sheet。', '数据块编号': '记录 Brouwer Raw Data 的原始数据点位置，例如 RawData!RV4690，可直接回溯来源。', '溶质': 'Raw Data 第一行溶质列头；若 Corrections 指定逐点身份更正，则使用更正后的名称。', '溶质分子式': 'Brouwer 来源未提供，当前阶段留空。', '溶质InChI': 'Brouwer 来源未提供，当前阶段留空。', '溶质InChIKey': 'Brouwer 来源未提供，当前阶段留空。', '溶剂组分数': 'Molecular Solvent / Ionic Liquid 记为 1；Deep Eutectic Solvent 的实际组分数来源未明确，留空。', '溶剂': 'Raw Data E 列；已核实的溶剂身份更正按 Corrections 应用。', '溶剂InChI': 'Brouwer 来源未提供，当前阶段留空。', '溶剂InChIKey': 'Brouwer 来源未提供，当前阶段留空。', '温度_K': 'Raw Data 对应溶质的 Temp.(K)；小数逗号仅做数值格式标准化。', '压力_kPa': 'Brouwer Raw Data 未提供，留空。', '无限稀释活度系数': 'Raw Data 对应溶质的 y_inf / gamma∞ 实验值。', '活度系数不确定度': 'Brouwer Raw Data 未逐点提供，留空。', '实验方法': 'Brouwer Raw Data 未结构化提供，留空。', '相态': 'Brouwer Raw Data 未结构化提供，留空。', '标准态': 'Brouwer Raw Data 未结构化提供，留空。', '无限稀释判定方式': 'Brouwer 2021 Supplementary Raw Data 本身是 IDAC 专用汇编，实验列明确标记为 y_inf。', '稀释变量': '来源未逐点提供显式组成变量，留空。', '稀释组分': '来源未逐点提供显式组成变量，留空。', '稀释值': '来源未逐点提供显式 x→0 数值，留空。', '其他固定条件': '保存 Brouwer 原表中仍有价值但 ThermoML 标准字段没有单独位置的溶剂类型、CAS、分子量和溶质类别。', '实验目的': 'Brouwer Raw Data 未结构化提供，留空。', 'IDAC判定等级': 'SOURCE_原始汇编明确γ∞：Brouwer IDAC 专用 Raw Data 的 y_inf 实验点。', '数据质量状态': 'OK / OK_CORRECTED / REVIEW_CORRECTION_SCOPE / SOURCE_ANOMALY。', '建议用于建模': '是/否；来源异常或更正范围未逐点核实的数据默认不进入建模。', '质量备注': '记录小数逗号标准化、Corrections 更正信息和异常说明；普通未更正数据保持简洁。'}

def norm_text(x: Any) -> str:
    if x is None:
        return ''
    return str(x).strip()

def unique_preserve(items: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for x in items:
        s = norm_text(x)
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out

def join_unique(items: Iterable[str], sep: str=' | ') -> str:
    return sep.join(unique_preserve(items))

def source_number(x: Any) -> Tuple[Optional[float], bool]:
    """Return (numeric value, decimal-comma-was-normalized)."""
    s = norm_text(x)
    if not s:
        return (None, False)
    normalized_comma = False
    candidate = s
    if ',' in candidate and '.' not in candidate:
        candidate = candidate.replace(',', '.')
        normalized_comma = True
    try:
        v = float(candidate)
    except (TypeError, ValueError):
        return (None, normalized_comma)
    if not math.isfinite(v):
        return (None, normalized_comma)
    return (v, normalized_comma)

def intish(x: Any) -> Any:
    v, _ = source_number(x)
    if v is None:
        return norm_text(x)
    if abs(v - round(v)) < 1e-09:
        return int(round(v))
    return v

def stable_ref_id(reference: str, anchor_row: int) -> str:
    key = f'{norm_text(reference)}\n{anchor_row}'
    return 'BRWREF_' + hashlib.sha1(key.encode('utf-8')).hexdigest()[:16].upper()

def extract_doi(text: str) -> str:
    s = norm_text(text)
    if not s:
        return ''
    m = re.search('\\b10\\.\\d{4,9}/[-._;()/:A-Z0-9]+', s, flags=re.I)
    if not m:
        return ''
    return m.group(0).rstrip('.,;)')

def safe_output_value(x: Any) -> Any:
    if x is None:
        return ''
    if isinstance(x, (int, float, bool)):
        return x
    return str(x)

def col_to_num(col: str) -> int:
    n = 0
    for ch in col.upper():
        if 'A' <= ch <= 'Z':
            n = n * 26 + (ord(ch) - 64)
    return n

def num_to_col(n: int) -> str:
    s = ''
    while n > 0:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s

def split_cell_ref(ref: str) -> Tuple[int, int]:
    m = re.fullmatch('([A-Z]+)(\\d+)', ref.upper().strip())
    if not m:
        raise ValueError(f'Invalid cell reference: {ref!r}')
    return (col_to_num(m.group(1)), int(m.group(2)))

def parse_location_rows(location: str) -> List[int]:
    s = norm_text(location).upper()
    if not s:
        return []
    if ':' not in s:
        try:
            _, row = split_cell_ref(s)
            return [row]
        except Exception:
            return []
    a, b = s.split(':', 1)
    try:
        _, r1 = split_cell_ref(a)
        _, r2 = split_cell_ref(b)
    except Exception:
        return []
    lo, hi = sorted((r1, r2))
    return list(range(lo, hi + 1))

class XLSXReader:

    def __init__(self, path: Path):
        self.path = path
        try:
            self.zf = zipfile.ZipFile(path, 'r')
        except Exception as exc:
            raise RuntimeError(f'Source is not a readable XLSX/ZIP package: {path}: {exc}') from exc
        bad_member = self.zf.testzip()
        if bad_member:
            self.zf.close()
            raise RuntimeError(f'Corrupt XLSX member: {bad_member}')
        required_parts = {'xl/workbook.xml', 'xl/_rels/workbook.xml.rels', '[Content_Types].xml'}
        missing_parts = sorted(required_parts - set(self.zf.namelist()))
        if missing_parts:
            self.zf.close()
            raise RuntimeError(f'Source XLSX is missing required package parts: {missing_parts}')
        self.shared_strings = self._read_shared_strings()
        self.sheet_targets = self._read_sheet_targets()

    def close(self) -> None:
        self.zf.close()

    def __enter__(self) -> 'XLSXReader':
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _read_shared_strings(self) -> List[str]:
        if 'xl/sharedStrings.xml' not in self.zf.namelist():
            return []
        root = ET.fromstring(self.zf.read('xl/sharedStrings.xml'))
        out: List[str] = []
        for si in root.findall('m:si', NS):
            parts = []
            for node in si.iter(f'{{{NS_MAIN}}}t'):
                parts.append(node.text or '')
            out.append(''.join(parts))
        return out

    def _read_sheet_targets(self) -> Dict[str, str]:
        wb = ET.fromstring(self.zf.read('xl/workbook.xml'))
        rels = ET.fromstring(self.zf.read('xl/_rels/workbook.xml.rels'))
        relmap = {rel.attrib['Id']: rel.attrib['Target'] for rel in rels if 'Id' in rel.attrib and 'Target' in rel.attrib}
        out: Dict[str, str] = {}
        sheets = wb.find('m:sheets', NS)
        if sheets is None:
            return out
        for sh in sheets:
            name = sh.attrib.get('name', '')
            rid = sh.attrib.get(f'{{{NS_REL}}}id', '')
            target = relmap.get(rid, '')
            if target.startswith('/'):
                target = target.lstrip('/')
            elif not target.startswith('xl/'):
                target = 'xl/' + target
            if name and target:
                out[name] = target
        return out

    def cell_value(self, cell: ET.Element) -> str:
        ctype = cell.attrib.get('t', '')
        if ctype == 'inlineStr':
            return ''.join((t.text or '' for t in cell.iter(f'{{{NS_MAIN}}}t')))
        v = cell.find('m:v', NS)
        if v is None or v.text is None:
            return ''
        raw = v.text
        if ctype == 's':
            try:
                return self.shared_strings[int(raw)]
            except Exception:
                return raw
        if ctype == 'b':
            return 'TRUE' if raw == '1' else 'FALSE'
        return raw

    def iter_rows(self, sheet_name: str) -> Iterator[Tuple[int, Dict[int, str]]]:
        target = self.sheet_targets.get(sheet_name)
        if not target:
            raise RuntimeError(f'Worksheet not found: {sheet_name}')
        with self.zf.open(target) as fh:
            for event, elem in ET.iterparse(fh, events=('end',)):
                if elem.tag != f'{{{NS_MAIN}}}row':
                    continue
                rownum = int(elem.attrib.get('r', '0') or 0)
                row: Dict[int, str] = {}
                for cell in elem.findall('m:c', NS):
                    ref = cell.attrib.get('r', '')
                    m = re.match('([A-Z]+)', ref)
                    if not m:
                        continue
                    row[col_to_num(m.group(1))] = self.cell_value(cell)
                yield (rownum, row)
                elem.clear()

    def read_rows(self, sheet_name: str, wanted_rows: Sequence[int]) -> Dict[int, Dict[int, str]]:
        wanted = set((int(x) for x in wanted_rows))
        out: Dict[int, Dict[int, str]] = {}
        for rownum, row in self.iter_rows(sheet_name):
            if rownum in wanted:
                out[rownum] = row
            if wanted and rownum > max(wanted):
                break
        return out

@dataclass
class CorrectionRecord:
    record_id: str
    location: str
    category: str
    old: str
    new: str
    basis: str

@dataclass
class CorrectionGroup:
    key: str
    category: str
    location: str
    old: str
    new: str
    record_ids: List[str] = field(default_factory=list)
    bases: List[str] = field(default_factory=list)
    verified_solutes: List[str] = field(default_factory=list)
    matched_points: int = 0

    @property
    def source_record_count(self) -> int:
        return len(self.record_ids)

    @property
    def doi(self) -> str:
        for text in self.bases:
            d = extract_doi(text)
            if d:
                return d
        return ''

def parse_corrections(reader: XLSXReader) -> Tuple[List[CorrectionRecord], Dict[str, CorrectionGroup]]:
    rows = list(reader.iter_rows(CORRECTIONS_SHEET))
    if not rows:
        raise RuntimeError('Corrections sheet is empty')
    header = rows[0][1]
    header_by_col = {c: norm_text(v) for c, v in header.items()}
    col_by_header = {v: c for c, v in header_by_col.items() if v}
    required = ['记录ID', '原表位置', '更正类别', '原信息', '已核实信息', '依据或处理说明']
    missing = [h for h in required if h not in col_by_header]
    if missing:
        raise RuntimeError(f'Corrections missing columns: {missing}')
    records: List[CorrectionRecord] = []
    groups: Dict[str, CorrectionGroup] = {}
    for rownum, row in rows[1:]:
        rec = CorrectionRecord(record_id=norm_text(row.get(col_by_header['记录ID'])), location=norm_text(row.get(col_by_header['原表位置'])), category=norm_text(row.get(col_by_header['更正类别'])), old=norm_text(row.get(col_by_header['原信息'])), new=norm_text(row.get(col_by_header['已核实信息'])), basis=norm_text(row.get(col_by_header['依据或处理说明'])))
        if not rec.category and (not rec.location):
            continue
        records.append(rec)
        key_src = '\n'.join([rec.category, rec.location, rec.old, rec.new])
        key = hashlib.sha1(key_src.encode('utf-8')).hexdigest()[:16]
        grp = groups.get(key)
        if grp is None:
            grp = CorrectionGroup(key=key, category=rec.category, location=rec.location, old=rec.old, new=rec.new)
            groups[key] = grp
        if rec.record_id:
            grp.record_ids.append(rec.record_id)
        if rec.basis:
            grp.bases.append(rec.basis)
        m = re.search('(?:^|;)\\s*solute\\s+([^;]+)', rec.basis, flags=re.I)
        if m:
            solute = m.group(1).strip()
            if solute and solute not in grp.verified_solutes:
                grp.verified_solutes.append(solute)
    for grp in groups.values():
        grp.record_ids = unique_preserve(grp.record_ids)
        grp.bases = unique_preserve(grp.bases)
        grp.verified_solutes = unique_preserve(grp.verified_solutes)
    return (records, groups)

def correction_indexes(groups: Dict[str, CorrectionGroup]) -> Dict[str, Any]:
    temp_by_cell: Dict[str, CorrectionGroup] = {}
    solute_by_gamma_cell: Dict[str, CorrectionGroup] = {}
    ref_by_anchor_row: Dict[int, CorrectionGroup] = {}
    solvent_by_row: Dict[int, CorrectionGroup] = {}
    for grp in groups.values():
        loc = grp.location.upper().strip()
        cat = grp.category
        if '温度更正' in cat and ':' not in loc:
            temp_by_cell[loc] = grp
        elif '逐点身份更正' in cat and ':' not in loc:
            solute_by_gamma_cell[loc] = grp
        elif '引用补全' in cat and ':' not in loc:
            try:
                col, row = split_cell_ref(loc)
                if col == 1:
                    ref_by_anchor_row[row] = grp
            except Exception:
                pass
        elif '溶剂身份更正' in cat:
            for row in parse_location_rows(loc):
                solvent_by_row[row] = grp
    return {'temp_by_cell': temp_by_cell, 'solute_by_gamma_cell': solute_by_gamma_cell, 'ref_by_anchor_row': ref_by_anchor_row, 'solvent_by_row': solvent_by_row}

def _correction_value_matches(source_value: str, old_value: str, new_value: str) -> bool:
    """Accept either the pre-correction or already-corrected source state."""
    src = norm_text(source_value)
    candidates = [Path(__file__).resolve().parent / '00_Source' / 'Brouwer_IDAC_Source.xlsx', norm_text(old_value), norm_text(new_value)]
    if src in candidates:
        return True
    src_num, _ = source_number(src)
    if src_num is not None:
        for candidate in candidates:
            cand_num, _ = source_number(candidate)
            if cand_num is not None and abs(src_num - cand_num) <= 1e-12:
                return True
    return False

def validate_corrections_against_source(reader: XLSXReader, groups: Dict[str, CorrectionGroup], raw_header_row1: Dict[int, str]) -> int:
    """
    Verify that the frozen Raw Data and Corrections sheet belong together.

    The source workbook is a corrected copy, so a corrected field may already equal
    the verified new value. Pointwise solute corrections are deliberately different:
    their gamma cell is the locator and the shared solute column header should still
    carry the old identity.
    """
    wanted_rows = set()
    for grp in groups.values():
        loc = norm_text(grp.location).upper()
        if not loc:
            continue
        if ':' in loc:
            parts = loc.split(':', 1)
            for part in parts:
                try:
                    _, rr = split_cell_ref(part)
                    wanted_rows.add(rr)
                except Exception:
                    pass
        else:
            try:
                _, rr = split_cell_ref(loc)
                wanted_rows.add(rr)
            except Exception:
                pass
    rows = reader.read_rows(SOURCE_SHEET, sorted(wanted_rows))
    checked = 0
    problems: List[str] = []
    for grp in groups.values():
        loc = norm_text(grp.location).upper()
        cat = norm_text(grp.category)
        if not loc:
            continue
        if '温度更正' in cat and ':' not in loc:
            try:
                col, rr = split_cell_ref(loc)
                source_value = norm_text(rows.get(rr, {}).get(col))
                if not _correction_value_matches(source_value, grp.old, grp.new):
                    problems.append(f'{cat} {loc}: Raw Data value {source_value!r} matches neither old={grp.old!r} nor new={grp.new!r}')
                checked += 1
            except Exception as exc:
                problems.append(f'{cat} {loc}: cannot inspect source cell ({exc})')
        elif '引用补全' in cat and ':' not in loc:
            try:
                col, rr = split_cell_ref(loc)
                source_value = norm_text(rows.get(rr, {}).get(col))
                if not _correction_value_matches(source_value, grp.old, grp.new):
                    problems.append(f'{cat} {loc}: current reference does not match the correction record')
                checked += 1
            except Exception as exc:
                problems.append(f'{cat} {loc}: cannot inspect reference cell ({exc})')
        elif '溶剂身份更正' in cat:
            row_numbers = parse_location_rows(loc)
            for rr in row_numbers:
                source_value = norm_text(rows.get(rr, {}).get(5))
                if not _correction_value_matches(source_value, grp.old, grp.new):
                    problems.append(f'{cat} {loc}: Raw Data!E{rr}={source_value!r} matches neither old={grp.old!r} nor new={grp.new!r}')
                checked += 1
        elif '逐点身份更正' in cat and ':' not in loc:
            try:
                gamma_col, rr = split_cell_ref(loc)
                gamma_value = norm_text(rows.get(rr, {}).get(gamma_col))
                header_value = norm_text(raw_header_row1.get(gamma_col - 1))
                if not gamma_value:
                    problems.append(f'{cat} {loc}: locator gamma cell is empty')
                if header_value not in {norm_text(grp.old), norm_text(grp.new)}:
                    problems.append(f'{cat} {loc}: shared solute header {header_value!r} matches neither old={grp.old!r} nor new={grp.new!r}')
                checked += 1
            except Exception as exc:
                problems.append(f'{cat} {loc}: cannot inspect point locator/header ({exc})')
    if problems:
        preview = '\n'.join((f'- {x}' for x in problems[:20]))
        more = '' if len(problems) <= 20 else f'\n... and {len(problems) - 20} more'
        raise RuntimeError('Corrections sheet does not match the current Raw Data source:\n' + preview + more)
    return checked

def correction_handling_text(grp: CorrectionGroup) -> str:
    cat = grp.category
    if '温度更正' in cat:
        return '按指定温度单元格应用已核实值；原始单元格文本仍保留在追溯列。'
    if '逐点身份更正' in cat:
        return '仅对指定 gamma∞ 单元格对应的数据点覆盖溶质名称；不修改共享列头。'
    if '溶剂身份更正' in cat:
        return '按原表位置对该行溶剂名称使用已核实身份；Corrections 未提供可靠新 CAS/分子量时不猜测。有逐点核实溶质记录的点标为已核实；同一行但未列入逐点核实的点标为范围需复核。'
    if '引用补全' in cat:
        return '以指定 A 列引用单元格为文献组锚点，并向其空白续行传播补全后的引用；Corrections 中明确 DOI 同时传播。'
    return '保留并写入更正追溯字段。'

def find_default_source() -> Optional[Path]:
    candidates = [Path(__file__).resolve().parent / '00_Source' / 'Brouwer_IDAC_Source.xlsx', Path('00_Source_Data') / '03_Brouwer' / 'Brouwer_IDAC_Source.xlsx', Path('03_Brouwer') / 'Brouwer_IDAC_Source.xlsx', Path('Brouwer_IDAC_Source.xlsx')]
    for p in candidates:
        if p.is_file():
            return p
    return None

def read_raw_headers(reader: XLSXReader) -> Tuple[Dict[int, str], Dict[int, str], Dict[int, str], int, List[int]]:
    rows = reader.read_rows(SOURCE_SHEET, [1, 2, 3])
    if not all((r in rows for r in (1, 2, 3))):
        raise RuntimeError('Raw Data must contain header rows 1-3')
    r1, r2, r3 = (rows[1], rows[2], rows[3])
    expected_v, _ = source_number(r3.get(5, ''))
    if expected_v is None:
        raise RuntimeError('Raw Data!E3 does not contain the declared total data-point count')
    expected_points = int(round(expected_v))
    temp_cols: List[int] = []
    max_col = max(set(r1) | set(r2) | set(r3))
    for c in range(7, max_col + 1):
        name = norm_text(r1.get(c))
        tlabel = norm_text(r2.get(c)).casefold()
        glabel = norm_text(r2.get(c + 1)).casefold() if c + 1 <= max_col + 1 else ''
        if name and 'temp' in tlabel and ('y_inf' in glabel or 'y inf' in glabel or 'inf' in glabel):
            temp_cols.append(c)
    if not temp_cols:
        raise RuntimeError('Could not identify any Temp.(K) / y_inf solute column pairs')
    return (r1, r2, r3, expected_points, temp_cols)

def build_point_correction_fields(groups: Sequence[CorrectionGroup], exact_ids: Sequence[str]=()) -> Dict[str, str]:
    if not groups:
        return {'更正状态': '无', '更正类别': '', '更正来源位置': '', '更正记录ID': '', '更正前信息': '', '更正后信息': '', '更正依据': ''}
    cats = join_unique((g.category for g in groups))
    locs = join_unique((g.location for g in groups))
    old = join_unique((g.old for g in groups))
    new = join_unique((g.new for g in groups))
    bases = join_unique((b for g in groups for b in g.bases))
    ids = list(exact_ids)
    for g in groups:
        if not any((rid in ids for rid in g.record_ids)):
            ids.append(f'GROUP::{g.location}')
    return {'更正状态': '已核实更正', '更正类别': cats, '更正来源位置': locs, '更正记录ID': join_unique(ids), '更正前信息': old, '更正后信息': new, '更正依据': bases}

def process_source(source: Path, output: Optional[Path], validate_only: bool, progress_every: int, row_sink=None) -> Dict[str, Any]:
    start = time.time()
    counters: Counter = Counter()
    unique_solutes = set()
    unique_solvents = set()
    unique_refs = set()
    solvent_types: Counter = Counter()
    with XLSXReader(source) as reader:
        required_sheets = {SOURCE_SHEET, DATA_TREATMENT_SHEET, CORRECTIONS_SHEET}
        missing = sorted(required_sheets - set(reader.sheet_targets))
        if missing:
            raise RuntimeError(f'Source workbook missing worksheets: {missing}')
        _, groups = parse_corrections(reader)
        cidx = correction_indexes(groups)
        r1, r2, r3, expected_points, temp_cols = read_raw_headers(reader)
        correction_source_checks = validate_corrections_against_source(reader, groups, r1)
        outbook = row_sink
        temp_output: Optional[Path] = None
        current_reference = ''
        current_ref_anchor = 0
        current_ref_doi = ''
        current_ref_group: Optional[CorrectionGroup] = None
        for rownum, row in reader.iter_rows(SOURCE_SHEET):
            if rownum <= 3:
                continue
            ref_cell = norm_text(row.get(1))
            if ref_cell:
                current_reference = ref_cell
                current_ref_anchor = rownum
                current_ref_group = cidx['ref_by_anchor_row'].get(rownum)
                if current_ref_group:
                    if current_ref_group.new:
                        current_reference = current_ref_group.new
                    current_ref_doi = current_ref_group.doi
                else:
                    current_ref_doi = extract_doi(current_reference)
            year = intish(row.get(2, ''))
            solvent_type_code = norm_text(row.get(3))
            solvent_cas = norm_text(row.get(4))
            solvent_name = norm_text(row.get(5))
            solvent_mw, solvent_mw_comma = source_number(row.get(6, ''))
            solvent_group: Optional[CorrectionGroup] = cidx['solvent_by_row'].get(rownum)
            solvent_verified_solutes = set()
            if solvent_group:
                if solvent_group.new:
                    solvent_name = solvent_group.new
                solvent_verified_solutes = {s.casefold() for s in solvent_group.verified_solutes}
                solvent_cas = ''
                solvent_mw = None
            if solvent_name:
                unique_solvents.add(solvent_name)
            if solvent_type_code:
                solvent_types[solvent_type_code] += 1
            ref_id = stable_ref_id(current_reference, current_ref_anchor or rownum)
            if current_reference:
                unique_refs.add(ref_id)
            for temp_col in temp_cols:
                gamma_col = temp_col + 1
                gamma_raw = norm_text(row.get(gamma_col))
                if not gamma_raw:
                    continue
                temp_raw = norm_text(row.get(temp_col))
                temp_cell = f'{num_to_col(temp_col)}{rownum}'
                gamma_cell = f'{num_to_col(gamma_col)}{rownum}'
                header_cell = f'{num_to_col(temp_col)}1'
                solute = norm_text(r1.get(temp_col))
                solute_category = norm_text(r3.get(temp_col))
                point_groups: List[CorrectionGroup] = []
                exact_record_ids: List[str] = []
                if current_ref_group is not None:
                    point_groups.append(current_ref_group)
                    current_ref_group.matched_points += 1
                if solvent_group is not None:
                    point_groups.append(solvent_group)
                    solvent_group.matched_points += 1
                temp_group = cidx['temp_by_cell'].get(temp_cell)
                if temp_group:
                    point_groups.append(temp_group)
                    temp_group.matched_points += 1
                    exact_record_ids.extend(temp_group.record_ids)
                    if temp_group.new:
                        temp_raw_for_value = temp_group.new
                    else:
                        temp_raw_for_value = temp_raw
                else:
                    temp_raw_for_value = temp_raw
                solute_group = cidx['solute_by_gamma_cell'].get(gamma_cell)
                if solute_group:
                    point_groups.append(solute_group)
                    solute_group.matched_points += 1
                    exact_record_ids.extend(solute_group.record_ids)
                    if solute_group.new:
                        solute = solute_group.new
                temp_value, temp_comma = source_number(temp_raw_for_value)
                gamma_value, gamma_comma = source_number(gamma_raw)
                unique_solutes.add(solute)
                counters['points'] += 1
                if temp_comma:
                    counters['decimal_comma_temperature'] += 1
                if gamma_comma:
                    counters['decimal_comma_gamma'] += 1
                if temp_value is None:
                    counters['temperature_parse_failure'] += 1
                if gamma_value is None:
                    counters['gamma_parse_failure'] += 1
                if gamma_value is not None and gamma_value <= 0:
                    counters['gamma_nonpositive'] += 1
                notes: List[str] = []
                status = 'OK'
                use = '是'
                if temp_comma or gamma_comma:
                    notes.append('原表小数逗号已标准化为小数点；原始文本保留在追溯列')
                if temp_value is None:
                    status = 'SOURCE_ANOMALY'
                    use = '否'
                    notes.append('温度无法解析为数值')
                if gamma_value is None:
                    status = 'SOURCE_ANOMALY'
                    use = '否'
                    notes.append('gamma∞ 无法解析为数值')
                elif gamma_value <= 0:
                    status = 'SOURCE_ANOMALY'
                    use = '否'
                    notes.append('gamma∞ 应为正值；来源中存在非正值，未自动修正')
                correction_fields = build_point_correction_fields(point_groups, exact_record_ids)
                if point_groups and status == 'OK':
                    status = 'OK_CORRECTED'
                    notes.append('存在来源内已核实更正；更正信息已并入本行质量备注，原始依据保留在来源工作簿 Corrections sheet')
                if solvent_group is not None and solvent_verified_solutes:
                    if solute.casefold() not in solvent_verified_solutes:
                        if status != 'SOURCE_ANOMALY':
                            status = 'REVIEW_CORRECTION_SCOPE'
                            use = '否'
                        correction_fields['更正状态'] = '更正范围需复核'
                        notes.append('同一行溶剂身份已在来源表整体更正，但该溶质未出现在 Corrections 的逐点核实清单中')
                        counters['correction_scope_review'] += 1
                if point_groups:
                    counters['corrected_points'] += 1
                if status == 'OK_CORRECTED':
                    counters['ok_corrected'] += 1
                elif status == 'OK':
                    counters['ok'] += 1
                elif status == 'REVIEW_CORRECTION_SCOPE':
                    counters['review'] += 1
                elif status == 'SOURCE_ANOMALY':
                    counters['anomaly'] += 1
                if use == '是':
                    counters['model_yes'] += 1
                else:
                    counters['model_no'] += 1
                if point_groups:
                    corr_bits = []
                    if correction_fields.get('更正类别'):
                        corr_bits.append(f"更正类别={correction_fields['更正类别']}")
                    if correction_fields.get('更正来源位置'):
                        corr_bits.append(f"位置={correction_fields['更正来源位置']}")
                    if correction_fields.get('更正前信息') or correction_fields.get('更正后信息'):
                        corr_bits.append(f"{correction_fields.get('更正前信息', '')} -> {correction_fields.get('更正后信息', '')}")
                    if correction_fields.get('更正依据'):
                        corr_bits.append(f"依据={correction_fields['更正依据']}")
                    if corr_bits:
                        notes.append('Corrections: ' + '；'.join(corr_bits))
                extra_conditions = []
                stype = SOLVENT_TYPE.get(solvent_type_code, '')
                if stype:
                    extra_conditions.append(f'Brouwer solvent type={stype}')
                if solvent_cas:
                    extra_conditions.append(f'solvent CAS={solvent_cas}')
                if solvent_mw is not None:
                    extra_conditions.append(f'solvent MW={solvent_mw:g} g/mol')
                if solute_category:
                    extra_conditions.append(f'Brouwer solute category={solute_category}')
                solvent_component_count: Any = 1 if solvent_type_code in {'1', '2'} else ''
                row_out: Dict[str, Any] = {'IDAC记录ID': f'BROUWER::RawData!{gamma_cell}', 'DOI': current_ref_doi, '文献标题': '', '期刊': '', '发表年份': year, '原始JSON文件': f'{source.name}::Raw Data', '数据块编号': f'RawData!{gamma_cell}', '溶质': solute, '溶质分子式': '', '溶质InChI': '', '溶质InChIKey': '', '溶剂组分数': solvent_component_count, '溶剂': solvent_name, '溶剂InChI': '', '溶剂InChIKey': '', '温度_K': temp_value if temp_value is not None else '', '压力_kPa': '', '无限稀释活度系数': gamma_value if gamma_value is not None else '', '活度系数不确定度': '', '实验方法': '', '相态': '', '标准态': '', '无限稀释判定方式': 'Brouwer 2021 Supplementary Raw Data：IDAC 专用汇编，实验列明确标记为 y_inf', '稀释变量': '', '稀释组分': '', '稀释值': '', '其他固定条件': '；'.join(extra_conditions), '实验目的': '', 'IDAC判定等级': 'SOURCE_原始汇编明确γ∞', '数据质量状态': status, '建议用于建模': use, '质量备注': '；'.join(notes)}
                if solvent_component_count != 1:
                    counters['mixed_excluded'] += 1
                    continue
                # Reviewed bibliography scope is metadata, never a replacement
                # for the frozen original 32 workbook fields.
                row_out['_original_study_reference']=current_reference
                row_out['_original_study_ref_id']=ref_id
                if gamma_value is None or gamma_value <= 0 or temp_value is None or (temp_value <= 0):
                    counters['invalid_excluded'] += 1
                    continue
                counters['kept'] += 1
                counters['kept_model_yes' if use == '是' else 'kept_model_no'] += 1
                if outbook is not None:
                    outbook.write_row(row_out)
            if rownum % progress_every == 0:
                print(f"[PROGRESS] Raw Data row {rownum:,} | IDAC points {counters['points']:,}", flush=True)
        if counters['points'] != expected_points:
            if outbook is not None:
                outbook.close()
            if temp_output and temp_output.exists():
                temp_output.unlink()
            raise RuntimeError(f"Source-count audit failed: Raw Data!E3 declares {expected_points:,} points, but parser unfolded {counters['points']:,}. No formal output was kept.")
        elapsed = time.time() - start
        summary = {'source': str(source), 'expected_points': expected_points, 'output_points': counters['kept'], 'unfolded_source_points': counters['points'], 'mixed_excluded': counters['mixed_excluded'], 'invalid_excluded': counters['invalid_excluded'], 'unique_solutes': len(unique_solutes), 'unique_solvents': len(unique_solvents), 'unique_references': len(unique_refs), 'solvent_type_counts': dict(solvent_types), 'decimal_comma_temperature': counters['decimal_comma_temperature'], 'decimal_comma_gamma': counters['decimal_comma_gamma'], 'temperature_parse_failure': counters['temperature_parse_failure'], 'gamma_parse_failure': counters['gamma_parse_failure'], 'gamma_nonpositive': counters['gamma_nonpositive'], 'corrected_points': counters['corrected_points'], 'correction_scope_review': counters['correction_scope_review'], 'model_yes': counters['kept_model_yes'], 'model_no': counters['kept_model_no'], 'logical_correction_groups': len(groups), 'correction_source_checks': correction_source_checks, 'elapsed_seconds': elapsed}
        return summary
