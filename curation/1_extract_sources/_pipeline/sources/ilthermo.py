"""
NIST ILThermo frozen Activity source -> MASTER + standardized IDAC

This is a BUILD-ONLY script.
It does NOT download ILThermo data and does NOT modify the frozen source.

Expected source:
    00_Source/
    └─ ILThermo.Activity.source/
       ├─ set_ids.txt
       ├─ source_manifest.json
       ├─ COMPLETE.flag
       └─ ilset/
          ├─ <set_id>.json
          └─ <set_id>__<sha1-10>.json   # only for case-insensitive collisions

Outputs:
    01_MASTER.xlsx
    02_IDAC.xlsx

Scientific rules for confirmed IDAC:
1) The NIST /ilset dataset directly reports an "Activity coefficient ..." column.
2) The property target component can be matched to a component in the dataset.
3) The SAME target component has an explicit mole fraction 0 <= x <= --x-threshold
   (default 1e-5) in that experimental row.
4) gamma is taken directly from NIST. Activity/x is never calculated.
5) Missing pressure remains blank.
6) No external SMILES/InChI/InChIKey lookup is performed here.
7) Exactly one solvent component is required; mixed solvents remain in MASTER only.

The IDAC main table uses the same 32-column schema as
02_NIST_ThermoML_IDAC.xlsx / 05_Brouwer_IDAC.xlsx.

Dependencies:
    python -m pip install xlsxwriter

Run:
    python ilthermo_build_simple.py

Validation only:
    python ilthermo_build_simple.py --validate-only
"""
from __future__ import annotations
import argparse
import hashlib
import html
import json
import math
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable
SCRIPT_VERSION = '2.0.1'
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE = SCRIPT_DIR / '00_Source' / 'ILThermo.Activity.source'
DEFAULT_MASTER = '01_MASTER.xlsx'
DEFAULT_IDAC = '02_IDAC.xlsx'
DEFAULT_X_THRESHOLD = 1e-05
EXCEL_TEXT_LIMIT = 32000
MAX_DATA_ROWS_PER_SHEET = 1000000
MASTER_HEADERS = ['数据点ID', '数据来源', 'ILThermo数据集ID', '数据点序号', '原始JSON文件', '参考文献原文', '文献标题', '作者', '期刊及卷页', '发表年份', 'DOI', '原始数据集标题', '实验方法', '系统相态', '体系组分数', '体系组分', '体系组分内部ID', '目标组分', '目标组分内部ID', '目标组分分子式', '性质名称', '性质数值', '活度系数不确定度', '性质相态', '温度_K', '压力_kPa', '组成条件', '其他实验变量', '固定条件', '溶剂_原始字段', '脚注', '组分1名称', '组分1内部ID', '组分1分子式', '组分1摩尔质量_gmol', '组分1样品信息', '组分2名称', '组分2内部ID', '组分2分子式', '组分2摩尔质量_gmol', '组分2样品信息', '组分3名称', '组分3内部ID', '组分3分子式', '组分3摩尔质量_gmol', '组分3样品信息', '原始表头映射JSON', '原始行数据JSON', '原始组分JSON', '原始固定条件JSON']
IDAC_HEADERS = ['IDAC记录ID', 'DOI', '文献标题', '期刊', '发表年份', '原始JSON文件', '数据块编号', '溶质', '溶质分子式', '溶质InChI', '溶质InChIKey', '溶剂组分数', '溶剂', '溶剂InChI', '溶剂InChIKey', '温度_K', '压力_kPa', '无限稀释活度系数', '活度系数不确定度', '实验方法', '相态', '标准态', '无限稀释判定方式', '稀释变量', '稀释组分', '稀释值', '其他固定条件', '实验目的', 'IDAC判定等级', '数据质量状态', '建议用于建模', '质量备注']
MASTER_FIELD_HELP = {'数据点ID': '稳定行级主键：ILTACT::<set_id>::POINT<n>。', '数据来源': '固定为 NIST ILThermo。', 'ILThermo数据集ID': 'NIST ILThermo /ilset 的 case-sensitive set ID。', '原始JSON文件': '相对于 ILThermo.Activity.source 的原始 JSON 路径。', '目标组分': '从直接报告性质列（例如 Activity coefficient of benzene）解析出的目标组分。', '性质数值': 'NIST /ilset data 行中直接报告的性质值。', '活度系数不确定度': '同一 data cell 的第二个数值（若 NIST 提供）。', '温度_K': 'Temperature, K；缺失留空。', '压力_kPa': 'Pressure, kPa；缺失留空，不默认 101.325 kPa。', '组成条件': '所有明确报告的组成变量；不自行补 1-x。', '固定条件': 'NIST constr 原始内容的人可读展开。', '原始表头映射JSON': '完整 dhead，用于审计。', '原始行数据JSON': '完整原始 data 行，用于审计。'}
IDAC_FIELD_HELP = {'IDAC记录ID': '稳定主键：ILTIDAC::<set_id>::POINT<n>。', 'DOI': '仅在 ILThermo 原始 reference 可直接解析到 DOI 时填写，否则留空。', '原始JSON文件': '可回溯到冻结 ILThermo 来源的 JSON。', '数据块编号': 'ILThermo set ID。', '溶质': '直接报告 Activity coefficient 的目标组分，并与近零摩尔分数组分一致。', '溶质分子式': 'ILThermo components 中的原始 formula（HTML SUB 标签已清理）。', '溶质InChI': 'ILThermo /ilset 未提供时留空；本脚本不外部补造。', '溶质InChIKey': 'ILThermo /ilset 未提供时留空；本脚本不外部补造。', '溶剂组分数': '体系总组分中除溶质外的组分数。', '溶剂': '单一溶剂组分；混合溶剂体系仅保留在主表，不进入IDAC。', '溶剂InChI': '本阶段不外部补造，通常留空。', '溶剂InChIKey': '本阶段不外部补造，通常留空。', '无限稀释活度系数': 'NIST 直接报告的 Activity coefficient；绝不由 Activity/x 计算。', 'IDAC判定等级': 'A_明确零组成 或 A_近零组成≤阈值。', '数据质量状态': 'OK 或 SOURCE_ANOMALY。', '建议用于建模': '确认且 gamma>0 的记录为是；源数值异常为否。'}

def clean_text(x: Any) -> str:
    if x is None:
        return ''
    s = html.unescape(str(x))
    s = re.sub('<\\s*/?\\s*SUB\\s*>', '', s, flags=re.I)
    s = re.sub('<[^>]+>', '', s)
    return re.sub('\\s+', ' ', s).strip()

def norm(x: Any) -> str:
    return clean_text(x).casefold()

def as_list(x: Any) -> list:
    if x is None:
        return []
    return x if isinstance(x, list) else [x]

def safe_float(x: Any) -> float | None:
    if x is None or x == '':
        return None
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except Exception:
        return None

def jdump(x: Any) -> str:
    return json.dumps(x, ensure_ascii=False, separators=(',', ':'))

def excel_safe(x: Any) -> Any:
    if x is None:
        return ''
    if isinstance(x, (int, float)) and (not isinstance(x, bool)):
        return x
    s = str(x)
    s = re.sub('[\\x00-\\x08\\x0B\\x0C\\x0E-\\x1F]', '', s)
    if len(s) > EXCEL_TEXT_LIMIT:
        s = s[:EXCEL_TEXT_LIMIT] + ' …[完整内容见原始JSON]'
    return s

def sha256_lines(lines: Iterable[str]) -> str:
    h = hashlib.sha256()
    for line in lines:
        h.update(str(line).encode('utf-8'))
        h.update(b'\n')
    return h.hexdigest()

def stable_id(set_id: str, point_no: int, prefix: str) -> str:
    return f'{prefix}::{set_id}::POINT{point_no}'

def read_set_ids(path: Path) -> list[str]:
    if not path.is_file():
        raise RuntimeError(f'Missing frozen set list: {path}')
    ids = [x.strip() for x in path.read_text(encoding='utf-8').splitlines()]
    ids = [x for x in ids if x]
    if not ids:
        raise RuntimeError('set_ids.txt is empty')
    if len(ids) != len(set(ids)):
        raise RuntimeError('set_ids.txt contains exact duplicate IDs')
    return sorted(ids)

def collision_groups(set_ids: list[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for sid in set_ids:
        groups.setdefault(sid.casefold(), []).append(sid)
    return {k: sorted(v) for k, v in groups.items() if len(v) > 1}

def collision_safe_filename(set_id: str) -> str:
    digest = hashlib.sha1(set_id.encode('utf-8')).hexdigest()[:10]
    return f'{set_id}__{digest}.json'

def filename_for_set(set_id: str, collisions: dict[str, list[str]]) -> str:
    if set_id.casefold() in collisions:
        return collision_safe_filename(set_id)
    return f'{set_id}.json'

def validate_ilset_payload(obj: Any) -> tuple[bool, str]:
    if not isinstance(obj, dict) or not obj:
        return (False, 'top-level response is not a non-empty JSON object')
    for k in ('error', 'errors'):
        if k in obj and obj[k] not in (None, '', False, [], {}):
            return (False, f'server error envelope: {k}={obj[k]!r}')
    if not isinstance(obj.get('dhead'), list) or not obj.get('dhead'):
        return (False, 'missing/non-list/empty dhead')
    if not isinstance(obj.get('data'), list) or not obj.get('data'):
        return (False, 'missing/non-list/empty data')
    if not isinstance(obj.get('components'), list) or not obj.get('components'):
        return (False, 'missing/non-list/empty components')
    return (True, 'ok')

def source_inventory(source: Path) -> dict[str, Any]:
    ilset_dir = source / 'ilset'
    if not ilset_dir.is_dir():
        raise RuntimeError(f'Missing raw ILThermo directory: {ilset_dir}')
    set_ids = read_set_ids(source / 'set_ids.txt')
    collisions = collision_groups(set_ids)
    items = []
    invalid = []
    for sid in set_ids:
        filename = filename_for_set(sid, collisions)
        path = ilset_dir / filename
        if not path.is_file():
            invalid.append((sid, filename, 'missing'))
            continue
        try:
            obj = json.loads(path.read_text(encoding='utf-8-sig'))
        except Exception as exc:
            invalid.append((sid, filename, f'JSON parse failed: {type(exc).__name__}: {exc}'))
            continue
        ok, reason = validate_ilset_payload(obj)
        if not ok:
            invalid.append((sid, filename, reason))
            continue
        items.append((sid, path))
    if invalid:
        msg = '\n'.join((f'  {sid}: {fn} -> {why}' for sid, fn, why in invalid[:30]))
        raise RuntimeError(f'Frozen ILThermo source is incomplete/invalid: {len(invalid)} set(s)\n{msg}')
    return {'set_ids': set_ids, 'collisions': collisions, 'items': items, 'set_ids_sha256': sha256_lines(set_ids)}

def parse_reference(obj: Any) -> dict[str, str]:
    ref = obj if isinstance(obj, dict) else {}
    full = clean_text(ref.get('full', ''))
    title = clean_text(ref.get('title', ''))
    doi = clean_text(ref.get('doi') or ref.get('DOI') or '')
    if not doi:
        m = re.search('\\b10\\.\\d{4,9}/[-._;()/:A-Z0-9]+\\b', full, flags=re.I)
        if m:
            doi = m.group(0).rstrip('.,;)')
    ym = re.search('\\((19\\d{2}|20\\d{2})[a-z]?\\)', full)
    year = ym.group(1) if ym else ''
    authors = full[:ym.start()].strip(' ,;') if ym else ''
    journal_tail = full[ym.end():].strip(' ,;') if ym else ''
    return {'full': full, 'title': title, 'doi': doi, 'year': year, 'authors': authors, 'journal_tail': journal_tail}

def parse_components(response: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for i, raw in enumerate(as_list(response.get('components')), start=1):
        if not isinstance(raw, dict):
            continue
        sample_raw = raw.get('sample', [])
        sample = {}
        if isinstance(sample_raw, list):
            for pair in sample_raw:
                if isinstance(pair, list) and len(pair) >= 2:
                    sample[clean_text(pair[0]).rstrip(':')] = clean_text(pair[1])
        out.append({'index': i, 'id': clean_text(raw.get('idout') or raw.get('id') or ''), 'name': clean_text(raw.get('name', '')), 'formula': clean_text(raw.get('formula', '')), 'mw': safe_float(raw.get('mw')) if raw.get('mw') not in (None, '') else '', 'sample': sample if sample else sample_raw, 'raw': raw})
    return out

def header_pairs(response: dict[str, Any], raw_row: Any) -> list[dict[str, Any]]:
    dhead = as_list(response.get('dhead'))
    row = raw_row if isinstance(raw_row, list) else []
    if len(row) != len(dhead):
        raise ValueError(f'dhead/data width mismatch: dhead={len(dhead)}, row={len(row)}')
    out = []
    for i, head in enumerate(dhead):
        hs = as_list(head)
        label = clean_text(hs[0] if hs else '')
        phase = clean_text(hs[1] if len(hs) > 1 else '')
        cell = row[i]
        vals = cell if isinstance(cell, list) else [cell]
        if len(vals) not in (1, 2):
            raise ValueError(f'Unexpected data cell width {len(vals)} at column {i}: {cell!r}')
        out.append({'index': i, 'label': label, 'phase': phase, 'value': vals[0] if vals else None, 'error': vals[1] if len(vals) == 2 else None, 'raw': cell})
    return out

def is_temperature(label: str) -> bool:
    return bool(re.search('\\btemperature\\b', norm(label)))

def is_pressure(label: str) -> bool:
    return bool(re.search('\\bpressure\\b', norm(label)))

def composition_info(label: str) -> tuple[str, str, bool]:
    """
    Return (composition_type, component_token, explicitly_solvent_prefixed).

    Handles:
      Mole fraction of benzene
      Solvent: Mole fraction of glycerol
      Solvent: Mole ratio of ... to other component of binary solvent
      MolaLity of ...
    """
    s = clean_text(label)
    solvent_prefix = bool(re.match('^\\s*solvent\\s*:', s, flags=re.I))
    if solvent_prefix:
        s = re.sub('^\\s*solvent\\s*:\\s*', '', s, flags=re.I)
    patterns = [('Mole fraction', '^mole fraction of\\s+(.+)$'), ('Mass fraction', '^(?:mass|weight) fraction of\\s+(.+)$'), ('Volume fraction', '^volume fraction of\\s+(.+)$'), ('Molality', '^molality of\\s+(.+?)(?:,\\s*mol/kg)?$'), ('Molarity', '^molarity of\\s+(.+?)(?:,\\s*mol/(?:l|L))?$'), ('Mole ratio', '^mole ratio of\\s+(.+)$')]
    for typ, patt in patterns:
        m = re.match(patt, s, flags=re.I)
        if m:
            return (typ, clean_text(m.group(1)), solvent_prefix)
    return ('', '', solvent_prefix)

def component_from_token(token: str, comps: list[dict[str, Any]]) -> dict[str, Any] | None:
    nt = norm(token)
    if not nt:
        return None
    m = re.search('\\bcomponent\\s*([123])\\b', nt)
    if m:
        idx = int(m.group(1))
        if 1 <= idx <= len(comps):
            return comps[idx - 1]
    exact = [c for c in comps if norm(c['name']) == nt]
    if len(exact) == 1:
        return exact[0]
    matches = []
    for c in comps:
        nn = norm(c['name'])
        if nn and (nn in nt or nt in nn):
            matches.append((len(nn), c))
    if not matches:
        return None
    matches.sort(key=lambda x: -x[0])
    if len(matches) == 1 or matches[0][0] > matches[1][0]:
        return matches[0][1]
    return None

def activity_coefficient_target(label: str, comps: list[dict[str, Any]]) -> dict[str, Any] | None:
    s = clean_text(label)
    m = re.match('^activity coefficient of\\s+(.+)$', s, flags=re.I)
    if not m:
        return None
    return component_from_token(clean_text(m.group(1)), comps)

def find_property_pair(pairs: list[dict[str, Any]]) -> dict[str, Any] | None:
    ac = [p for p in pairs if re.match('^activity coefficient of\\b', clean_text(p['label']), flags=re.I)]
    if len(ac) == 1:
        return ac[0]
    if len(ac) > 1:
        return None
    act = [p for p in pairs if re.match('^activity of\\b', clean_text(p['label']), flags=re.I) and 'coefficient' not in norm(p['label'])]
    return act[0] if len(act) == 1 else None

def fixed_condition_text(raw: Any) -> str:
    if raw in (None, [], {}, ''):
        return ''
    if isinstance(raw, str):
        return clean_text(raw)
    return jdump(raw)

def condition_value(pairs: list[dict[str, Any]], kind: str) -> float | None:
    pred = is_temperature if kind == 'temperature' else is_pressure
    for p in pairs:
        if pred(p['label']):
            return safe_float(p['value'])
    return None

def phase_text(response: dict[str, Any], prop_pair: dict[str, Any] | None) -> str:
    if prop_pair and prop_pair.get('phase'):
        return clean_text(prop_pair['phase'])
    vals = [clean_text(x) for x in as_list(response.get('phases'))]
    return ' | '.join((x for x in vals if x))

def make_master_row(source: Path, sid: str, path: Path, response: dict[str, Any], point_no: int, raw_row: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    pairs = header_pairs(response, raw_row)
    comps = parse_components(response)
    ref = parse_reference(response.get('ref'))
    prop_pair = find_property_pair(pairs)
    target = activity_coefficient_target(prop_pair['label'], comps) if prop_pair else None
    temp = condition_value(pairs, 'temperature')
    pressure = condition_value(pairs, 'pressure')
    composition_parts = []
    other_vars = []
    for p in pairs:
        if prop_pair is p or is_temperature(p['label']) or is_pressure(p['label']):
            continue
        typ, token, solvent_pref = composition_info(p['label'])
        if typ:
            prefix = 'Solvent: ' if solvent_pref else ''
            composition_parts.append(f"{prefix}{p['label']} = {clean_text(p['value'])}")
        else:
            other_vars.append(f"{p['label']} = {clean_text(p['value'])}")
    row = {h: '' for h in MASTER_HEADERS}
    row.update({'数据点ID': stable_id(sid, point_no, 'ILTACT'), '数据来源': 'NIST ILThermo', 'ILThermo数据集ID': sid, '数据点序号': point_no, '原始JSON文件': str(path.relative_to(source)).replace('\\', '/'), '参考文献原文': ref['full'], '文献标题': ref['title'], '作者': ref['authors'], '期刊及卷页': ref['journal_tail'], '发表年份': ref['year'], 'DOI': ref['doi'], '原始数据集标题': clean_text(response.get('title', '')), '实验方法': clean_text(response.get('expmeth', '')), '系统相态': ' | '.join((clean_text(x) for x in as_list(response.get('phases')) if clean_text(x))), '体系组分数': len(comps), '体系组分': ' | '.join((c['name'] for c in comps)), '体系组分内部ID': ' | '.join((c['id'] for c in comps)), '目标组分': target['name'] if target else '', '目标组分内部ID': target['id'] if target else '', '目标组分分子式': target['formula'] if target else '', '性质名称': clean_text(prop_pair['label']) if prop_pair else '', '性质数值': prop_pair['value'] if prop_pair else '', '活度系数不确定度': prop_pair['error'] if prop_pair and prop_pair.get('error') is not None else '', '性质相态': clean_text(prop_pair['phase']) if prop_pair else '', '温度_K': temp if temp is not None else '', '压力_kPa': pressure if pressure is not None else '', '组成条件': ' | '.join(composition_parts), '其他实验变量': ' | '.join(other_vars), '固定条件': fixed_condition_text(response.get('constr')), '溶剂_原始字段': clean_text(response.get('solvent', '')), '脚注': clean_text(response.get('footer', '')), '原始表头映射JSON': jdump(response.get('dhead', [])), '原始行数据JSON': jdump(raw_row), '原始组分JSON': jdump(response.get('components', [])), '原始固定条件JSON': jdump(response.get('constr', []))})
    for i, comp in enumerate(comps[:3], start=1):
        row[f'组分{i}名称'] = comp['name']
        row[f'组分{i}内部ID'] = comp['id']
        row[f'组分{i}分子式'] = comp['formula']
        row[f'组分{i}摩尔质量_gmol'] = comp['mw']
        row[f'组分{i}样品信息'] = jdump(comp['sample'])
    ctx = {'sid': sid, 'path': path, 'response': response, 'pairs': pairs, 'comps': comps, 'ref': ref, 'prop_pair': prop_pair, 'target': target, 'temperature': temp, 'pressure': pressure, 'master': row}
    return (row, ctx)

def make_idac_row(ctx: dict[str, Any], point_no: int, threshold: float) -> tuple[dict[str, Any] | None, str]:
    p = ctx['prop_pair']
    if not p:
        return (None, 'no_property_column')
    if not re.match('^activity coefficient of\\b', clean_text(p['label']), flags=re.I):
        return (None, 'not_activity_coefficient')
    target = ctx['target']
    if not target:
        return (None, 'activity_coefficient_target_unresolved')
    target_x = []
    for pair in ctx['pairs']:
        typ, token, _solvent_pref = composition_info(pair['label'])
        if typ != 'Mole fraction':
            continue
        comp = component_from_token(token, ctx['comps'])
        if comp and comp['id'] == target['id']:
            v = safe_float(pair['value'])
            if v is not None:
                target_x.append((pair, v))
    if not target_x:
        return (None, 'no_target_mole_fraction')
    near_zero = [(pair, v) for pair, v in target_x if 0 <= v <= threshold]
    if len(near_zero) != 1:
        if not near_zero:
            return (None, 'target_not_at_infinite_dilution')
        return (None, 'multiple_target_mole_fraction_columns')
    dilution_pair, x_value = near_zero[0]
    gamma = safe_float(p['value'])
    quality = 'OK'
    use = '是'
    notes = []
    if gamma is None or gamma <= 0:
        quality = 'SOURCE_ANOMALY'
        use = '否'
        notes.append('NIST直接报告的活度系数缺失或非正；保留追溯但默认不用于建模')
    comps = ctx['comps']
    solvents = [c for c in comps if c['id'] != target['id']]
    if len(solvents) != 1:
        return (None, 'mixed_solvent_excluded' if len(solvents) > 1 else 'no_solvent_component')
    extra_conditions = []
    for pair in ctx['pairs']:
        if pair is dilution_pair or is_temperature(pair['label']) or is_pressure(pair['label']):
            continue
        typ, token, solvent_pref = composition_info(pair['label'])
        if not typ:
            continue
        comp = component_from_token(token, solvents)
        if comp or solvent_pref:
            extra_conditions.append(f"{pair['label']} = {clean_text(pair['value'])}")
    fixed = fixed_condition_text(ctx['response'].get('constr'))
    if fixed:
        extra_conditions.append(f'constr = {fixed}')
    ref = ctx['ref']
    grade = 'A_明确零组成' if abs(x_value) <= 1e-14 else f'A_近零组成≤{threshold:g}'
    row = {h: '' for h in IDAC_HEADERS}
    row.update({'IDAC记录ID': stable_id(ctx['sid'], point_no, 'ILTIDAC'), 'DOI': ref['doi'], '文献标题': ref['title'], '期刊': ref['journal_tail'], '发表年份': ref['year'], '原始JSON文件': ctx['master']['原始JSON文件'], '数据块编号': ctx['sid'], '溶质': target['name'], '溶质分子式': target['formula'], '溶质InChI': '', '溶质InChIKey': '', '溶剂组分数': len(solvents), '溶剂': ' + '.join((c['name'] for c in solvents)), '溶剂InChI': '', '溶剂InChIKey': '', '温度_K': ctx['temperature'] if ctx['temperature'] is not None else '', '压力_kPa': ctx['pressure'] if ctx['pressure'] is not None else '', '无限稀释活度系数': gamma if gamma is not None else '', '活度系数不确定度': p['error'] if p.get('error') is not None else '', '实验方法': clean_text(ctx['response'].get('expmeth', '')), '相态': phase_text(ctx['response'], p), '标准态': '', '无限稀释判定方式': f"NIST直接报告 Activity coefficient of {target['name']}；同一数据行中 Mole fraction of {target['name']} = {x_value:g} <= {threshold:g}", '稀释变量': 'Mole fraction', '稀释组分': target['name'], '稀释值': x_value, '其他固定条件': ' | '.join(extra_conditions), '实验目的': '', 'IDAC判定等级': grade, '数据质量状态': quality, '建议用于建模': use, '质量备注': '；'.join(notes)})
    return (row, 'confirmed' if quality == 'OK' else 'source_anomaly')
