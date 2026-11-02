"""
ThermoML v2020-09-30 JSON -> 2 个 Excel：可读母表 + IDAC 筛选表

源目录保持原样：
    ThermoML.v2020-09-30/
        10.1007/*.json
        10.1016/*.json
        10.1021/*.json

默认输出两个文件到脚本所在来源目录：
    01_MASTER.xlsx
    02_IDAC.xlsx

母表一行 = 一个 NumValues 中的 PropertyValue。
IDAC 表一行 = 一个单溶剂的无限稀释活度系数数据点；混合溶剂只保留在母表，源数值异常保留并标记。

母表超过 Excel 单 sheet 1,048,576 行时，会在同一个工作簿内自动拆成
DATA_001 / DATA_002 / DATA_003 ...，不会生成额外数据文件。
解析摘要和异常也写进母表工作簿内部，不再生成 audit/csv/warnings 等旁文件。

依赖：
    Python 3.9+
    XlsxWriter  (python -m pip install XlsxWriter)
    orjson 可选，用于提速  (python -m pip install orjson)

完整运行（重建母表 + IDAC）：
    python thermoml_build_simple.py

只重建 IDAC（母表数据页保持原样，同步更新其 IDAC 运行摘要）：
    python thermoml_build_simple.py --idac-only

快速测试前 20 个 JSON：
    python thermoml_build_simple.py --limit-files 20 --output-dir ./test_output
"""
from __future__ import annotations
import argparse
import copy
import csv
import gzip
import json
import math
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from xml.etree import ElementTree as ET
from zipfile import ZipFile
try:
    import orjson
except Exception:
    orjson = None
MASTER_HEADERS = ['数据点ID', 'DOI', '文献标题', '期刊', '发表年份', '作者', '原始JSON文件', '数据类型', '数据块编号', '数据点序号', '体系组分数', '体系组分', '体系组分InChIKey', '目标组分', '目标组分分子式', '目标组分InChI', '目标组分InChIKey', '溶剂', '溶剂InChIKey', '性质大类', '性质名称', '性质数值', '单位', '实验方法', '相态', '标准态', '温度_K', '压力_kPa', '组成条件', '其他实验变量', '固定条件', '实验值不确定度', '不确定度说明', '实验目的', '数据录入日期', 'ThermoML编译者', 'ThermoML贡献者', '化合物内部编号', '性质内部编号', '变量内部编号']
IDAC_HEADERS = ['IDAC记录ID', 'DOI', '文献标题', '期刊', '发表年份', '原始JSON文件', '数据块编号', '溶质', '溶质分子式', '溶质InChI', '溶质InChIKey', '溶剂组分数', '溶剂', '溶剂InChI', '溶剂InChIKey', '温度_K', '压力_kPa', '无限稀释活度系数', '活度系数不确定度', '实验方法', '相态', '标准态', '无限稀释判定方式', '稀释变量', '稀释组分', '稀释值', '其他固定条件', '实验目的', 'IDAC判定等级', '数据质量状态', '建议用于建模', '质量备注']
MASTER_FIELD_HELP = {'数据点ID': '稳定、可追溯的行级唯一标识：DOI/文件 + 数据块 + 数据点 + PropertyValue。', 'DOI': '原始文献 DOI。', '文献标题': '原始论文标题。', '期刊': '来源期刊。', '发表年份': '文献发表年份。', '作者': 'ThermoML Citation 中的作者列表。', '原始JSON文件': '相对于源目录的 JSON 路径，可直接回溯原始记录。', '数据类型': '纯物质/混合物数据（PureOrMixtureData）或反应数据（ReactionData）。', '数据块编号': 'ThermoML 中 nPureOrMixtureDataNumber / nReactionDataNumber。', '数据点序号': '当前数据块 NumValues 中的 1-based 数据点序号。', '体系组分数': '该数据块涉及的组分/反应参与物数量。', '体系组分': '人可读的全部组分；反应数据同时显示化学计量数和相态。', '体系组分InChIKey': '与体系组分对应的标准 InChIKey。', '目标组分': '若性质明确绑定某个 RegNum，则显示对应化合物。', '目标组分分子式': '目标组分分子式。', '目标组分InChI': '目标组分标准 InChI。', '目标组分InChIKey': '目标组分标准 InChIKey。', '溶剂': 'ThermoML Property 中明确指定的 solvent；不主动猜测普通物性数据的溶剂。', '溶剂InChIKey': '溶剂标准 InChIKey。', '性质大类': 'PropertyGroup 下的大类，例如 ActivityFugacityOsmoticProp。', '性质名称': 'ePropName；若原始名称含单位，单位拆到“单位”列。', '性质数值': 'nPropValue；若原始数据为上下限，则保留为 ≤x / ≥x。', '单位': '从 ThermoML 性质名称提取的单位；明确无量纲的性质标为“无量纲”。', '实验方法': 'eMethodName / sMethodName。', '相态': 'Property phase；若缺失则保留数据块的 phase 信息。', '标准态': 'ThermoML eStandardState。', '温度_K': '从 Variable / Constraint / Reaction property metadata 中统一提取的温度。', '压力_kPa': '统一换算为 kPa 的压力；原始条件仍完整保留在其他列。', '组成条件': '所有组成相关 Variable/Constraint，含具体组分与数值。', '其他实验变量': '除温度、压力、组成以外的逐点 Variable。', '固定条件': 'ThermoML Constraint 的完整人可读表示。', '实验值不确定度': 'PropertyValue 中的 expanded uncertainty 等数值信息。', '不确定度说明': 'Property/PropertyValue 中的不确定度评估方法、置信水平等。', '实验目的': 'ThermoML eExpPurpose/sExpPurpose。', '数据录入日期': 'dateDateAdded。', 'ThermoML编译者': 'sCompiler。', 'ThermoML贡献者': 'sContributor。', '化合物内部编号': '当前数据块涉及的 nOrgNum -> 化合物名称映射。', '性质内部编号': '当前 PropertyValue 的 nPropNumber。', '变量内部编号': 'nVarNumber -> Variable 定义映射，便于回溯。'}
IDAC_FIELD_HELP = {'IDAC记录ID': '来自母表数据点的唯一标识。', 'DOI': '原始文献 DOI。', '文献标题': '原始论文标题。', '期刊': '来源期刊。', '发表年份': '发表年份。', '原始JSON文件': '原始 JSON 路径。', '数据块编号': '原始 ThermoML 数据块编号。', '溶质': '被无限稀释的目标组分。', '溶质分子式': '溶质分子式。', '溶质InChI': '溶质标准 InChI。', '溶质InChIKey': '溶质标准 InChIKey。', '溶剂组分数': '本表仅保留1个溶剂组分；混合溶剂记录排除，仍保留在母表。', '溶剂': 'ThermoML 显式 solvent；缺失时仅在可明确判断时由体系中除溶质外的组分推断。', '溶剂InChI': '溶剂标准 InChI。', '溶剂InChIKey': '溶剂标准 InChIKey。', '温度_K': 'IDAC 测量温度。', '压力_kPa': 'IDAC 测量压力。', '无限稀释活度系数': '实验 gamma^infinity。', '活度系数不确定度': '对应 PropertyValue 的不确定度。', '实验方法': '例如 Chromatography。', '相态': '通常为 Liquid。', '标准态': 'ThermoML eStandardState。', '无限稀释判定方式': '程序为何把此行识别为 IDAC；用于人工审计。', '稀释变量': '例如 Mole fraction。', '稀释组分': '被设为 0（或明确无限稀释）的组分。', '稀释值': '通常为 0。', '其他固定条件': '除温度、压力和用于判定无限稀释的组成条件外的 Constraint。', '实验目的': 'ThermoML 实验目的。', 'IDAC判定等级': 'A=结构直接证明无限稀释；B=数据块无组成变量，但文献明确说明该组数据为 IDAC。', '数据质量状态': 'OK=可直接使用；SOURCE_ANOMALY=ThermoML 数值存在已确认的源异常，保留但需回原文核对。', '建议用于建模': '是/否。源数据异常行保留以保证完整性，但默认不进入建模数据。', '质量备注': '对判定依据或源数据异常的可读说明。'}

def as_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]

def clean_text(x: Any) -> str:
    if x is None:
        return ''
    if isinstance(x, (str, int, float)):
        return str(x).strip()
    return ''

def join_nonempty(items: Iterable[Any], sep: str=' | ') -> str:
    vals = [clean_text(x) for x in items]
    return sep.join((v for v in vals if v))

def first_common_name(compound: Dict[str, Any]) -> str:
    names = as_list(compound.get('sCommonName'))
    for name in names:
        s = clean_text(name)
        if s:
            return s
    formula = clean_text(compound.get('sFormulaMolec'))
    return formula or '(未命名化合物)'

def orgnum_from_regnum(regnum: Any) -> Optional[int]:
    if isinstance(regnum, dict):
        v = regnum.get('nOrgNum')
    else:
        v = None
    try:
        return int(v) if v is not None else None
    except (TypeError, ValueError):
        return None

def orgnums_from_regnum_container(obj: Any) -> List[int]:
    """Accept {'RegNum': {... or [...]}} or a RegNum itself."""
    if obj is None:
        return []
    if isinstance(obj, dict) and 'RegNum' in obj:
        regs = as_list(obj.get('RegNum'))
    else:
        regs = as_list(obj)
    out: List[int] = []
    for reg in regs:
        n = orgnum_from_regnum(reg)
        if n is not None:
            out.append(n)
    return out

def safe_float(x: Any) -> Optional[float]:
    if x is None or x == '':
        return None
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None

def is_zero(x: Any, tol: float=1e-14) -> bool:
    v = safe_float(x)
    return v is not None and abs(v) <= tol

def load_json_file(path: Path) -> Dict[str, Any]:
    if orjson is not None:
        return orjson.loads(path.read_bytes())
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)

def dict_without_noise(d: Any) -> Dict[str, Any]:
    if not isinstance(d, dict):
        return {}
    return {k: v for k, v in d.items() if k != 'tml_elements'}

def simple_dict_text(d: Any, sep: str='; ') -> str:
    """Human-readable, deterministic representation for small metadata dicts."""
    if not isinstance(d, dict):
        return clean_text(d)
    parts: List[str] = []
    for k in sorted(d.keys()):
        if k == 'tml_elements':
            continue
        v = d[k]
        if isinstance(v, dict):
            sub = simple_dict_text(v, sep=', ')
            if sub:
                parts.append(f'{k}({sub})')
        elif isinstance(v, list):
            text = ', '.join((clean_text(x) if not isinstance(x, dict) else simple_dict_text(x, sep=', ') for x in v))
            if text:
                parts.append(f'{k}={text}')
        else:
            s = clean_text(v)
            if s:
                parts.append(f'{k}={s}')
    return sep.join(parts)

def split_name_unit(raw_name: str) -> Tuple[str, str]:
    raw_name = clean_text(raw_name)
    if not raw_name:
        return ('', '')
    if ',' in raw_name:
        left, right = raw_name.rsplit(',', 1)
        unit = right.strip()
        if unit and len(unit) <= 40:
            return (left.strip(), unit)
    low = raw_name.casefold()
    dimensionless_markers = ['activity coefficient', 'fugacity coefficient', 'osmotic coefficient', 'mole fraction', 'mass fraction', 'compressibility factor', 'refractive index']
    if any((m in low for m in dimensionless_markers)):
        return (raw_name, '无量纲')
    return (raw_name, '')

def property_value_text(pv: Dict[str, Any]) -> Any:
    if 'nPropValue' in pv:
        v = safe_float(pv.get('nPropValue'))
        return v if v is not None else clean_text(pv.get('nPropValue'))
    lim = pv.get('PropLimit')
    if isinstance(lim, dict):
        upper = lim.get('nPropUpperLimitValue')
        lower = lim.get('nPropLowerLimitValue')
        if upper is not None and lower is not None:
            return f'{lower}–{upper}'
        if upper is not None:
            return f'≤{upper}'
        if lower is not None:
            return f'≥{lower}'
    return ''

def build_compound_map(doc: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    cmap: Dict[int, Dict[str, Any]] = {}
    for comp in as_list(doc.get('Compound')):
        if not isinstance(comp, dict):
            continue
        n = orgnum_from_regnum(comp.get('RegNum'))
        if n is None:
            continue
        cmap[n] = {'orgnum': n, 'name': first_common_name(comp), 'formula': clean_text(comp.get('sFormulaMolec')), 'inchi': clean_text(comp.get('sStandardInChI')), 'inchikey': clean_text(comp.get('sStandardInChIKey'))}
    return cmap

def compound(cmap: Dict[int, Dict[str, Any]], orgnum: Optional[int]) -> Dict[str, Any]:
    if orgnum is None:
        return {'orgnum': '', 'name': '', 'formula': '', 'inchi': '', 'inchikey': ''}
    return cmap.get(orgnum, {'orgnum': orgnum, 'name': f'OrgNum {orgnum}', 'formula': '', 'inchi': '', 'inchikey': ''})

def citation_fields(doc: Dict[str, Any]) -> Dict[str, str]:
    c = doc.get('Citation') if isinstance(doc.get('Citation'), dict) else {}
    return {'doi': clean_text(c.get('sDOI')), 'title': clean_text(c.get('sTitle')), 'journal': clean_text(c.get('sPubName')), 'year': clean_text(c.get('yrPubYr')), 'authors': join_nonempty(as_list(c.get('sAuthor')), sep='; ')}

def citation_explicitly_idac(doc: Dict[str, Any]) -> bool:
    """High-recall semantic evidence that the article explicitly reports IDAC.

    This is deliberately used only when the relevant property point/block has no
    composition metadata. It therefore recovers true IDAC datasets such as
    10.1021/je0102107 without converting finite-composition VLE/activity data
    merely because an abstract mentions an infinite-dilution limit.
    """
    c = doc.get('Citation') if isinstance(doc.get('Citation'), dict) else {}
    text = ' '.join([clean_text(c.get('sTitle')), clean_text(c.get('sAbstract')), join_nonempty(as_list(c.get('sKeyword')), sep=' ')])
    n = normalize_activity_name(text)
    patterns = ('activity coefficient at infinite dilution', 'activity coefficients at infinite dilution', 'infinite dilution activity coefficient', 'infinite dilution activity coefficients', 'limiting activity coefficient', 'limiting activity coefficients')
    return any((p in n for p in patterns))

def property_definition(prop: Dict[str, Any]) -> Dict[str, Any]:
    pmid = prop.get('Property-MethodID') if isinstance(prop.get('Property-MethodID'), dict) else {}
    pgroup = pmid.get('PropertyGroup') if isinstance(pmid.get('PropertyGroup'), dict) else {}
    group_name = ''
    details: Dict[str, Any] = {}
    for k, v in pgroup.items():
        if k == 'tml_elements':
            continue
        group_name = k
        details = v if isinstance(v, dict) else {}
        break
    raw_name = clean_text(details.get('ePropName'))
    if not raw_name:
        for k, v in details.items():
            if k.lower().endswith('propname'):
                raw_name = clean_text(v)
                if raw_name:
                    break
    method_parts = []
    for key in ('eMethodName', 'sMethodName'):
        val = details.get(key)
        if val:
            method_parts.extend(as_list(val))
    method = join_nonempty(method_parts, sep='; ')
    target_orgnum = orgnum_from_regnum(pmid.get('RegNum'))
    if target_orgnum is None:
        target_orgnum = orgnum_from_regnum(prop.get('RegNum'))
    solvent_orgnums = orgnums_from_regnum_container(prop.get('Solvent'))
    phase_obj = prop.get('PropPhaseID') if isinstance(prop.get('PropPhaseID'), dict) else {}
    phase = clean_text(phase_obj.get('ePropPhase'))
    name, unit = split_name_unit(raw_name)
    return {'nPropNumber': prop.get('nPropNumber'), 'group': group_name, 'name': name, 'raw_name': raw_name, 'unit': unit, 'method': method, 'target_orgnum': target_orgnum, 'solvent_orgnums': solvent_orgnums, 'phase': phase, 'standard_state': clean_text(prop.get('eStandardState')), 'temperature_K': safe_float(prop.get('nTemperature-K')), 'pressure_kPa': safe_float(prop.get('nPressure-kPa')), 'uncertainty_meta': prop.get('CombinedUncertainty') if isinstance(prop.get('CombinedUncertainty'), dict) else {}}

def variable_or_constraint_definition(item: Dict[str, Any], kind: str) -> Dict[str, Any]:
    if kind == 'variable':
        number = item.get('nVarNumber')
        idobj = item.get('VariableID') if isinstance(item.get('VariableID'), dict) else {}
        typeobj = idobj.get('VariableType') if isinstance(idobj.get('VariableType'), dict) else {}
        phaseobj = item.get('VarPhaseID') if isinstance(item.get('VarPhaseID'), dict) else {}
        phase = clean_text(phaseobj.get('eVarPhase'))
    else:
        number = item.get('nConstraintNumber')
        idobj = item.get('ConstraintID') if isinstance(item.get('ConstraintID'), dict) else {}
        typeobj = idobj.get('ConstraintType') if isinstance(idobj.get('ConstraintType'), dict) else {}
        phaseobj = item.get('ConstraintPhaseID') if isinstance(item.get('ConstraintPhaseID'), dict) else {}
        phase = clean_text(phaseobj.get('eConstraintPhase'))
    label = ''
    type_key = ''
    for k, v in typeobj.items():
        if k == 'tml_elements':
            continue
        type_key = k
        label = clean_text(v)
        if label:
            break
    orgnum = orgnum_from_regnum(idobj.get('RegNum'))
    return {'number': number, 'type_key': type_key, 'label': label, 'orgnum': orgnum, 'phase': phase}

def classify_condition(label: str, type_key: str='') -> str:
    s = f'{type_key} {label}'.casefold()
    if 'temperature' in s:
        return 'temperature'
    if 'pressure' in s:
        return 'pressure'
    composition_words = ('composition', 'fraction', 'molality', 'molarity', 'concentration', 'amount ratio', 'mass ratio', 'mole ratio')
    if any((w in s for w in composition_words)):
        return 'composition'
    return 'other'

def pressure_to_kpa(value: Any, label: str) -> Optional[float]:
    v = safe_float(value)
    if v is None:
        return None
    s = label.casefold().replace(' ', '')
    if 'mpa' in s:
        return v * 1000.0
    if 'kpa' in s:
        return v
    if re.search('(^|[,/(])pa($|[)/*])', s) or s.endswith(',pa'):
        return v / 1000.0
    if 'bar' in s:
        return v * 100.0
    if 'atm' in s or 'atmosphere' in s:
        return v * 101.325
    return None

def temperature_to_k(value: Any, label: str) -> Optional[float]:
    v = safe_float(value)
    if v is None:
        return None
    s = label.casefold().replace(' ', '')
    if ',k' in s or s.endswith('k') or 'kelvin' in s:
        return v
    if '°c' in s or ',c' in s or 'celsius' in s:
        return v + 273.15
    return None

def condition_text(defn: Dict[str, Any], value: Any, cmap: Dict[int, Dict[str, Any]]) -> str:
    label = clean_text(defn.get('label')) or clean_text(defn.get('type_key')) or '条件'
    orgnum = defn.get('orgnum')
    comp_name = compound(cmap, orgnum).get('name', '') if orgnum is not None else ''
    phase = clean_text(defn.get('phase'))
    left = label
    if comp_name:
        left += f'[{comp_name}]'
    if phase:
        left += f'({phase})'
    return f'{left}={clean_text(value)}'

def block_phases(block: Dict[str, Any], cmap: Dict[int, Dict[str, Any]]) -> str:
    parts: List[str] = []
    for p in as_list(block.get('PhaseID')):
        if not isinstance(p, dict):
            continue
        phase = clean_text(p.get('ePhase'))
        n = orgnum_from_regnum(p.get('RegNum'))
        if phase and n is not None:
            parts.append(f"{phase}[{compound(cmap, n)['name']}]")
        elif phase:
            parts.append(phase)
    if parts:
        return join_nonempty(parts)
    for p in as_list(block.get('Participant')):
        if isinstance(p, dict):
            phase = clean_text(p.get('ePhase'))
            if phase:
                parts.append(phase)
    return join_nonempty(dict.fromkeys(parts))

def system_components(block: Dict[str, Any], data_kind: str, cmap: Dict[int, Dict[str, Any]]) -> Tuple[List[int], str, str, str]:
    orgnums: List[int] = []
    readable: List[str] = []
    inchikeys: List[str] = []
    internal_map: List[str] = []
    if data_kind == 'POMD':
        entries = as_list(block.get('Component'))
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            n = orgnum_from_regnum(entry.get('RegNum'))
            if n is None:
                continue
            orgnums.append(n)
            c = compound(cmap, n)
            readable.append(c['name'])
            inchikeys.append(c['inchikey'])
            internal_map.append(f"{n}={c['name']}")
    else:
        entries = as_list(block.get('Participant'))
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            n = orgnum_from_regnum(entry.get('RegNum'))
            if n is None:
                continue
            orgnums.append(n)
            c = compound(cmap, n)
            stoich = clean_text(entry.get('nStoichiometricCoef'))
            phase = clean_text(entry.get('ePhase'))
            desc = c['name']
            extras = []
            if stoich:
                extras.append(f'ν={stoich}')
            if phase:
                extras.append(phase)
            if extras:
                desc += f" ({', '.join(extras)})"
            readable.append(desc)
            inchikeys.append(c['inchikey'])
            internal_map.append(f"{n}={c['name']}")
    seen = set()
    unique_orgnums = []
    for n in orgnums:
        if n not in seen:
            seen.add(n)
            unique_orgnums.append(n)
    return (unique_orgnums, join_nonempty(readable), join_nonempty(inchikeys), join_nonempty(internal_map))

def uncertainty_fields(prop_meta: Dict[str, Any], pv: Dict[str, Any]) -> Tuple[Any, str]:
    pv_unc = pv.get('CombinedUncertainty') if isinstance(pv.get('CombinedUncertainty'), dict) else {}
    prop_unc = prop_meta if isinstance(prop_meta, dict) else {}
    value = ''
    value_candidates = ['nCombExpandUncertValue', 'nCombStdUncertValue', 'nCombUncertValue']
    for key in value_candidates:
        if key in pv_unc:
            v = safe_float(pv_unc.get(key))
            value = v if v is not None else clean_text(pv_unc.get(key))
            break
    desc_parts: List[str] = []
    for src_name, src in (('Property', prop_unc), ('PropertyValue', pv_unc)):
        if not src:
            continue
        slim = {k: v for k, v in src.items() if k not in {'tml_elements', 'nCombExpandUncertValue', 'nCombStdUncertValue', 'nCombUncertValue'}}
        text = simple_dict_text(slim)
        if text:
            desc_parts.append(f'{src_name}: {text}')
    return (value, join_nonempty(desc_parts, sep=' | '))

def variable_internal_map(var_defs: Dict[int, Dict[str, Any]], cmap: Dict[int, Dict[str, Any]]) -> str:
    parts = []
    for n in sorted(var_defs):
        d = var_defs[n]
        label = d.get('label') or d.get('type_key') or 'Variable'
        org = d.get('orgnum')
        if org is not None:
            label = f"{label}[{compound(cmap, org)['name']}]"
        parts.append(f'{n}={label}')
    return join_nonempty(parts)

def extract_conditions(block: Dict[str, Any], point: Dict[str, Any], cmap: Dict[int, Dict[str, Any]], propdef: Dict[str, Any]) -> Dict[str, Any]:
    var_defs: Dict[int, Dict[str, Any]] = {}
    for item in as_list(block.get('Variable')):
        if not isinstance(item, dict):
            continue
        d = variable_or_constraint_definition(item, 'variable')
        try:
            n = int(d['number'])
            var_defs[n] = d
        except Exception:
            continue
    constraint_defs: List[Tuple[Dict[str, Any], Any]] = []
    for item in as_list(block.get('Constraint')):
        if not isinstance(item, dict):
            continue
        d = variable_or_constraint_definition(item, 'constraint')
        value = item.get('nConstraintValue')
        constraint_defs.append((d, value))
    temperature_K = propdef.get('temperature_K')
    pressure_kPa = propdef.get('pressure_kPa')
    composition_entries: List[Dict[str, Any]] = []
    other_vars: List[str] = []
    full_constraints: List[str] = []
    other_constraints: List[str] = []
    for vv in as_list(point.get('VariableValue')):
        if not isinstance(vv, dict):
            continue
        try:
            n = int(vv.get('nVarNumber'))
        except Exception:
            continue
        d = var_defs.get(n, {'number': n, 'type_key': '', 'label': f'Variable {n}', 'orgnum': None, 'phase': ''})
        value = vv.get('nVarValue')
        cls = classify_condition(clean_text(d.get('label')), clean_text(d.get('type_key')))
        if cls == 'temperature':
            t = temperature_to_k(value, clean_text(d.get('label')))
            if t is not None:
                temperature_K = t
        elif cls == 'pressure':
            p = pressure_to_kpa(value, clean_text(d.get('label')))
            if p is not None:
                pressure_kPa = p
        elif cls == 'composition':
            composition_entries.append({'source': 'variable', 'label': clean_text(d.get('label')), 'type_key': clean_text(d.get('type_key')), 'orgnum': d.get('orgnum'), 'value': value, 'text': condition_text(d, value, cmap)})
        else:
            other_vars.append(condition_text(d, value, cmap))
    for d, value in constraint_defs:
        text = condition_text(d, value, cmap)
        full_constraints.append(text)
        cls = classify_condition(clean_text(d.get('label')), clean_text(d.get('type_key')))
        if cls == 'temperature':
            t = temperature_to_k(value, clean_text(d.get('label')))
            if t is not None and temperature_K is None:
                temperature_K = t
        elif cls == 'pressure':
            p = pressure_to_kpa(value, clean_text(d.get('label')))
            if p is not None and pressure_kPa is None:
                pressure_kPa = p
        elif cls == 'composition':
            composition_entries.append({'source': 'constraint', 'label': clean_text(d.get('label')), 'type_key': clean_text(d.get('type_key')), 'orgnum': d.get('orgnum'), 'value': value, 'text': text})
        if cls not in {'temperature', 'pressure'}:
            other_constraints.append(text)
    return {'temperature_K': temperature_K, 'pressure_kPa': pressure_kPa, 'composition_entries': composition_entries, 'composition_text': join_nonempty((e['text'] for e in composition_entries)), 'other_variables': join_nonempty(other_vars), 'full_constraints': join_nonempty(full_constraints), 'other_constraints': join_nonempty(other_constraints), 'var_defs': var_defs}

def normalize_activity_name(name: str) -> str:
    return re.sub('[^a-z0-9]+', ' ', clean_text(name).casefold()).strip()

def idac_decision(propdef: Dict[str, Any], conditions: Dict[str, Any], system_orgnums: Sequence[int], citation_has_idac: bool=False) -> Tuple[bool, Optional[int], Optional[Dict[str, Any]], str, str]:
    """Final audited IDAC recognition rule.

    Accepted evidence:
      A1) molecular activity coefficient and the target solute composition is 0;
      A2) molecular activity-coefficient property name itself explicitly states
          infinite dilution;
      B)  molecular activity coefficient, no composition metadata in the point/block,
          and the article explicitly states that the reported data are IDAC.

    Explicitly NOT sufficient:
      - eStandardState == 'Infinite dilution solute' by itself;
      - citation text mentioning an infinite-dilution limit when finite composition
        variables are present;
      - mean ionic activity coefficients; relative activity; osmotic coefficients.
    """
    pname = normalize_activity_name(propdef.get('raw_name') or propdef.get('name'))
    if 'activity coefficient' not in pname:
        return (False, None, None, '', '')
    if 'mean ionic activity coefficient' in pname:
        return (False, None, None, '', '')
    target = propdef.get('target_orgnum')
    comp_entries = conditions.get('composition_entries', [])
    zeros = [e for e in comp_entries if is_zero(e.get('value'))]
    positives = [e for e in comp_entries if safe_float(e.get('value')) is not None and safe_float(e.get('value')) > 0]
    solute: Optional[int] = int(target) if target is not None else None
    if solute is None:
        zero_orgnums = list(dict.fromkeys((int(e.get('orgnum')) for e in zeros if e.get('orgnum') is not None)))
        if len(zero_orgnums) == 1:
            solute = zero_orgnums[0]
        else:
            explicit_solvents = [n for n in propdef.get('solvent_orgnums', []) if n in system_orgnums]
            remainder = [n for n in system_orgnums if n not in explicit_solvents]
            if explicit_solvents and len(remainder) == 1:
                solute = int(remainder[0])
    if solute is None:
        return (False, None, None, '', '')
    target_positive = any((e.get('orgnum') == solute and safe_float(e.get('value')) is not None and (safe_float(e.get('value')) > 0) for e in comp_entries))
    if target_positive:
        return (False, None, None, '', '')
    matching_zero = next((e for e in zeros if e.get('orgnum') == solute), None)
    if matching_zero is not None:
        basis = f"Activity coefficient + {matching_zero.get('label') or 'composition'} of target component = 0"
        return (True, solute, matching_zero, basis, 'A_明确零组成')
    prop_explicit_inf = 'infinite dilution' in pname or 'limiting activity coefficient' in pname
    if prop_explicit_inf:
        return (True, solute, None, 'Property name explicitly identifies an infinite-dilution activity coefficient', 'A_性质名明确无限稀释')
    if citation_has_idac and (not comp_entries):
        return (True, solute, None, 'Article explicitly reports IDAC and this property block contains no composition variable/constraint', 'B_文献明确无限稀释')
    return (False, None, None, '', '')

def idac_quality_status(doi: str, blocknum: Any, pointno: int, gamma: Optional[float], temperature_k: Optional[float]) -> Tuple[str, str, str]:
    """Return (status, model_recommendation, note).

    Source anomalies are retained in the IDAC workbook for traceability but are
    not recommended for modeling until checked against the original article.
    """
    notes: List[str] = []
    status = 'OK'
    use = '是'
    if gamma is None or gamma <= 0:
        status = 'SOURCE_ANOMALY'
        use = '否'
        notes.append('活度系数应为正值；ThermoML 原始值为非正数，未自动修正')
    if clean_text(doi).casefold() == '10.1021/je200628n' and clean_text(blocknum) == '28' and (pointno in (1, 2, 3)):
        status = 'SOURCE_ANOMALY'
        use = '否'
        notes.append('已确认 ThermoML 此数据块温度字段异常；保留原值，需回原文核对后再建模')
    return (status, use, '；'.join(notes))

def solvents_for_idac(propdef: Dict[str, Any], system_orgnums: Sequence[int], solute_orgnum: int) -> Tuple[List[int], str]:
    explicit = [n for n in propdef.get('solvent_orgnums', []) if n != solute_orgnum]
    if explicit:
        return (list(dict.fromkeys(explicit)), 'ThermoML explicit Solvent')
    inferred = [n for n in system_orgnums if n != solute_orgnum]
    if inferred:
        return (inferred, 'Inferred from system components excluding solute')
    return ([], '')

def stable_record_id(cit: Dict[str, str], relfile: str, kind: str, blocknum: Any, pointno: int, propno: Any, pv_index: int) -> str:
    base = cit.get('doi') or Path(relfile).stem
    return f'{base}::{kind}{blocknum}::POINT{pointno}::PROP{propno}::PV{pv_index}'

def parse_one_document(doc: Dict[str, Any], relfile: str, counters: Counter, warning_rows: List[List[str]]) -> Iterable[Tuple[Dict[str, Any], Optional[Dict[str, Any]]]]:
    cit = citation_fields(doc)
    citation_has_idac = citation_explicitly_idac(doc)
    cmap = build_compound_map(doc)
    for data_key, kind, kind_cn in (('PureOrMixtureData', 'POMD', '纯物质/混合物数据'), ('ReactionData', 'RXND', '反应数据')):
        blocks = as_list(doc.get(data_key))
        for block_idx, block in enumerate(blocks, start=1):
            if not isinstance(block, dict):
                continue
            counters[f'blocks_{kind}'] += 1
            if kind == 'POMD':
                blocknum = block.get('nPureOrMixtureDataNumber', block_idx)
            else:
                blocknum = block.get('nReactionDataNumber', block_idx)
            system_orgnums, system_names, system_keys, internal_comp_map = system_components(block, kind, cmap)
            phases = block_phases(block, cmap)
            prop_defs: Dict[int, Dict[str, Any]] = {}
            for prop in as_list(block.get('Property')):
                if not isinstance(prop, dict):
                    continue
                pd = property_definition(prop)
                try:
                    n = int(pd.get('nPropNumber'))
                except Exception:
                    continue
                prop_defs[n] = pd
            points = as_list(block.get('NumValues'))
            for pointno, point in enumerate(points, start=1):
                if not isinstance(point, dict):
                    continue
                pvals = as_list(point.get('PropertyValue'))
                for pv_index, pv in enumerate(pvals, start=1):
                    if not isinstance(pv, dict):
                        continue
                    counters['property_values'] += 1
                    try:
                        propno = int(pv.get('nPropNumber'))
                    except Exception:
                        propno = pv.get('nPropNumber', '')
                    propdef = prop_defs.get(propno) if isinstance(propno, int) else None
                    if propdef is None:
                        counters['unresolved_property_number'] += 1
                        warning_rows.append([relfile, kind, clean_text(blocknum), clean_text(pointno), f'PropertyValue nPropNumber={propno} 找不到对应 Property 定义'])
                        propdef = {'nPropNumber': propno, 'group': '', 'name': '', 'raw_name': '', 'unit': '', 'method': '', 'target_orgnum': None, 'solvent_orgnums': [], 'phase': '', 'standard_state': '', 'temperature_K': None, 'pressure_kPa': None, 'uncertainty_meta': {}}
                    conditions = extract_conditions(block, point, cmap, propdef)
                    target = compound(cmap, propdef.get('target_orgnum'))
                    solvents = [compound(cmap, n) for n in propdef.get('solvent_orgnums', [])]
                    solvent_names = join_nonempty((c['name'] for c in solvents))
                    solvent_keys = join_nonempty((c['inchikey'] for c in solvents))
                    property_phase = propdef.get('phase') or phases
                    unc_value, unc_desc = uncertainty_fields(propdef.get('uncertainty_meta', {}), pv)
                    value = property_value_text(pv)
                    rec_id = stable_record_id(cit, relfile, kind, blocknum, pointno, propno, pv_index)
                    master = {'数据点ID': rec_id, 'DOI': cit['doi'], '文献标题': cit['title'], '期刊': cit['journal'], '发表年份': cit['year'], '作者': cit['authors'], '原始JSON文件': relfile, '数据类型': kind_cn, '数据块编号': blocknum, '数据点序号': pointno, '体系组分数': len(system_orgnums), '体系组分': system_names, '体系组分InChIKey': system_keys, '目标组分': target['name'], '目标组分分子式': target['formula'], '目标组分InChI': target['inchi'], '目标组分InChIKey': target['inchikey'], '溶剂': solvent_names, '溶剂InChIKey': solvent_keys, '性质大类': propdef.get('group', ''), '性质名称': propdef.get('name', ''), '性质数值': value, '单位': propdef.get('unit', ''), '实验方法': propdef.get('method', ''), '相态': property_phase, '标准态': propdef.get('standard_state', ''), '温度_K': conditions.get('temperature_K') if conditions.get('temperature_K') is not None else '', '压力_kPa': conditions.get('pressure_kPa') if conditions.get('pressure_kPa') is not None else '', '组成条件': conditions.get('composition_text', ''), '其他实验变量': conditions.get('other_variables', ''), '固定条件': conditions.get('full_constraints', ''), '实验值不确定度': unc_value, '不确定度说明': unc_desc, '实验目的': clean_text(block.get('eExpPurpose') or block.get('sExpPurpose')), '数据录入日期': clean_text(block.get('dateDateAdded')), 'ThermoML编译者': clean_text(block.get('sCompiler')), 'ThermoML贡献者': clean_text(block.get('sContributor')), '化合物内部编号': internal_comp_map, '性质内部编号': propno, '变量内部编号': variable_internal_map(conditions.get('var_defs', {}), cmap)}
                    is_idac, solute_orgnum, dilution_entry, basis, evidence_grade = idac_decision(propdef, conditions, system_orgnums, citation_has_idac=citation_has_idac)
                    idac_row: Optional[Dict[str, Any]] = None
                    if is_idac and solute_orgnum is not None:
                        gamma = safe_float(pv.get('nPropValue'))
                        if gamma is None:
                            counters['idac_non_numeric_skipped'] += 1
                        else:
                            solute = compound(cmap, solute_orgnum)
                            solvent_orgnums, solvent_source = solvents_for_idac(propdef, system_orgnums, solute_orgnum)
                            solvent_compounds = [compound(cmap, n) for n in solvent_orgnums]
                            if not solvent_orgnums:
                                counters['idac_no_solvent_skipped'] += 1
                                warning_rows.append([relfile, kind, clean_text(blocknum), clean_text(pointno), '识别到 IDAC，但无法确定溶剂，已从 IDAC 表跳过（母表仍保留）'])
                            elif len(solvent_orgnums) > 1:
                                counters['idac_mixed_solvent_skipped'] += 1
                            else:
                                other_fixed_parts = []
                                dilution_text = dilution_entry.get('text') if dilution_entry else ''
                                for part in clean_text(conditions.get('full_constraints', '')).split(' | '):
                                    if part and part != dilution_text:
                                        low = part.casefold()
                                        if 'temperature' in low or 'pressure' in low:
                                            continue
                                        other_fixed_parts.append(part)
                                basis_full = basis
                                if solvent_source:
                                    basis_full += f'; solvent: {solvent_source}'
                                q_status, q_use, q_note = idac_quality_status(cit.get('doi', ''), blocknum, pointno, gamma, conditions.get('temperature_K'))
                                idac_row = {'IDAC记录ID': rec_id, 'DOI': cit['doi'], '文献标题': cit['title'], '期刊': cit['journal'], '发表年份': cit['year'], '原始JSON文件': relfile, '数据块编号': blocknum, '溶质': solute['name'], '溶质分子式': solute['formula'], '溶质InChI': solute['inchi'], '溶质InChIKey': solute['inchikey'], '溶剂组分数': len(solvent_orgnums), '溶剂': join_nonempty((c['name'] for c in solvent_compounds), sep=' + '), '溶剂InChI': join_nonempty((c['inchi'] for c in solvent_compounds), sep=' + '), '溶剂InChIKey': join_nonempty((c['inchikey'] for c in solvent_compounds), sep=' + '), '温度_K': conditions.get('temperature_K') if conditions.get('temperature_K') is not None else '', '压力_kPa': conditions.get('pressure_kPa') if conditions.get('pressure_kPa') is not None else '', '无限稀释活度系数': gamma, '活度系数不确定度': unc_value, '实验方法': propdef.get('method', ''), '相态': property_phase, '标准态': propdef.get('standard_state', ''), '无限稀释判定方式': basis_full, '稀释变量': dilution_entry.get('label', '') if dilution_entry else '', '稀释组分': compound(cmap, dilution_entry.get('orgnum'))['name'] if dilution_entry and dilution_entry.get('orgnum') is not None else solute['name'], '稀释值': dilution_entry.get('value', '') if dilution_entry else '', '其他固定条件': join_nonempty(other_fixed_parts), '实验目的': clean_text(block.get('eExpPurpose') or block.get('sExpPurpose')), 'IDAC判定等级': evidence_grade, '数据质量状态': q_status, '建议用于建模': q_use, '质量备注': q_note}
                                counters['idac_rows'] += 1
                                counters[f'idac_evidence_{evidence_grade}'] += 1
                                if q_status == 'SOURCE_ANOMALY':
                                    counters['idac_source_anomaly_rows'] += 1
                                if q_use == '是':
                                    counters['idac_model_ready_rows'] += 1
                    yield (master, idac_row)
