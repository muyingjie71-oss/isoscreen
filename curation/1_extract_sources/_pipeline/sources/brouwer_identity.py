"""Brouwer audited additions, separate from all original source fields."""
from pathlib import Path
from functools import lru_cache
import hashlib, json, re
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
VERSION = 'BROUWER_PENDING_SOLVENT_REVIEW_V7_2026-10-09'
HEADERS = ['溶质补充分子式', '溶质补充InChI', '溶质补充InChIKey', '溶质补充CanonicalSMILES', '溶质结构核对状态', '溶质立体核对', '溶质待核原因', '溶质结构依据', '溶质原始身份位置', '溶剂补充分子式', '溶剂补充InChI', '溶剂补充InChIKey', '溶剂补充CanonicalSMILES', '溶剂结构核对状态', '溶剂立体核对', '溶剂阳离子SMILES', '溶剂阴离子SMILES', '溶剂待核原因', '溶剂结构依据', '溶剂原始身份位置', '既有质量审核状态', '既有质量审核原因', '结构审核版本', '溶剂核实名称', '溶剂身份更正说明', '核实原研究DOI', '原CAS核查结果', '当前单分子模型状态', '模型暂缓或排除原因']
SOURCE_FIELDS = ['IDAC记录ID', 'DOI', '文献标题', '期刊', '发表年份', '原始JSON文件', '数据块编号', '溶质', '溶质分子式', '溶质InChI', '溶质InChIKey', '溶剂组分数', '溶剂', '溶剂InChI', '溶剂InChIKey', '温度_K', '压力_kPa', '无限稀释活度系数', '活度系数不确定度', '实验方法', '相态', '标准态', '无限稀释判定方式', '稀释变量', '稀释组分', '稀释值', '其他固定条件', '实验目的', 'IDAC判定等级', '数据质量状态', '建议用于建模', '质量备注']
NUMERIC = {'发表年份', '溶剂组分数', '温度_K', '压力_kPa', '无限稀释活度系数', '活度系数不确定度', '稀释值'}
_PAYLOAD_CACHE = {}

def row_digest(row):
    values = []
    for field in SOURCE_FIELDS:
        value = row.get(field, '')
        if value is None:
            value = ''
        if field in NUMERIC and value != '':
            value = float(value).hex()
        else:
            value = str(value)
        values.append(value)
    return hashlib.sha256(json.dumps(values, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()

def verified_payload(entry):
    if not entry or not entry.get('approved'):
        return None
    inchi, key, smiles = (entry.get(k, '') for k in ('inchi', 'inchikey', 'smiles'))
    signature = (entry.get('name'), entry.get('formula'), inchi, key, smiles, entry.get('status'), entry.get('evidence'))
    if signature in _PAYLOAD_CACHE:
        return _PAYLOAD_CACHE[signature]
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or not inchi or (not key):
        raise ValueError('Incomplete audited structure: ' + entry.get('name', ''))
    if Chem.InchiToInchiKey(Chem.MolToInchi(mol)) != key or Chem.InchiToInchiKey(inchi) != key:
        raise ValueError('Full InChI / key / physical SMILES disagree: ' + entry['name'])
    formula = rdMolDescriptors.CalcMolFormula(mol)
    if formula != entry['formula']:
        raise ValueError('Supplementary formula differs from structure: ' + entry['name'])
    payload = {'inchi': inchi, 'inchikey': key, 'canonical_smiles': Chem.MolToSmiles(mol, isomericSmiles=True), 'formula': formula, 'mol': mol, 'connectivity_key': key.split('-')[0], 'method': 'AUDITED_BROUWER_SUPPLEMENT', 'confidence': '已核对', 'evidence': entry.get('evidence', ''), 'audit_status': entry.get('status', ''), 'stereo_audit': entry.get('stereo', '')}
    _PAYLOAD_CACHE[signature] = payload
    return payload

def entries_for_row(row, registry=None):
    data = registry or load_registry()
    rid = str(row.get('IDAC记录ID') or '')
    binding = data['record_bindings'].get(rid)
    if binding is None or row_digest(row) != binding['source_row_sha256']:
        raise ValueError('Experiment outside audited source scope: ' + rid)
    solute = data['solutes'][str(row['溶质'])]
    solvent = data['solvents'][str(row['溶剂'])]
    override = data.get('solvent_record_overrides', {}).get(rid)
    if override:
        solvent = override
    if row.get('数据质量状态') == 'REVIEW_CORRECTION_SCOPE' and (not binding.get('identity_scope_verified')):
        solvent = dict(solvent, approved=False, status='实验溶剂归属待核', reason='来源表已整体更正溶剂，但此溶质不在逐点核实清单中，不能确认该实验的溶剂归属。需回原研究核对。')
    if binding.get('solvent_context_reason'):
        solvent = dict(solvent, approved=False, status='原始身份信息冲突，待核', reason=binding['solvent_context_reason'])
    return (solute, solvent, binding)

def supplements_for_row(row, registry=None):
    solute, solvent, binding = entries_for_row(row, registry)
    out = {h: '' for h in HEADERS}
    for prefix, entry in [('溶质', solute), ('溶剂', solvent)]:
        out[prefix + '结构核对状态'] = entry.get('status', '待核')
        out[prefix + '立体核对'] = entry.get('stereo', '')
        out[prefix + '待核原因'] = '' if entry.get('approved') else entry.get('reason', '身份证据不足，待核')
        out[prefix + '结构依据'] = entry.get('evidence', '')
        out[prefix + '原始身份位置'] = binding[prefix + '原始身份位置']
        if entry.get('approved'):
            for suffix, key in [('补充分子式', 'formula'), ('补充InChI', 'inchi'), ('补充InChIKey', 'inchikey'), ('补充CanonicalSMILES', 'smiles')]:
                out[prefix + suffix] = entry[key]
    if solvent.get('approved'):
        out['溶剂阳离子SMILES'] = solvent.get('cation_smiles', '')
        out['溶剂阴离子SMILES'] = solvent.get('anion_smiles', '')
    flag = (registry or load_registry()).get('quality_flags', {}).get(str(row['IDAC记录ID']), {})
    out['既有质量审核状态'] = flag.get('status_zh', flag.get('status', ''))
    out['既有质量审核原因'] = flag.get('reason', '')
    out['结构审核版本'] = VERSION
    out['溶剂核实名称'] = solvent.get('confirmed_name', '') if solvent.get('approved') else ''
    out['溶剂身份更正说明'] = solvent.get('identity_correction', '') or solvent.get('metadata_review', '')
    out['核实原研究DOI'] = binding.get('Confirmed_Study_DOI', solvent.get('Confirmed_Study_DOI', ''))
    out['原CAS核查结果'] = solvent.get('cas_status', '原CAS保持来源报告；本轮未独立核验该登记号。')
    if flag.get('status', '').startswith('EXCLUDE'):
        status = '审计保留，模型排除'
        reason = flag.get('reason', '')
    elif solvent.get('decision') == 'CONFIRMED_NON_SINGLE_STRUCTURE':
        status = '非单一结构，当前模型不适用'
        reason = solvent.get('reason', '')
    elif not solvent.get('approved'):
        status = '实验保留，溶剂结构待核'
        reason = solvent.get('reason', '')
    elif not solute.get('approved'):
        status = '实验保留，溶质结构待核'
        reason = solute.get('reason', '')
    else:
        status = '结构已核实，进入下游质量筛选'
        reason = '完整结构已核实；是否进入最终模型由去重、来源和质量筛选共同判定。'
    out['当前单分子模型状态'] = status
    out['模型暂缓或排除原因'] = reason
    return out
