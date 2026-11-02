"""Audited ILThermo additions. Original source structure fields are immutable.

The registry is separate from 00_Source and is bound to original component IDs,
names and formulas. No online resolver or formula-only inference is used here.
"""
from pathlib import Path
from functools import lru_cache
import json, re, html, hashlib
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
VERSION = 'ILTHERMO_IDENTITY_AUDIT_2026-10-09'
_PAYLOAD_CACHE = {}
HEADERS = ['溶质补充InChI', '溶质补充InChIKey', '溶质补充CanonicalSMILES', '溶质结构核对状态', '溶质立体核对', '溶质原始组件ID', '溶质结构依据', '溶剂原始分子式', '溶剂补充InChI', '溶剂补充InChIKey', '溶剂补充CanonicalSMILES', '溶剂结构核对状态', '溶剂立体核对', '溶剂阳离子SMILES', '溶剂阴离子SMILES', '溶剂原始组件ID', '溶剂结构依据', '结构审核版本']

def clean_formula(value):
    return re.sub('\\s+', '', html.unescape(re.sub('<[^>]*>', '', str(value or ''))))

def verified_payload(entry):
    if not entry or not entry.get('approved'):
        return None
    inchi, key, smiles = (entry.get(k, '') for k in ('inchi', 'inchikey', 'smiles'))
    signature = (entry.get('name'), entry.get('formula'), inchi, key, smiles, entry.get('status'), entry.get('evidence'))
    if signature in _PAYLOAD_CACHE:
        return _PAYLOAD_CACHE[signature]
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or not inchi or (not key):
        raise ValueError('Incomplete approved structure: ' + str(entry.get('name')))
    got_inchi = Chem.MolToInchi(mol)
    got_key = Chem.InchiToInchiKey(got_inchi)
    if got_key != key or Chem.InchiToInchiKey(inchi) != key:
        raise ValueError('Full InChI / key / SMILES disagree: ' + str(entry.get('name')))
    formula = rdMolDescriptors.CalcMolFormula(mol)
    if clean_formula(formula) != clean_formula(entry.get('formula')):
        raise ValueError('Source formula differs from approved SMILES: ' + str(entry.get('name')))
    payload = {'inchi': inchi, 'inchikey': key, 'canonical_smiles': Chem.MolToSmiles(mol, isomericSmiles=True), 'formula': formula, 'mol': mol, 'connectivity_key': key.split('-')[0], 'method': 'AUDITED_SUPPLEMENT_INCHI', 'confidence': '高', 'evidence': entry.get('evidence', ''), 'audit_status': entry.get('status', ''), 'stereo_audit': entry.get('stereo', '')}
    _PAYLOAD_CACHE[signature] = payload
    return payload

def entries_for_row(row, registry=None):
    data = registry or load_registry()
    solute = data['solutes'].get(str(row.get('溶质') or ''))
    solvent = data['solvents'].get(str(row.get('溶剂') or ''))
    if solute and clean_formula(solute['formula']) != clean_formula(row.get('溶质分子式')):
        raise ValueError('Audited solute formula binding changed: ' + str(row.get('IDAC记录ID')))
    file = str(row.get('原始JSON文件') or '')
    binding = data.get('file_bindings', {}).get(file)
    if binding is None:
        raise ValueError('Original JSON is outside audited scope: ' + file)
    if binding['solute_name'] != row.get('溶质') or binding['solvent_name'] != row.get('溶剂'):
        raise ValueError('Original component name binding changed: ' + file)
    if solute and binding['solute_id'] not in solute['component_ids']:
        raise ValueError('Solute component ID binding changed: ' + file)
    if solvent and binding['solvent_id'] not in solvent['component_ids']:
        raise ValueError('Solvent component ID binding changed: ' + file)
    return (solute, solvent, binding)

def supplements_for_row(row, registry=None):
    a, b, binding = entries_for_row(row, registry)
    values = {h: '' for h in HEADERS}
    for prefix, entry, component in [('溶质', a, binding['solute_id']), ('溶剂', b, binding['solvent_id'])]:
        values[prefix + '原始组件ID'] = component
        values[prefix + '结构核对状态'] = entry.get('status', 'REVIEW') if entry else 'REVIEW'
        values[prefix + '立体核对'] = entry.get('stereo', '') if entry else '未确认'
        values[prefix + '结构依据'] = entry.get('evidence', '') if entry else '未找到审核映射'
        if entry and entry.get('approved'):
            values[prefix + '补充InChI'] = entry['inchi']
            values[prefix + '补充InChIKey'] = entry['inchikey']
            values[prefix + '补充CanonicalSMILES'] = entry['smiles']
    values['溶剂原始分子式'] = binding['solvent_formula']
    if b and b.get('approved'):
        values['溶剂阳离子SMILES'] = b.get('cation_smiles', '')
        values['溶剂阴离子SMILES'] = b.get('anion_smiles', '')
    values['结构审核版本'] = VERSION
    return values
