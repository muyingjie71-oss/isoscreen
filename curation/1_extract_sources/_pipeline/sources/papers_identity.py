"""Evidence-scoped solvent additions; original experimental fields stay intact."""
from pathlib import Path
from functools import lru_cache
import json, hashlib
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
VERSION = 'PAPERS_SOLVENT_IDENTITY_V1_2026-10-09'
NUMERIC = {'发表年份', '溶剂组分数', '温度_K', '压力_kPa', '无限稀释活度系数', '稀释值'}

def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def source_digest(row, headers):
    vals = [float(row.get(h)).hex() if h in NUMERIC and row.get(h) not in ('', None) else str(row.get(h) or '') for h in headers]
    return hashlib.sha256(json.dumps(vals, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()

def supplements_for_row(row, registry=None):
    reg = registry or load_registry()
    rid = str(row.get('IDAC记录ID') or '')
    binding = reg['record_bindings'].get(rid)
    if binding is None or source_digest(row, reg['source_fields']) != binding['source_row_sha256']:
        raise ValueError('Record outside audited original scope: ' + rid)
    return dict(binding['supplements'])
