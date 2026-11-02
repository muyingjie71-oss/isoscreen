"""Rebuild experimental-only IDAC from the audited derived curation layer.
Original PDF/XML/XLSX sources are under 00_Source. The derived CSV and its manifest
include every original locator and SHA256; neither contains target predictions.
Mixed solvents and redundant/unverified Cyrene journal/average candidates are
excluded. Both actual Cyrene experimental column loadings are retained.
With audited_structure_registry.json present, the default entry preserves the
frozen experimental table and adds the reviewed structures.
The original CSV/source manifest remain read-only. The legacy exporter below
is used only in source folders without that reviewed registry.
"""
from pathlib import Path
from collections import Counter
import csv, hashlib, json, re, sys
ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / '02_IDAC.xlsx'
SCRIPT_VERSION = '2.2.0-reviewed-solvent-supplements'
HEADERS = ['IDAC记录ID', 'DOI', '文献标题', '期刊', '发表年份', '原始JSON文件', '数据块编号', '溶质', '溶质分子式', '溶质InChI', '溶质InChIKey', '溶剂组分数', '溶剂', '溶剂InChI', '溶剂InChIKey', '温度_K', '压力_kPa', '无限稀释活度系数', '活度系数不确定度', '实验方法', '相态', '标准态', '无限稀释判定方式', '稀释变量', '稀释组分', '稀释值', '其他固定条件', '实验目的', 'IDAC判定等级', '数据质量状态', '建议用于建模', '质量备注']

def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def load_rows():
    """Read the audited curation layer and verify its real source files before writing."""
    bank = ROOT / 'verified_experimental_records.csv'
    manifest = json.loads((ROOT / 'verified_sources.json').read_text(encoding='utf-8'))
    if sha256(bank) != manifest['curation_sha256']:
        raise RuntimeError('Audited curation SHA256 mismatch; re-audit edits before rebuilding')
    for relative, expected in manifest['original_sources'].items():
        path = ROOT / relative
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f'Original experimental source changed/missing: {relative}')
    with bank.open('r', encoding='utf-8-sig', newline='') as f:
        data = list(csv.DictReader(f))
    numeric = ['溶剂组分数', '温度_K', '压力_kPa', '无限稀释活度系数', '稀释值', '发表年份']
    for row in data:
        for h in numeric:
            if row[h] != '':
                row[h] = float(row[h])
        if row['溶剂组分数'] != 1 or row['无限稀释活度系数'] <= 0 or row['温度_K'] <= 0:
            raise RuntimeError('Only positive experimental points in a single solvent are eligible')
        if not row['原始JSON文件'].startswith('00_Source/'):
            raise RuntimeError('Missing original source locator')
        if re.search('(?:pred|prediction|square_error)\\s*=', row['其他固定条件'], re.I):
            raise RuntimeError('Prediction metadata is not eligible for export')
    if len(data) != manifest['expected_records'] or len({r['IDAC记录ID'] for r in data}) != len(data):
        raise RuntimeError('Audited count/unique ID validation failed')
    return data
