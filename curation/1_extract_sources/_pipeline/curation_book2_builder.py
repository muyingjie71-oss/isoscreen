"""Scientific views for the second and final curation workbook.

This module only describes reviewed values and reader rows. It does not select
duplicates, change source workbooks, fit experiments, or choose family folds.
"""
from __future__ import annotations
from collections import Counter, defaultdict
from pathlib import Path
import csv, hashlib, json, math, re
from urllib.parse import urlsplit, unquote
from .curation_contract import load_locked_curation_assets, save_json

VERSION = 'CURATION_TWO_BOOKS_BOOK2_V1'
NAME = '02_EXPERIMENTS_PAIRS_ISOMERS'
ID_FIELDS = ['canonical_record_id', 'Member_ID', 'Study_ID', 'Solute_InChI',
             'Solvent_InChI', 'Solute_InChIKey', 'Solvent_InChIKey',
             'Solute_SMILES', 'Solvent_SMILES', 'Solvent_ID',
             'source_count', 'source_list', 'original_record_ids',
             'Original_Source_File', 'Original_Source_Location',
             'Original_Data_Target', 'Matched_Conditions', 'Source03_Gamma_XML',
             'Source03_T_XML', 'selected_input_record_id', 'alias_parent_record_ids',
             'Source03_Record_ID', 'original_record_id', 'Original_DOI', 'Reference',
             'Solute_Identity_Evidence', 'Solvent_Identity_Evidence',
             'Solute_Stereochemistry_Evidence', 'Solvent_Stereochemistry_Evidence',
             'Solvent_Cation_SMILES', 'Solvent_Anion_SMILES',
             'QC_Status', 'Model_Eligible', 'Model_Eligibility_Reason']


def load_rows(path, fields=None):
    csv.field_size_limit(40 * 1024 * 1024)
    path = Path(path)
    if path.suffix.lower() in ('.jsonl', '.ndjson'):
        with path.open(encoding='utf8') as f:
            rows = [json.loads(s) for s in f if s.strip()]
            return rows if fields is None else [{k: v for k, v in r.items() if k in fields} for r in rows]
    with path.open(encoding='utf-8-sig', newline='') as f:
        rdr = csv.DictReader(f)
        return list(rdr) if fields is None else [{k: v for k, v in r.items() if k in fields} for r in rdr]


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def col(label, width=18, kind='text', hidden=False):
    return {'label': label, 'key': label, 'width': width, 'kind': kind, 'hidden': hidden}


def cn(n):
    s = ''
    while n:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s


def native(value):
    if value in ('', None):
        return ''
    if isinstance(value, dict) and 'n' in value:
        return value
    token = str(value)
    if not math.isfinite(float(token)):
        raise ValueError('Non-finite experimental numeric token')
    return {'n': token}


def get(r, *keys, default=''):
    for k in keys:
        if r.get(k) not in ('', None):
            return r[k]
    return default


def original_link(r, suffix=''):
    target = get(r, 'Original_Data_Target' + suffix,
                 'Original_Link' + suffix, 'Original_Source_Link' + suffix,
                 'Original_Source_File' + suffix, 'Original_File' + suffix)
    if not target or '00_source' not in str(target).lower():
        raise ValueError('Experiment link must point directly to 00_Source')
    location = get(r, 'Original_Source_Location' + suffix,
                   'Original_Location' + suffix)
    if str(target).lower().startswith('file:'):
        parts = urlsplit(target)
        path = unquote(parts.path, encoding='utf8', errors='strict')
        if len(path) >= 4 and path[0] == '/' and path[2:4] == ':/':
            path = path[1:]
        if parts.netloc and parts.netloc.lower() != 'localhost':
            path = '//' + parts.netloc + path
        target = path.replace('/', '\\')
        location = unquote(parts.fragment) or location
    elif '#' in str(target):
        target, fragment = str(target).rsplit('#', 1)
        location = fragment or location
    return {'label': '源数据', 'target': target, 'location': location}


def measured(r, field):
    key = 'Source03_Gamma_XML' if field == 'gamma_inf' else 'Source03_T_XML'
    return native(r.get(key) or r[field])


def identity_value(r, key):
    if key == 'Solute_InChIKey':
        return get(r, key, 'Solute_Identity')
    if key == 'Solvent_InChIKey':
        return get(r, key, 'Solvent_Identity')
    if key == 'Solvent_ID':
        return get(r, key, 'Solvent_Mol_ID', 'Solvent_Audit_ID')
    return r.get(key, '')


def reason(r):
    return get(r, 'Model_Eligibility_Reason', 'Strict_Model_Reason_CN',
               'QC_Reason', 'Reason')


def qc(r):
    return get(r, 'QC_Status', 'Temperature_QC_Status', default='PASS')


def qc_display(r):
    return {'PASS': '通过', 'REVIEW': '温度趋势需复核',
            'REVIEW_HIGH': '温度严重偏离需复核',
            'NOT_EVALUATED': '无可评估温度系列',
            'EXCLUDE_QC': '温度严重异常，不入模型'}.get(qc(r), qc(r))


def study_display(r):
    doi = str(get(r, 'Original_DOI', 'DOI')).strip()
    if doi:
        for prefix in ('https://doi.org/', 'http://doi.org/',
                       'https://dx.doi.org/', 'http://dx.doi.org/', 'doi:'):
            if doi.lower().startswith(prefix):
                doi = doi[len(prefix):].strip()
                break
        return doi
    reference = str(get(r, 'Reference', 'Source_Original_Reference')).strip()
    if reference and not reference.startswith('BIB:'):
        first = re.split(r'[,;，；]', reference, maxsplit=1)[0].strip()
        year = re.search(r'[（(]\s*((?:18|19|20)\d{2}[a-z]?)\s*[)）]', reference, re.I)
        first = re.sub(r'[（(]\s*(?:18|19|20)\d{2}[a-z]?\s*[)）]', '', first,
                       flags=re.I).strip(' .:：')
        title_words = r'activity|dilution|coefficient|thermodynam|solubility|densit|measurement|prediction|活度|稀释|热力学|溶解|测定|预测'
        if (not first or len(first) > 30 or re.search(title_words, first, re.I)
                or (not re.search(r'[,;，；]', reference) and year is None)):
            return '原文引用已记录'
        return first + ('（' + year.group(1) + '）' if year else '')
    value = reference or str(get(r, 'Study_Label', 'Original_Study', 'Study_ID')).strip()
    if value.startswith('BIB:'):
        return '原文引用已记录'
    return value or '原研究未确认'


def is_model(r, model_ids):
    return r['canonical_record_id'] in model_ids


def method(r):
    explicit = get(r, 'Method', 'Experimental_Method', '实验方法')
    if explicit:
        return explicit
    cond = r.get('Matched_Conditions')
    if cond:
        values = json.loads(cond) if isinstance(cond, str) else cond
        if len(values) > 3:
            return values[3] or '未报告'
    return '未报告'


class Sheet:
    """Complete physical rows, with optional reader roles and row grouping."""
    def __init__(self, directory, name, columns, *, reader=False):
        self.path = Path(directory) / (name + '.jsonl')
        self.f = self.path.open('w', encoding='utf8', newline='')
        self.input_columns = columns
        self.order = sorted(range(len(columns)), key=lambda i: bool(columns[i]['hidden']))
        self.columns = [columns[i] for i in self.order]
        self.spec = {'name': name, 'file': str(self.path.resolve()),
                     'columns': self.columns, 'rows': 0, 'freeze': 1,
                     'header_rows': 0 if reader else 1, 'reader': reader,
                     'physical_rows': True, 'freeze_rows': 0 if reader else 1,
                     'samples': []}
        self.count = 0
        if not reader:
            self.add([c['label'] for c in columns], role='header')

    def add(self, values, role='data', *, outline=0, merged=False, height=None, status=''):
        values = list(values)
        if len(values) > len(self.columns):
            raise ValueError('More values than declared columns: ' + self.spec['name'])
        values.extend([''] * (len(self.columns) - len(values)))
        values = [values[i] for i in self.order]
        for v in values:
            if isinstance(v, str) and len(v) > 32767:
                raise ValueError('Source history exceeds Excel cell length')
        self.count += 1
        if self.count > 1048576:
            raise ValueError('Excel row limit exceeded')
        row = {'values': values, 'role': role}
        if status:
            row['row_status'] = status
        if role == 'data' and height is None:
            # Long explicit member lists and classification evidence must remain
            # legible; short experiment rows keep the author's compact defaults.
            lines = 1
            for v, c in zip(values, self.columns):
                if c['hidden'] or c['kind'] != 'text' or not isinstance(v, str):
                    continue
                lines = max(lines, sum(max(1, math.ceil(sum(
                    1.8 if ord(ch) > 127 else .85 for ch in line) /
                    max(8, c['width']))) for line in v.splitlines() or ['']))
            if lines > 7:
                height = min(409, lines * 13 + 5)
        if outline or height:
            row['row_attrs'] = {}
            if outline:
                row['row_attrs']['outlineLevel'] = str(outline)
            if height:
                row['row_attrs'].update(ht=str(height), customHeight='1')
        if merged:
            end = max(i for i, c in enumerate(self.columns, 1) if not c['hidden'])
            row['merges'] = [f'A{self.count}:{cn(end)}{self.count}']
        self.f.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')
        if len(self.spec['samples']) < 8:
            self.spec['samples'].append(row)
        return self.count

    def finish(self):
        self.f.close()
        with self.path.open('rb') as stream:
            self.spec['file_sha256'] = hashlib.file_digest(stream, 'sha256').hexdigest()
        self.spec['rows'] = self.count
        self.spec['data_rows'] = self.count - (0 if self.spec['reader'] else 1)
        self.spec['auto_filter'] = not self.spec['reader']
        self.spec['autofilter'] = not self.spec['reader']
        return self.spec

    def letter(self, label):
        return cn(next(i for i, c in enumerate(self.columns, 1) if c['label'] == label))


def relation_rows(catalog):
    if isinstance(catalog, list):
        rows, aliases = catalog, {}
    else:
        rows, aliases = catalog['rows'], catalog.get('field_aliases', {})
    alias = {'Family_ID': '异构体家族ID', 'Formula': '分子式',
             'Member_ID_A': '连接结构键A', 'Member_ID_B': '连接结构键B',
             'Isomer_A': '成员A', 'Isomer_B': '成员B',
             'Isomer_Relation': '主异构关系', 'Secondary_Relation': '伴随结构差异',
            'Basis': '分类依据', 'Classification_Origin': '分类产生方式',
             'Source_Relation_Locations': '来源关系表位置'}
    alias.update(aliases)
    result = []
    for row in rows:
        r = dict(row)
        for key, chinese in alias.items():
            if key not in r:
                r[key] = row.get(chinese, '')
        if not r.get('Classification_Origin'):
            r['Classification_Origin'] = row.get('分类来源', '')
        result.append(r)
    return result


def build_book2_descriptor(bank, directory, *, asset_dir=None):
    """bank paths: canonical, model_task1, model_task2, relations_catalog, report.

    Every canonical input point is displayed, including explicit QC/model holds.
    Only supplied model-qualified pairs feed matched views. No new pair or fold
    is generated here.
    """
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    snapshots = {key: (Path(path).stat().st_size, Path(path).stat().st_mtime_ns)
                 for key, path in bank.items()}
    canonical_fields = set(ID_FIELDS) | {'Family_ID', 'Solute', 'Solvent', 'T_K',
        'gamma_inf', 'Source', 'QC_Status', 'Temperature_QC_Status', 'QC_Reason',
        'Model_Eligibility_Reason', 'Strict_Model_Reason_CN', 'Reason', 'Method',
        'Experimental_Method', '实验方法', 'Study_Label', 'Original_Study', 'DOI',
        'Solute_Identity', 'Solvent_Identity', 'Solvent_Audit_ID', 'Solvent_Mol_ID',
        'Original_Source_Link', 'Solute_Formula', 'Solvent_Formula', 'Source_Original_Reference'}
    rows = load_rows(bank['canonical'], canonical_fields)
    model = load_rows(bank['model_task1'], {'canonical_record_id', 'Family_ID'})
    pairs = load_rows(bank['model_task2'])
    catalog = load_json(bank['relations_catalog'])
    report = load_json(bank['report'])
    if report.get('status') != 'PASS':
        raise ValueError('Scientific bank report did not pass')
    if (len(rows), len(model), len(pairs)) != (
            report['experimental_master_task1'], report['task1'], report['task2_pairs']):
        raise ValueError('Scientific data files and preparation report are inconsistent')
    locked = load_locked_curation_assets(asset_dir)
    assignment = locked['outer']['assignment']
    rounds = locked['complete']['rounds']
    ids = {r['canonical_record_id']: r for r in rows}
    model_ids = {r['canonical_record_id'] for r in model}
    if len(ids) != len(rows) or len(model_ids) != len(model):
        raise ValueError('Canonical/model record IDs must be unique')
    if not model_ids <= set(ids):
        raise ValueError('Model Task1 point absent from canonical experiment bank')
    if not {r['Family_ID'] for r in model} <= set(assignment):
        raise ValueError('Model experiment family is absent from fixed split')
    pair_ids = set()
    groups = defaultdict(list)
    fam = defaultdict(lambda: defaultdict(list))
    metrics = defaultdict(lambda: {'points': 0, 'model_points': 0,
                                  'groups': set(), 'pairs': 0, 'types': Counter()})
    for r in rows:
        fam[r['Family_ID']][r['Member_ID']].append(r)
        metrics[r['Family_ID']]['points'] += 1
        metrics[r['Family_ID']]['model_points'] += int(is_model(r, model_ids))
    for p in pairs:
        pid = p.get('Pair_ID') or p.get('pair_id')
        if not pid or pid in pair_ids:
            raise ValueError('Missing/duplicate stable Pair_ID')
        pair_ids.add(pid)
        a, b = p['canonical_record_id_A'], p['canonical_record_id_B']
        if a not in model_ids or b not in model_ids:
            raise ValueError('Pair endpoint is not an approved model experiment')
        ra, rb = ids[a], ids[b]
        for key in ('Family_ID', 'Study_ID'):
            if ra.get(key) != rb.get(key):
                raise ValueError('Pair endpoint mismatch: ' + key)
        if float(ra['T_K']) != float(rb['T_K']):
            raise ValueError('Pair endpoint mismatch: temperature')
        solvent_a = get(ra, 'Solvent_Audit_ID', 'Solvent_Identity', 'Solvent_ID')
        solvent_b = get(rb, 'Solvent_Audit_ID', 'Solvent_Identity', 'Solvent_ID')
        if solvent_a != solvent_b:
            raise ValueError('Pair endpoint mismatch: solvent identity')
        if p['Family_ID'] != ra['Family_ID']:
            raise ValueError('Pair family differs from endpoint family')
        da = math.log(float(ra['gamma_inf'])) - math.log(float(rb['gamma_inf']))
        if not math.isclose(da, float(p['D_AB']), abs_tol=2e-12):
            raise ValueError('Pair target differs from original experimental values')
        groups[p['MatchedGroup_ID']].append(p)
        m = metrics[p['Family_ID']]
        m['groups'].add(p['MatchedGroup_ID']); m['pairs'] += 1
        m['types'][p['Isomer_Relation']] += 1
    members = {r['Member_ID'] for r in rows}
    relations = []
    seen = set()
    for r in relation_rows(catalog):
        if r['Member_ID_A'] not in members or r['Member_ID_B'] not in members:
            continue
        key = tuple(sorted((r['Member_ID_A'], r['Member_ID_B'])))
        if key in seen:
            continue
        seen.add(key); relations.append(r)
    result = []

    # The root common author supplies the requested compact flow figure.
    flow = [
        ['输入实验', f'来自01_MERGED_DEDUPLICATED保留的{len(rows):,}个canonical实验点。重复副本、冲突和来源历史保存在第一工作簿。'],
        ['温度QC', '同研究、同溶质、同溶剂、同方法的多温度系列做lnγ∞对1/T一致性检查。严重孤立数量级异常仅取消模型资格，原实验值不改。'],
        ['模型单点', f'{len(model):,}个点通过来源、温度QC及完整InChI身份资格；另{len(rows)-len(model):,}个点在全部实验中保留状态和具体原因。'],
        ['配对条件', '同原研究、同溶剂、同温度、同family，且方法、压力、相态、标准态等条件唯一相容。'],
        ['配对结果', f'{len(pairs):,}对，{len(groups):,}个matched group；两端都来自通过模型资格的真实实验。'],
        ['构造分类', f'{len(fam)}个当前实验family，{len(members)}个构造成员，{len(relations)}条已审核分子异构关系。'],
        ['固定划分', '原84个family的Fold_ID及五轮train/validation/test成员清单完整迁入本簿。同family两项任务不拆分，只重算当前数量。'],
        ['原始数值', '实测γ∞和温度保持原始值。lnγ∞和D_AB由实测值计算，不替换为温度趋势预测值。']]
    # Family -> member -> actual experiments. It is a reader, not a matched
    # group registry, and includes every canonical experiment exactly once.
    read_cols = [col('溶剂', 25), col('温度/K', 10, 'number'),
                 col('实测γ∞', 13, 'gamma'), col('原研究', 17),
                 col('实验方法', 23, hidden=True), col('来源', 11), col('原始数据', 11, 'link'),
                 col('QC状态', 17), col('模型资格', 9), col('未进模型原因', 32),
                 col('canonical_record_id', 44, hidden=True),
                 col('Member_ID', 40, hidden=True), col('Family_ID', 23, hidden=True),
                 col('Fold_ID', 17, hidden=True)]
    reader_extra = [k for k in ID_FIELDS if k not in
                    {'canonical_record_id', 'Member_ID', 'Family_ID', 'Fold_ID'}]
    read_cols += [col(k, 44, hidden=True) for k in reader_extra]
    read_cols += [col('原始溶质名称', 32, hidden=True)]
    reader = Sheet(directory, '分组阅读', read_cols, reader=True)
    directory_rows = []
    for family, member_bank in sorted(fam.items()):
        m = metrics[family]
        member_names = sorted({get(rr[0], 'Solute', '溶质') for rr in member_bank.values()})
        formula = family.removeprefix('ISO_')
        header = reader.add([f'{family}  {formula}  {len(member_bank)}个成员  '
                             f'{m["points"]:,}条实验  {assignment.get(family,"未入固定划分")}'],
                            role='family', merged=True, height=29)
        totals = Counter()
        for kind, n in m['types'].items():
            cat = '位置异构pair' if '位置' in kind else '骨架异构pair' if '骨架' in kind else '官能团异构pair' if '官能团' in kind else '其他pair'
            totals[cat] += n
        directory_rows.append([family, formula, '\n'.join(member_names),
                               len(member_bank), m['points'], m['model_points'],
                               len(m['groups']), m['pairs'],
                               totals['位置异构pair'], totals['骨架异构pair'],
                               totals['官能团异构pair'], totals['其他pair'],
                               assignment.get(family, '未入固定划分'),
                               {'label': '查看实验', 'target': '', 'location': f"'分组阅读'!A{header}"}])
        for mid, rr in sorted(member_bank.items(), key=lambda kv: (kv[1][0]['Solute'], kv[0])):
            reader.add([f'{rr[0]["Solute"]}  ·  {len(rr):,}条实验'],
                       role='member', outline=1, merged=True, height=26)
            reader.add([c['label'] for c in read_cols], role='header', outline=1, height=26)
            for r in sorted(rr, key=lambda x: (x['Solvent'], x['Study_ID'], float(x['T_K']), x['canonical_record_id'])):
                eligible = is_model(r, model_ids)
                reader.add([r['Solvent'], measured(r, 'T_K'), measured(r, 'gamma_inf'),
                            study_display(r),
                            method(r), r['Source'], original_link(r), qc_display(r),
                            '可建模' if eligible else '暂不建模', '' if eligible else reason(r),
                            r['canonical_record_id'], mid, family, assignment.get(family, ''),
                            *[identity_value(r, k) for k in reader_extra], r['Solute']],
                           outline=2, status='' if eligible else 'HOLD_MODEL')
        reader.add([], role='blank', height=10)
    reader_spec = reader.finish()
    c = [col('Family_ID', 18), col('分子式', 12), col('成员', 32, hidden=True),
         col('成员数', 9, 'integer'), col('实验单点', 13, 'integer'),
         col('模型单点', 13, 'integer'), col('matched group', 13, 'integer'),
         col('pair', 11, 'integer'), col('位置异构pair', 16, 'integer', True),
         col('骨架异构pair', 16, 'integer', True), col('官能团异构pair', 17, 'integer', True),
         col('其他pair', 13, 'integer', True), col('Fold_ID', 13), col('查看实验', 11, 'link')]
    s = Sheet(directory, '家族汇总', c)
    for row in directory_rows:
        s.add(row)
    result.extend([s.finish(), reader_spec])

    c = [col('Family_ID', 21, hidden=True), col('溶质', 22), col('溶剂', 25),
         col('温度/K', 11, 'number'), col('实测γ∞', 15, 'gamma'),
         col('lnγ∞', 14, 'log', True), col('来源', 12, hidden=True),
         col('原始数据', 12, 'link'), col('QC状态', 17, hidden=True),
         col('模型资格', 13), col('未进模型原因', 34), col('Fold_ID', 15, hidden=True)]
    c += [col(k, 44, hidden=True) for k in ID_FIELDS]
    s = Sheet(directory, '全部实验', c)
    for r in sorted(rows, key=lambda r: (r['Family_ID'], r['Member_ID'], r['Solvent'], r['Study_ID'], float(r['T_K']), r['canonical_record_id'])):
        i = s.count + 1
        eligible = is_model(r, model_ids)
        s.add([r['Family_ID'], r['Solute'], r['Solvent'], measured(r, 'T_K'),
               measured(r, 'gamma_inf'), {'f': f'LN({s.letter("实测γ∞")}{i})', 'v': math.log(float(r['gamma_inf']))},
               r['Source'], original_link(r), qc_display(r), '可建模' if eligible else '暂不建模',
               '' if eligible else reason(r), assignment.get(r['Family_ID'], ''),
               *[identity_value(r, k) for k in ID_FIELDS]],
              status='' if eligible else 'HOLD_MODEL')
    result.append(s.finish())

    c = [col('MatchedGroup_ID', 27, hidden=True), col('Family_ID', 20, hidden=True), col('溶剂', 22),
         col('温度/K', 10, 'number'), col('异构体A', 20), col('异构体B', 20),
         col('γ∞A', 11, 'gamma'), col('γ∞B', 11, 'gamma'), col('D_AB', 11, 'log'),
         col('异构关系', 16), col('实验方法', 23, hidden=True), col('原始数据A', 10, 'link'),
         col('原始数据B', 10, 'link'), col('Fold_ID', 15, hidden=True)]
    hidden = ['Pair_ID', 'Study_ID', 'canonical_record_id_A', 'canonical_record_id_B',
              'ln_gamma_A', 'ln_gamma_B', 'Matched_Conditions', 'Member_ID_A',
              'Member_ID_B', 'Original_Link_A', 'Original_Link_B']
    c += [col(k, 44, hidden=True) for k in hidden]
    s = Sheet(directory, '异构体配对', c)
    for p in sorted(pairs, key=lambda r: (r['Family_ID'], r['MatchedGroup_ID'], r['Pair_ID'])):
        i = s.count + 1
        r_a, r_b = ids[p['canonical_record_id_A']], ids[p['canonical_record_id_B']]
        s.add([p['MatchedGroup_ID'], p['Family_ID'], p['Solvent'], native(p['T_K']),
               p['Isomer_A'], p['Isomer_B'], measured(r_a, 'gamma_inf'), measured(r_b, 'gamma_inf'),
               {'f': f'LN({s.letter("γ∞A")}{i})-LN({s.letter("γ∞B")}{i})', 'v': float(p['D_AB'])},
               p['Isomer_Relation'], method(p), original_link(r_a), original_link(r_b),
               assignment[p['Family_ID']], *[p.get(k, '') for k in hidden]])
    result.append(s.finish())

    c = [col('MatchedGroup_ID', 23), col('Family_ID', 21, hidden=True), col('溶剂', 26),
         col('温度/K', 10, 'number'), col('实验方法', 18), col('成员数', 10, 'integer'),
         col('pair数', 10, 'integer'), col('成员', 35), col('原研究', 35, hidden=True), col('Fold_ID', 15, hidden=True),
         col('Matched_Conditions', 55, hidden=True), col('canonical_record_ids', 55, hidden=True)]
    s = Sheet(directory, '配对组', c)
    for gid, pp in sorted(groups.items()):
        first = pp[0]
        point_ids = sorted({p[k] for p in pp for k in ('canonical_record_id_A', 'canonical_record_id_B')})
        s.add([gid, first['Family_ID'], first['Solvent'], native(first['T_K']),
               method(first), len(point_ids), len(pp),
               '\n'.join(ids[i]['Solute'] for i in point_ids), first['Study_ID'],
               assignment[first['Family_ID']], first['Matched_Conditions'], '; '.join(point_ids)])
    result.append(s.finish())

    c = [col('Family_ID', 18), col('异构体A', 23), col('异构体B', 23),
         col('主异构关系', 20), col('细分关系', 22), col('分类依据', 39), col('关系来源', 45, hidden=True)]
    rel_hidden = ['Member_ID_A', 'Member_ID_B', 'Formula', 'SMILES_A', 'SMILES_B', 'Source_Relation_Locations']
    c += [col(k, 48, hidden=True) for k in rel_hidden]
    s = Sheet(directory, '异构关系', c)
    for r in sorted(relations, key=lambda r: (r['Family_ID'], r['Isomer_A'], r['Isomer_B'])):
        s.add([r['Family_ID'], r['Isomer_A'], r['Isomer_B'], r['Isomer_Relation'],
               r.get('Secondary_Relation', ''), r.get('Basis', ''),
               get(r, 'Classification_Origin', 'Source_Relation_Locations'),
               *[r.get(k, '') for k in rel_hidden]])
    result.append(s.finish())

    c = [col('轮次', 10, 'integer'), col('用途', 12), col('固定Family数', 18, 'integer'),
         col('当前有模型点Family数', 24, 'integer'), col('实验单点', 17, 'integer'),
         col('模型Task1单点', 19, 'integer'), col('Matched group', 20, 'integer'),
         col('Task2 pair', 18, 'integer')]
    s = Sheet(directory, '每轮汇总', c)
    roles = [('train', '训练'), ('validation', '验证'), ('test', '测试')]
    fold_counts = []
    for rid, values in sorted(rounds.items(), key=lambda kv: int(kv[0])):
        for role, label in roles:
            families = values[role + '_families']
            vals = [int(rid), label, len(families), sum(metrics[f]['model_points'] > 0 for f in families),
                    sum(metrics[f]['points'] for f in families),
                    sum(metrics[f]['model_points'] for f in families),
                    sum(len(metrics[f]['groups']) for f in families),
                    sum(metrics[f]['pairs'] for f in families)]
            s.add(vals); fold_counts.append(vals)
    result.append(s.finish())
    c = [col('轮次', 10, 'integer'), col('用途', 12), col('Family_ID', 23), col('Fold_ID', 16),
         col('实验单点', 17, 'integer'), col('模型Task1单点', 19, 'integer'),
         col('Matched group', 20, 'integer'), col('Task2 pair', 18, 'integer')]
    s = Sheet(directory, 'Family清单', c)
    for rid, values in sorted(rounds.items(), key=lambda kv: int(kv[0])):
        for role, label in roles:
            for f in sorted(values[role + '_families']):
                m = metrics[f]
                s.add([int(rid), label, f, assignment[f], m['points'], m['model_points'], len(m['groups']), m['pairs']])
    result.append(s.finish())
    proof = {'canonical_records': len(rows), 'model_task1': len(model),
             'task2_pairs': len(pairs), 'matched_groups': len(groups),
             'observed_families': len(fam), 'members': len(members),
             'relations': len(relations), 'fixed_families': len(assignment),
             'family_role_rows': 5 * len(assignment), 'fold_counts': fold_counts,
             'input_report': report}
    qc_counts = Counter(qc(r) for r in rows)
    native_flow = {
        'title': '全部实验、异构体配对与分类',
        'origin': f'来自08第一册的{len(rows):,}条去重审计实验；保留待核状态。单点与配对使用同一模型资格门槛。',
        'stages': [
            {'label': '全部审计实验（含暂缓）', 'count': len(rows), 'unit': '个实验点',
             'detail': f'{len(rows):,}条均留“全部实验／分组阅读”，含暂缓。\n原实验值保留；不是全部进入训练。'},
            {'label': '温度QC复查（本轮新增排除数）', 'count': len(rows), 'unit': '个实验点',
             'detail': f'同研究、溶质、溶剂、方法分系列；本轮新增严重异常{qc_counts.get("EXCLUDE_QC",0)}条。不能重复扣除上游已隔离记录。',
             'branch': {'label': '本轮新增QC排除', 'count': qc_counts.get('EXCLUDE_QC', 0), 'unit': '个点',
                        'detail': f'当前七份表三范围内新增{qc_counts.get("EXCLUDE_QC",0)}条；不表示历史原数据从未出现异常。'}},
            {'label': '最终单点训练资格通过', 'count': len(model), 'unit': '个单点',
             'detail': f'{len(rows):,} − {len(rows)-len(model):,}暂缓 = {len(model):,}条。\n原因留全部实验；配对仅取合格端点。',
             'branch': {'label': '暂缓单点及实验配对', 'count': len(rows) - len(model), 'unit': '个点',
                        'detail': f'{report["model_hold_counts"].get("HOLD_MODEL_POTENTIAL_REPEAT",0):,}转载待核；{report["model_hold_counts"].get("HOLD_MODEL_POTENTIAL_SOURCE_ESTIMATE",0):,}实测出处待核。均不用于单点或配对。'}},
            {'label': '判定：同研究／同条件／同家族？', 'count': len(pairs), 'unit': '对',
             'detail': f'同研究、溶剂、温度、家族及唯一相容条件。\n{len(groups):,}组；两端均为合格单点，暂缓点参与0条。'},
            {'label': '分子结构分类（不是实验配对）', 'count': len(fam), 'unit': '个家族',
             'detail': f'{len(fam)}家族、{len(members)}成员、{len(relations)}对分子关系（含审计成员）。\n另有{len(pairs):,}对合格实验配对，两种对数分别统计。'},
            {'label': '固定family划分', 'count': len(assignment), 'unit': '个固定家族',
             'detail': '原五轮train、validation、test成员集合不变。只按当前实验与pair重算数量；同family始终同fold。'}],
        'footer': ''
    }
    native_flow['footer'] = f'全部审计{len(rows):,}条；训练单点{len(model):,}条；matched实验{len(pairs):,}对／{len(groups):,}组。D_AB只由合格实测端点计算；暂缓点不参与。五折按同family整体分配。'
    bundle = {'version': VERSION, 'kind': NAME, 'workbook': NAME + '.xlsx',
              'source_folder': '08_Dataset_Curation', 'source_name': '七来源汇总',
              'title': '实验单点、异构体配对与分类',
              'flow': native_flow,
              'sheets': result, 'statistics': proof,
              'fixed_family_role_signature': locked['signature'],
              'flow_rows': flow, 'source_bank': {k: str(v) for k, v in bank.items()}}
    manifest = directory / 'bundle.json'
    for key, path in bank.items():
        now = (Path(path).stat().st_size, Path(path).stat().st_mtime_ns)
        if now != snapshots[key]:
            raise ValueError('Scientific input changed during descriptor creation: ' + key)
    save_json(manifest, bundle)
    return manifest
