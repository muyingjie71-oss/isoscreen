"""Union the seven original relation sheets and fill cross-source gaps.

Uses the centrally migrated ThermoML classify_pair_relation policy.
It never equates stereoisomers to constitutional isomers or guesses structures.
Public API: load_relations(project, records) -> dict[(Member_ID, Member_ID), label].
After loading, RELATIONS_ROWS (list of dictionaries), RELATION_HEADERS and
LAST_REPORT can be consumed by workbook/pipeline callers. get_relation_rows()
returns the same row dictionaries. All persisted outputs are beside this module.
"""
from __future__ import annotations
import hashlib
import importlib.util
import itertools
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

HERE = Path(__file__).resolve().parent
RELATION_HEADERS = [
 '异构体家族ID','分子式','成员A','成员B','主异构关系','伴随结构差异','置信度','分类依据',
 '连接结构键A','连接结构键B','成员A_CanonicalSMILES','成员B_CanonicalSMILES',
 '成员A_IDAC记录数','成员B_IDAC记录数','成员A_来源数','成员B_来源数',
 '分类产生方式','来源关系表位置','结构确认来源']
FIELD_ALIASES = {
 'Member_ID_A':'连接结构键A','Member_ID_B':'连接结构键B','Formula':'分子式',
 'Family_ID':'异构体家族ID','Isomer_A':'成员A','Isomer_B':'成员B',
 'Isomer_Relation':'主异构关系','Secondary_Relation':'伴随结构差异',
 'Confidence':'置信度','Basis':'分类依据',
 'SMILES_A':'成员A_CanonicalSMILES','SMILES_B':'成员B_CanonicalSMILES',
 'N_Task1_A':'成员A_IDAC记录数','N_Task1_B':'成员B_IDAC记录数',
 'N_Source_A':'成员A_来源数','N_Source_B':'成员B_来源数',
 'Classification_Origin':'分类产生方式','Source_Relation_Locations':'来源关系表位置',
 'Structure_Evidence':'结构确认来源'}
RELATIONS_ROWS = []
relations_rows = RELATIONS_ROWS
LAST_REPORT = {}

def _import_at(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

def _pair(a,b):
    return tuple(sorted((str(a).strip(),str(b).strip())))

def _member(mol):
    key = Chem.MolToInchiKey(mol)
    if not key:
        raise ValueError('Unable to calculate full solute InChIKey')
    return key[:14]

def _structure(smiles, expected, location):
    mol = Chem.MolFromSmiles(str(smiles)) if smiles else None
    if mol is None:
        raise ValueError(f'Invalid or missing confirmed solute SMILES: {expected}; {location}')
    got = _member(mol)
    if got != expected:
        raise ValueError(f'Solute connectivity mismatch: {expected} != {got}; {location}')
    return {'mol':mol,'formula':rdMolDescriptors.CalcMolFormula(mol),
            'connectivity_smiles':Chem.MolToSmiles(mol,isomericSmiles=False),
            'smiles':Chem.MolToSmiles(mol,isomericSmiles=True)}

def _validate_constitutional(a,b,key):
    if a['formula'] != b['formula']:
        raise ValueError(f'Constitutional relation requires equal formulas: {key}: {a["formula"]} / {b["formula"]}')
    if key[0] == key[1] or a['connectivity_smiles'] == b['connectivity_smiles']:
        raise ValueError(f'Identical connectivity / stereoisomer-only pair is not constitutional: {key}')

def _sources(r):
    v = r.get('source_list', '')
    if isinstance(v,list):
        result = v
    elif isinstance(v,str) and v.startswith('['):
        try: result = json.loads(v)
        except json.JSONDecodeError: result = re.split('[;,]',v)
    else:
        result = re.split('[;,]',str(v))
    return {str(x).strip() for x in result+[r.get('Source','')] if str(x).strip()}

def _name(names, default):
    return sorted(names, key=lambda x:(-names[x],x))[0] if names else default

def get_relation_rows():
    return RELATIONS_ROWS

def get_report():
    return LAST_REPORT

def load_relations(project, records):
    project = Path(project)
    records = list(records)
    old_paths = list(sys.path)
    try:
        source_reader = _import_at(HERE/'source_overlap_audit.py','_relations_v2_source_reader')
        sys.path.insert(0,str(project))
        from _pipeline.science import classification_for_source
        classifier = classification_for_source('01_ThermoML')
        classifier_path = project/'_pipeline/sources/thermoml_classification_policy.py'
    finally:
        sys.path[:] = old_paths

    originals = defaultdict(list)
    structure_bank = {}
    sources_report = []
    source_rows = 0
    for folder in sorted(project.glob('0[1-7]_*')):
        path = folder/'03_ISOMERS.xlsx'
        if not path.is_file():
            raise FileNotFoundError(f'Required source isomer table missing: {path}')
        n_rows = 0
        with ZipFile(path) as z:
            strings = source_reader.strings(z)
            target = dict(source_reader.sheet_targets(z))['异构关系']
            for excel_row,c in source_reader.rows(z,target,strings):
                if excel_row == 1:
                    continue
                vals = [source_reader.val(c,source_reader.colname(i)) for i in range(1,17)]
                row = dict(zip(RELATION_HEADERS[:16],vals))
                key = _pair(row['连接结构键A'],row['连接结构键B'])
                if not all(key) or not row['主异构关系']:
                    raise ValueError(f'Empty original constitutional pair at {path} row {excel_row}')
                location = f'{folder.name}/03_ISOMERS.xlsx#异构关系!A{excel_row}:P{excel_row}'
                for suffix in ('A','B'):
                    member = row['连接结构键'+suffix]
                    st = _structure(row['成员'+suffix+'_CanonicalSMILES'],member,location)
                    st['evidence'] = [location]
                    prior = structure_bank.get(member)
                    if prior and prior['connectivity_smiles'] != st['connectivity_smiles']:
                        raise ValueError(f'Original relation SMILES conflict for {member}')
                    if prior:
                        prior['evidence'].append(location)
                    else:
                        structure_bank[member] = st
                _validate_constitutional(structure_bank[key[0]],structure_bank[key[1]],key)
                if row['分子式'] != structure_bank[key[0]]['formula']:
                    raise ValueError(f'Original relation formula mismatch at {location}')
                originals[key].append({'row':row,'location':location,'source':folder.name})
                n_rows += 1
        source_rows += n_rows
        sources_report.append({'source':folder.name,'relation_rows':n_rows})

    if len(sources_report) != 7:
        raise ValueError(f'Expected seven source relation workbooks, got {len(sources_report)}')
    labels = {}
    for key,entries in originals.items():
        types = {e['row']['主异构关系'] for e in entries}
        if len(types) != 1:
            raise ValueError(f'Conflicting original source relation labels for {key}: {sorted(types)}')
        labels[key] = next(iter(types))

    member_stats = defaultdict(lambda:{'names':Counter(),'sources':set(),'n':0,'families':set(),'record_ids':[]})
    families = defaultdict(set)
    record_structure_cache = {}
    for r in records:
        member = str(r.get('Member_ID') or '').strip()
        family = str(r.get('Family_ID') or '').strip()
        if not member or not family:
            continue
        stats = member_stats[member]
        stats['names'][str(r.get('Solute') or member)] += 1
        stats['sources'].update(_sources(r))
        stats['n'] += 1
        stats['families'].add(family)
        stats['record_ids'].append(str(r.get('canonical_record_id','')))
        families[family].add(member)
        smiles = r.get('Solute_SMILES')
        if smiles:
            cache_key = (member,str(smiles))
            st = record_structure_cache.get(cache_key)
            if st is None:
                st = _structure(smiles,member,str(r.get('canonical_record_id','')))
                record_structure_cache[cache_key] = st
            prior = structure_bank.get(member)
            if prior and prior['connectivity_smiles'] != st['connectivity_smiles']:
                raise ValueError(f'Current solute vs original relation structural conflict for {member}')
            if not prior:
                st['evidence'] = ['当前规范实验记录确认的 Solute_SMILES; '+str(r.get('canonical_record_id',''))]
                structure_bank[member] = st

    generated = {}
    requested_pairs = set()
    for family,members in sorted(families.items()):
        for key in itertools.combinations(sorted(members),2):
            requested_pairs.add(key)
            if not all(k in structure_bank for k in key):
                raise ValueError(f'Confirmed full solute structures required to validate family relation: {family}; {key}')
            x,y = structure_bank[key[0]], structure_bank[key[1]]
            _validate_constitutional(x,y,key)
            if key not in labels:
                a = {'features':classifier.structure_features(x['mol'])}
                b = {'features':classifier.structure_features(y['mol'])}
                classification = classifier.classify_pair_relation(a,b)
                labels[key] = classification['type']
                generated[key] = classification

    rows_out = []
    for key in sorted(labels):
        original = originals.get(key,[])
        x,y = structure_bank[key[0]],structure_bank[key[1]]
        if original:
            # Preserve the established label/basis, orient by stable connectivity keys.
            first = original[0]['row']
            row = dict(first)
            if first['连接结构键A'] != key[0]:
                for a,b in [('成员A','成员B'),('连接结构键A','连接结构键B'),
                 ('成员A_CanonicalSMILES','成员B_CanonicalSMILES'),('成员A_IDAC记录数','成员B_IDAC记录数'),
                 ('成员A_来源数','成员B_来源数')]:
                    row[a],row[b] = row[b],row[a]
            row['分类产生方式'] = '七源既有03关系合并'
            row['来源关系表位置'] = '; '.join(e['location'] for e in original)
        else:
            c = generated[key]
            row = {'异构体家族ID':next(iter(sorted(member_stats[key[0]]['families'] & member_stats[key[1]]['families']))),
              '分子式':x['formula'],'成员A':key[0],'成员B':key[1],
              '主异构关系':c['type'],'伴随结构差异':'；'.join(c['secondary_tags']),
              '置信度':c['confidence'],'分类依据':c['basis'],
              '连接结构键A':key[0],'连接结构键B':key[1],
              '成员A_CanonicalSMILES':x['smiles'],'成员B_CanonicalSMILES':y['smiles'],
              '成员A_IDAC记录数':0,'成员B_IDAC记录数':0,'成员A_来源数':0,'成员B_来源数':0,
              '分类产生方式':'同一既有分类代码补充跨来源组合','来源关系表位置':str(classifier_path)}
        for suffix,member in zip(('A','B'),key):
            stats = member_stats.get(member)
            if stats:
                row['成员'+suffix] = _name(stats['names'],row['成员'+suffix])
                row['成员'+suffix+'_IDAC记录数'] = stats['n']
                row['成员'+suffix+'_来源数'] = len(stats['sources'])
                common = member_stats[key[0]]['families'] & member_stats[key[1]]['families'] if key[0] in member_stats and key[1] in member_stats else set()
                if common:
                    row['异构体家族ID'] = next(iter(sorted(common)))
            else:
                row['成员'+suffix+'_IDAC记录数'] = 0
                row['成员'+suffix+'_来源数'] = 0
        row['结构确认来源'] = '; '.join(dict.fromkeys(x['evidence'] + y['evidence']))
        row.update({english:row[chinese] for english,chinese in FIELD_ALIASES.items()})
        rows_out.append(row)

    report = {'source_workbooks':sources_report,'source_relation_rows':source_rows,
      'source_unique_pairs':len(originals),'source_label_conflicts':0,
      'current_records':len(records),'current_families':len(families),'current_members':len(member_stats),
      'current_family_combinations':len(requested_pairs),'generated_cross_source_pairs':len(generated),
      'total_relation_pairs':len(labels),'strict_formula_and_connectivity_validation':True,
      'classifier_source':str(classifier_path),'classifier_sha256':hashlib.sha256(classifier_path.read_bytes()).hexdigest(),
      'generated_labels':dict(Counter(v['type'] for v in generated.values())),
      'generated_pairs':[{'Member_ID_A':k[0],'Member_ID_B':k[1],**v} for k,v in sorted(generated.items())]}
    RELATIONS_ROWS[:] = rows_out
    LAST_REPORT.clear(); LAST_REPORT.update(report)
    (HERE/'relations_v2_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    (HERE/'relations_v2_rows.json').write_text(json.dumps({'headers':RELATION_HEADERS,'field_aliases':FIELD_ALIASES,'rows':rows_out},ensure_ascii=False,indent=2),encoding='utf8')
    return labels

if __name__ == '__main__':
    import argparse,csv
    parser=argparse.ArgumentParser()
    parser.add_argument('--project',required=True)
    parser.add_argument('--records',required=True)
    args=parser.parse_args()
    with Path(args.records).open(encoding='utf-8-sig',newline='') as f:
        rr=list(csv.DictReader(f))
    result=load_relations(args.project,rr)
    print(json.dumps({k:v for k,v in LAST_REPORT.items() if k!='generated_pairs'},ensure_ascii=False,indent=2))
