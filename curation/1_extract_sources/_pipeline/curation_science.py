"""Rebuild current curation science from a SHA-locked seven-source03 snapshot.

Only output_dir is written. Each workbook entry may call this into temporary
work, then author its one requested workbook. Original source workbooks,
another curated workbook and training packages are never implicit outputs.
"""
from __future__ import annotations
from pathlib import Path
from collections import Counter
import csv,gzip,hashlib,json,math,sys

HERE=Path(__file__).resolve().parent

def _sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def _load(path):return json.loads(Path(path).read_text(encoding='utf-8'))

def _save(path,obj):Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def _rows(path):
    path=Path(path);opener=gzip.open if path.suffix=='.gz' else open
    with opener(path,'rt',encoding='utf-8-sig',newline='') as f:yield from csv.DictReader(f)

def _list(v):return v if isinstance(v,list) else json.loads(v) if v else []

def _write(path,rows,fields=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if fields is None:fields=list(dict.fromkeys(k for r in rows for k in r if k!='Condition_History'))
    opener=gzip.open if path.suffix=='.gz' else open
    with opener(path,'wt',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader()
        for r in rows:w.writerow({k:json.dumps(v,ensure_ascii=False,separators=(',',':')) if isinstance(v,(dict,list,tuple)) else v for k,v in r.items() if k in fields})

def _cache_output_paths(stage,report):
    """Every path exposed to workbook callers, plus existing CSV audit outputs."""
    stage=Path(stage);paths={Path(v).relative_to(stage).as_posix() for k,v in _bank(stage).items() if k!='stage'}
    paths.update(Path(v).as_posix() for v in report['csv_sha256'])
    return sorted(paths)

def _cache_output_hashes(stage,report):
    return {relative:_sha(Path(stage)/relative) for relative in _cache_output_paths(stage,report)}

def _verified_cache_bank(stage,ready):
    """Reject incomplete or altered science; never bless files merely by reading them."""
    stage=Path(stage);report_path=stage/'curation/data_preparation_report.json'
    error='科学缓存校验失败，请清理该工作缓存后重新重建；不能继续使用旧缓存：'
    if not report_path.is_file() or ready.get('report_sha256')!=_sha(report_path):
        raise ValueError(error+'数据审计报告SHA不一致。')
    report=_load(report_path);expected=ready.get('bank_output_sha256')
    paths=_cache_output_paths(stage,report)
    if not isinstance(expected,dict) or set(expected)!=set(paths):
        raise ValueError(error+'缺少完整科学输出的SHA校验清单。')
    for relative in paths:
        path=stage/relative
        if not path.is_file() or _sha(path)!=expected[relative]:
            raise ValueError(error+relative+' 文件缺失或SHA不一致。')
    for relative,digest in report['csv_sha256'].items():
        if expected[Path(relative).as_posix()]!=digest:
            raise ValueError(error+'审计报告与输出校验清单不一致：'+relative)
    return _bank(stage)

def build_current_science(project_root,output_dir,asset_dir=None):
    """Return path bank after validating source03 + all central asset SHA locks."""
    project=Path(project_root).resolve();stage=Path(output_dir).resolve();assets=Path(asset_dir or HERE/'assets/curation').resolve()
    if stage==project or project in stage.parents:raise ValueError('科学重建只写独立工作目录，不能写来源或生产文件夹。')
    manifest_path=assets/'current_source03_inputs.json';manifest=_load(manifest_path)
    if manifest.get('source_scope')!=163074:raise ValueError('审定来源范围不是当前七份03的163074条。')
    # Always recheck, including a possible cache hit: changed source is not silently ignored.
    for entry in manifest['inputs']:
        path=project/entry['folder']/'03_ISOMERS.xlsx'
        if not path.is_file() or _sha(path)!=entry['current_workbook_sha256']:
            raise ValueError('来源表三已经变化，请重新审核并冻结中央来源bank：'+str(path))
    for filename,digest in manifest['asset_sha256'].items():
        if _sha(assets/filename)!=digest:raise ValueError('中央审定科学资产SHA不一致：'+filename)
    for relative,digest in manifest.get('raw_source_sha256',{}).items():
        if _sha(project/relative)!=digest:raise ValueError('原始00_Source文件SHA已经变化：'+relative)
    cache=stage/'SCIENCE_READY.json'
    if cache.is_file():
        ready=_load(cache)
        if ready.get('status')=='PASS' and ready.get('central_manifest_sha256')==_sha(manifest_path):
            return _verified_cache_bank(stage,ready)
    sys.path.insert(0,str(assets/'common_modules'))
    import canonicalize_v2 as canon
    import temperature_qc_v2 as tempqc
    import other_quality_v2 as other
    import dataset_helpers_v2 as helpers
    csv.field_size_limit(40*1024*1024)
    raw=list(_rows(assets/'current_source03_observations.csv.gz'));raw_map={r['canonical_record_id']:r for r in raw}
    if len(raw)!=163074 or len(raw_map)!=len(raw):raise ValueError('中央原记录范围或ID唯一性不合格。')
    # No structure inference: every payload is the reviewed source03 identity.
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors
    verified=set()
    for r in raw:
        for role in ('Solute','Solvent'):
            payload=tuple(r[role+'_'+k] for k in ('InChI','Identity','SMILES','Formula'))
            if payload in verified:continue
            inchi,key,smiles,formula=payload;m=Chem.MolFromSmiles(smiles)
            if m is None or Chem.InchiToInchiKey(inchi)!=key or Chem.MolToInchiKey(m)!=key or rdMolDescriptors.CalcMolFormula(m)!=formula:raise ValueError('中央完整结构载荷不一致：'+key)
            verified.add(payload)
    registry=_load(assets/'current_source03_solvent_registry.json');context=_load(assets/'current_source03_quality_context.json')
    relation_catalog=_load(assets/'current_source03_relations.json');labels={tuple(sorted((r['连接结构键A'],r['连接结构键B']))):r['主异构关系'] for r in relation_catalog['rows']}
    fixed=_load(assets/'family_fold_manifest.json')['assignment'];result=canon.canonicalize(raw,registry);canonical=result['kept']
    for r in canonical:
        parents=[raw_map[k] for k in _list(r['alias_parent_record_ids'])];equal=all(float(p['gamma_inf']).hex()==float(r['gamma_inf']).hex() for p in parents)
        r.update(Duplicate_Class='A 完全重复' if len(parents)>1 and equal else 'B 同原研究舍入重复' if len(parents)>1 else '唯一记录',Original_Measurement_Count=len(parents),Solute_Mol_ID='MOL_'+r['Solute_Identity'])
        if any(p.get('Audited_Potential_Source_Estimate')=='是' for p in parents):r.update(Audited_Potential_Source_Estimate='是',Audited_Potential_Source_Estimate_Reason='继承既有审计：转载候选原研究的实测出处尚未核实；实验主表保留，当前模型暂缓。')
    qc=tempqc.screen([other.prepare_temperature_record(r,context) for r in canonical]);repeat=helpers.unresolved_repeat_candidates(canonical);model=[];holds=[]
    for r in canonical:
        d=qc['results'][r['canonical_record_id']];r.update(QC_Status=d['status'],QC_Reason=d.get('reason',''),QC_Factor=d.get('factor',''),ln_gamma_inf=repr(math.log(float(r['gamma_inf']))),Fold_ID=fixed.get(r['Family_ID'],''));reason='';action=''
        if r['canonical_record_id'] in qc['removals']:reason=qc['removals'][r['canonical_record_id']]['reason'];action='EXCLUDE_QC'
        elif r.get('Audited_Potential_Source_Estimate')=='是':reason=r['Audited_Potential_Source_Estimate_Reason'];action='HOLD_MODEL_POTENTIAL_SOURCE_ESTIMATE'
        elif r['canonical_record_id'] in repeat:reason='原研究未唯一确定，与已知研究同条件近值，可能是转载副本；实验主表保留，模型暂缓，不冒认原研究或实验方法。';action='HOLD_MODEL_POTENTIAL_REPEAT'
        elif r['Family_ID'] not in fixed:reason='此家族不在已固定的84家族划分清单内；实验主表保留，当前模型暂缓，固定家族角色不改。';action='HOLD_MODEL_AUDITED_SCOPE'
        r.update(Model_Eligible='否' if reason else '是',Model_Eligibility_Reason=reason or '当前七份03最终来源结构与质量通过；同研究去重/冲突排除后保留真实实验。',Strict_Identity_Eligible='是',Strict_Model_Eligible='否' if reason else '是')
        if reason:holds.append(dict(r,Action=action,Reason=reason))
        else:model.append(r)
    all_pairs,unmatched=helpers.matched(canonical,labels);pairs,_=helpers.matched(model,labels)
    canonical.sort(key=lambda r:(r['Family_ID'],r['Member_ID'],r['Solvent_Audit_ID'],r['Study_ID'],float(r['T_K']),r['canonical_record_id']));model.sort(key=lambda r:r['canonical_record_id']);pairs.sort(key=lambda p:(p['MatchedGroup_ID'],p['Pair_ID']))
    rawids=set(raw_map);endpoints={r['canonical_record_id']:r for r in model}
    for r in canonical:
        selected=raw_map[r['selected_input_record_id']]
        if float(r['gamma_inf']).hex()!=float(selected['gamma_inf']).hex() or float(r['T_K']).hex()!=float(selected['T_K']).hex():raise AssertionError('canonical实验值发生了改写。')
        if set(_list(r['alias_parent_record_ids']))-rawids:raise AssertionError('canonical包含当前范围外父记录。')
    for p in pairs:
        for k in ('canonical_record_id_A','canonical_record_id_B'):
            if p[k] not in endpoints or p['Solvent_Mol_ID']!=endpoints[p[k]]['Solvent_Mol_ID']:raise AssertionError('配对端点或溶剂具体结构不一致。')
    messages={'KEEP':'保留原研究/文献优先的真实原值，未平均或预测。','MERGE_DUPLICATE':'同一原研究、双方完整结构、同温度及唯一兼容实验条件；全组γ∞跨度不超过0.5%，选一条真实原值，其余原值与来源历史保留。','HOLD_NUMERIC_CONFLICT':'同一原研究及实验条件的全组γ∞跨度超过0.5%，整组不进入保留实验；原值与来源位置保留待回原文。','HOLD_AMBIGUOUS_CONDITION':'缺失的条件可对应多个不同实验方法或条件系列，不能用γ∞强行指定归属；保留审计待原研究确认。','HOLD_INVALID_INPUT':'非正或非有限γ∞/温度，原值保留审计。'}
    primary=[dict(raw_map[d['canonical_record_id']],**d,Reason_CN=messages[d['Action']]) for d in result['decisions']]
    out=stage/'curation';data=stage/'data';out.mkdir(parents=True,exist_ok=True);data.mkdir(parents=True,exist_ok=True)
    _write(out/'source_observations.csv.gz',raw);_write(out/'deduplicated_experiments.csv',canonical);_write(out/'experimental_task1.csv',canonical);_write(out/'experimental_task2.csv',all_pairs);_write(out/'primary_record_decisions.csv',primary);_write(out/'record_decisions.csv',primary+holds);_write(out/'model_holds.csv',holds);_write(out/'unmatched_experiments.csv',unmatched);_write(data/'task1.csv',model);_write(data/'task2.csv',pairs)
    with gzip.open(out/'canonical_condition_history.json.gz','wt',encoding='utf-8') as f:
        f.write('{')
        for i,r in enumerate(canonical):
            if i:f.write(',')
            f.write(json.dumps(r['canonical_record_id']));f.write(':');f.write(r['Condition_History'])
        f.write('}')
    _save(out/'raw_to_canonical_mapping.json',result['mapping']);_save(out/'temperature_qc_results.json',qc);_save(out/'unresolved_repeat_candidates.json',repeat);_save(out/'canonicalization_report.json',result['stats']);_save(out/'solvent_registry.json',registry);_save(out/'relations_catalog.json',relation_catalog)
    import shutil
    for src,name in [('current_scope_exclusions.csv','scope_exclusions.csv'),('current_source03_family_members.csv','family_members.csv'),('current_source03_study_updates.json','study_identity_updates.json'),('current_source03_prior_mapping.csv','prior_to_current_canonical_mapping.csv')]:shutil.copy2(assets/src,out/name)
    report=dict(manifest['report_metadata']);report.update(canonicalization=result['stats'],canonical_before_temperature_qc=len(canonical),experimental_master_task1=len(canonical),experimental_master_task2_pairs=len(all_pairs),experimental_matched_groups=len({p['MatchedGroup_ID'] for p in all_pairs}),task1=len(model),task2_pairs=len(pairs),matched_groups=len({p['MatchedGroup_ID'] for p in pairs}),families_task1=len({r['Family_ID'] for r in model}),model_hold_counts=dict(Counter(r['Action'] for r in holds)),QC=qc['stats'],duplicate_class_counts=dict(Counter(r['Duplicate_Class'] for r in canonical)),primary_action_counts=dict(Counter(r['Action'] for r in primary)),central_manifest_sha256=_sha(manifest_path),status='PASS')
    report['csv_sha256']={str(p.relative_to(stage)):_sha(p) for p in [out/'source_observations.csv.gz',out/'deduplicated_experiments.csv',out/'experimental_task1.csv',out/'experimental_task2.csv',data/'task1.csv',data/'task2.csv',out/'primary_record_decisions.csv']}
    _save(out/'data_preparation_report.json',report);_save(cache,{'status':'PASS','revision':report.get('revision'),'public_final':report.get('public_final',False),'report_sha256':_sha(out/'data_preparation_report.json'),'bank_output_sha256':_cache_output_hashes(stage,report),'central_manifest_sha256':_sha(manifest_path),'source_scope':163074,'canonical':len(canonical),'model_task1':len(model),'model_task2':len(pairs)})
    return _bank(stage)

def _bank(stage):
    out=Path(stage)/'curation';data=Path(stage)/'data'
    return {'canonical':str(out/'experimental_task1.csv'),'model_task1':str(data/'task1.csv'),'model_task2':str(data/'task2.csv'),'relations_catalog':str(out/'relations_catalog.json'),'report':str(out/'data_preparation_report.json'),'primary_decisions':str(out/'primary_record_decisions.csv'),'model_holds':str(out/'model_holds.csv'),'mapping':str(out/'raw_to_canonical_mapping.json'),'raw_observations':str(out/'source_observations.csv.gz'),'canonical_history':str(out/'canonical_condition_history.json.gz'),'scope_exclusions':str(out/'scope_exclusions.csv'),'family_members':str(out/'family_members.csv'),'temperature_qc':str(out/'temperature_qc_results.json'),'stage':str(stage)}
