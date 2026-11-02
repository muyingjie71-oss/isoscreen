"""Condition-safe, provenance-preserving canonicalisation of experiment records.

Pure public canonicalize(): no original files are edited. Molecular eligibility
is separate: unresolved structures stay in the experimental master. Numeric
conflicts and ambiguous condition correspondence are quarantined in decisions.
"""
from __future__ import annotations
from collections import Counter, defaultdict
from pathlib import Path
import argparse, csv, hashlib, json, math, re, unicodedata

SOURCE_PRIORITY = ('ThermoML','ILThermo','Brouwer','DeepFM','Tan','Sherman','单独收集论文')
CONDITION_FIELDS = ('pressure_kPa','phase','standard_state','method','fixed_conditions')
RELATIVE_THRESHOLD = .005


def norm(x):
    return re.sub(r'\s+', '', unicodedata.normalize('NFKC',str(x or '')).casefold())


def normalize_method(x):
    value=norm(x)
    if value.startswith(('chrom:','chromatog','gc')) or 'chromatography' in value:
        return 'gc'
    if value in ('dilutor','dilutormethod','dilutortechnique'):
        return 'dilutor'
    if value in ('gasstrippingmethod','inertgasstripping'):
        return 'gasstripping'
    return value


def normalize_phase(x):
    value=norm(x)
    return {'liquidphase':'liquid','nematicphase':'nematic',
            'isotropicphase':'isotropic','supercoolednematicphase':'supercoolednematic'}.get(value,value)


def normalize_number(x):
    if x is None or x=='':return ''
    try:
        y=float(x)
    except (TypeError,ValueError):return norm(x)
    return format(y,'.12g') if math.isfinite(y) else norm(x)


def conditions(record):
    """Canonical declarations, without filling missing experimental metadata."""
    if 'conditions' in record:return tuple(record['conditions'])
    phase=normalize_phase(record.get('Raw_相态',''))
    name=str(record.get('Solvent','')).casefold()
    for label in ('supercooled nematic phase','isotropic phase','nematic phase'):
        if label in name:phase=normalize_phase(label);break
    fixed=norm(record.get('Raw_其他固定条件','')) if record.get('Source') in ('ThermoML','ILThermo') else ''
    return (normalize_number(record.get('Raw_压力_kPa','')), phase,
            norm(record.get('Raw_标准态','')), normalize_method(record.get('Raw_实验方法','')),fixed)


def compatible(left,right):
    if isinstance(left,dict):left=conditions(left)
    if isinstance(right,dict):right=conditions(right)
    return all(not x or not y or x==y for x,y in zip(left,right))


def maximal_signatures(records):
    """Unions of maximal cliques over UNIQUE observed declaration patterns.

    Wildcards are not transitive. Values from conflicting declarations are
    never combined in one signature. Duplicate patterns are compressed before
    exact clique enumeration to keep the seven-source bins small.
    """
    observed=sorted({conditions(r) if isinstance(r,dict) else tuple(r) for r in records})
    if not observed:return []
    adjacency={i:{j for j in range(len(observed)) if i!=j and compatible(observed[i],observed[j])}
               for i in range(len(observed))}
    signatures=set()
    def visit(included,potential,seen):
        if not potential and not seen:
            merged=[]
            for j in range(len(CONDITION_FIELDS)):
                values={observed[i][j] for i in included if observed[i][j]}
                if len(values)>1:raise AssertionError('An incompatible condition clique was formed')
                merged.append(next(iter(values),''))
            signatures.add(tuple(merged));return
        pivot=max(potential|seen,key=lambda i:len(potential&adjacency[i])) if potential|seen else None
        for node in sorted(potential-(adjacency[pivot] if pivot is not None else set())):
            visit(included|{node},potential&adjacency[node],seen&adjacency[node])
            potential.remove(node);seen.add(node)
    visit(set(),set(adjacency),set())
    return sorted(signatures)


condition_signatures=maximal_signatures


def known_study(record):
    return str(record.get('Study_ID','')).startswith(('DOI:','BIB:','TITLE:'))


def stable(prefix,parts):
    return prefix+hashlib.sha256(json.dumps(parts,ensure_ascii=False,sort_keys=True,
                                          separators=(',',':')).encode('utf8')).hexdigest()[:20]


def _json_list(value,fallback=()):
    if isinstance(value,list):return value
    if value:
        decoded=json.loads(value)
        if not isinstance(decoded,list):raise ValueError('Expected JSON list')
        return decoded
    return list(fallback)


def literature_rank(record):
    source=record.get('Source','');doi=record.get('Original_DOI','') or record.get('Audited_Confirmed_DOI','')
    fixed=str(record.get('Raw_其他固定条件',''))
    primary=(source=='单独收集论文' and bool(doi) and any(v in fixed for v in
        ('source_class=original_experimental_table','source_class=original_measurement_derived')))
    tier=0 if primary else 1 if doi else 2 if known_study(record) else 4
    source_rank=SOURCE_PRIORITY.index(source) if source in SOURCE_PRIORITY else len(SOURCE_PRIORITY)
    use=record.get('Raw_建议用于建模',record.get('Raw_建议建模',''))
    quality=record.get('Raw_数据质量状态',record.get('Raw_质量标记',''))
    support=sum(bool(record.get(k)) for k in ('Reference','Raw_实验方法','Raw_压力_kPa','Raw_相态','Raw_标准态'))
    # Explicit equality/quality proofs may select a confirmed existing parent
    # over an unresolved-paper representative; they never alter grouping.
    return (int(record.get('Audited_Representative_Priority',1)),tier,source_rank,use not in ('','是'),quality not in ('','OK','OK_CORRECTED'),
            -support,str(record.get('source_record_key',record['canonical_record_id'])))


def _provenance(record):
    return {k:v for k,v in record.items() if k.startswith(('Raw_','Audited_')) or k in
            ('canonical_record_id','source_record_key','audit_canonical_record_id','Source',
             'original_record_id','Original_DOI','Reference','Study_ID','Solute','Solvent',
             'Solute_Identity','Solvent_Identity','T_K','gamma_inf','Original_Source_File',
             'Original_Source_Location','Original_Data_Target','Source_Original_Reference',
             'Source_Solvent_CAS','Source_Solvent_MW','Source_Data_Type','Source_Field_F',
             'Reported_Study_ID','Study_Identity_Basis','Study_Identity_Evidence',
             'Structure_Override_Evidence')}


def _override_changes(record,overrides):
    """Accept per-ID maps or explicit {match:{...},set:{...},evidence:...} rules."""
    if isinstance(overrides,dict):
        return overrides.get(record['canonical_record_id'],overrides.get(record.get('source_record_key',''),{}))
    if not isinstance(overrides,list):raise TypeError('record_overrides must be a dict or rule list')
    merged={}
    for rule in overrides:
        if not isinstance(rule,dict):raise TypeError('Each override rule must be a dict')
        match=rule.get('match',rule.get('selector'))
        updates=rule.get('set',rule.get('updates',rule.get('values')))
        if match is None and (rule.get('canonical_record_id') or rule.get('source_record_key')):
            field='canonical_record_id' if rule.get('canonical_record_id') else 'source_record_key'
            match={field:rule[field]}
            if updates is None:updates={k:v for k,v in rule.items() if k not in (field,'evidence')}
        if not match or not isinstance(match,dict) or not isinstance(updates,dict):
            raise ValueError('Override rule needs nonempty match/selector and set/updates/values dictionaries')
        if not all(str(record.get(k,'')) in {str(x) for x in v} if isinstance(v,list)
                   else str(record.get(k,''))==str(v) for k,v in match.items()):continue
        for k,v in updates.items():
            if k in merged and merged[k]!=v:raise ValueError('Conflicting matching structure overrides for '+record['canonical_record_id'])
            merged[k]=v
        if rule.get('evidence'):merged['Structure_Override_Evidence']=rule['evidence']
    return merged


def canonicalize(records,registry,record_overrides=None):
    """Return kept/decisions/mapping/stats while preserving original targets.

    registry uses old Solvent_Identity keys with RESOLVED Mol_ID/smiles entries.
    record_overrides is an evidence-backed per-input-ID metadata dictionary or
    a list of explicit match/set/evidence rules;
    experimental gamma, T, source ID and original source location are immutable.
    """
    overrides=record_overrides or {};rows=[];bins=defaultdict(list)
    decisions=[];mapping={};kept=[];seen=set();stats=Counter()
    immutable={'T_K','gamma_inf','original_record_id','source_record_key','Source',
               'Original_Source_File','Original_Source_Location','Original_Data_Target',
               'Raw_温度_K','Raw_无限稀释活度系数'}
    for original in records:
        r=dict(original);rid=r['canonical_record_id']
        if rid in seen:raise ValueError('Duplicate input record ID '+rid)
        seen.add(rid);mapping[rid]=None
        change=_override_changes(r,overrides)
        if change:
            if any((k in immutable or k.startswith('Raw_')) and str(v)!=str(r.get(k,'')) for k,v in change.items()):
                raise ValueError('An override attempted to change original values/provenance: '+rid)
            r.update(change)
        identity=r.get('Solvent_Identity','');entry=registry.get(identity,{})
        if entry.get('status')=='RESOLVED' and entry.get('Mol_ID'):
            r['Solvent_Mol_ID']=entry['Mol_ID'];r['Solvent_SMILES']=entry.get('smiles','')
            r['Solvent_Structure_Status']='RESOLVED'
        else:
            r['Solvent_Structure_Status']=entry.get('status','UNREGISTERED')
            r['Solvent_Mol_ID']='';r['Solvent_SMILES']=''
        # Explicit per-record structure overrides take priority over ambiguous
        # global shorthand definitions; raw Solvent_Identity stays unchanged.
        if 'Solvent_Structure_Status' in change or 'Solvent_Mol_ID' in change:
            r['Solvent_Mol_ID']=change.get('Solvent_Mol_ID','')
            r['Solvent_SMILES']=change.get('Solvent_SMILES','')
            r['Solvent_Structure_Status']=change.get('Solvent_Structure_Status','RESOLVED_OVERRIDE')
        keysolvent=r.get('Solvent_Mol_ID') if r['Solvent_Structure_Status'] in ('RESOLVED','RESOLVED_OVERRIDE') else change.get('Solvent_Audit_ID',identity)
        r['Solvent_Audit_ID']=keysolvent
        r['Structure_Status']='RESOLVED' if r['Solvent_Structure_Status'] in ('RESOLVED','RESOLVED_OVERRIDE') else r['Solvent_Structure_Status']
        r['Structure_Reason']=str(change.get('Structure_Reason',entry.get('reason','No uniquely confirmed full solvent molecular identity')))
        try:
            gamma=float(r['gamma_inf']);temperature=float(r['T_K'])
            valid=math.isfinite(gamma) and gamma>0 and math.isfinite(temperature) and temperature>0
        except (KeyError,ValueError,TypeError):valid=False
        if not valid:
            decisions.append(dict(canonical_record_id=rid,Action='HOLD_INVALID_INPUT',
                Reason='Nonpositive/nonfinite experimental gamma or temperature',Retained_canonical_record_id=''))
            stats['hold_invalid_input']+=1;continue
        rows.append(r)
        if known_study(r):
            bins[(r['Study_ID'],r['Solute_Identity'],keysolvent,normalize_number(r['T_K']))].append(r)
        else:bins[('UNKNOWN_RECORD',rid)].append(r)
    stats['input_records']=len(seen)
    for base,rr in sorted(bins.items(),key=lambda x:repr(x[0])):
        signatures=maximal_signatures(rr);assigned=defaultdict(list)
        for r in rr:
            candidate=[s for s in signatures if compatible(conditions(r),s)]
            r['Reported_Condition_Declaration']=json.dumps(conditions(r),ensure_ascii=False)
            if len(candidate)!=1:
                decisions.append(dict(canonical_record_id=r['canonical_record_id'],
                    Action='HOLD_AMBIGUOUS_CONDITION',Reason='Reported metadata match multiple independent experimental condition signatures; gamma is not used to assign a method',
                    Retained_canonical_record_id='',Candidate_Condition_Signatures=json.dumps(candidate,ensure_ascii=False)))
                stats['hold_ambiguous_condition']+=1
            else:assigned[candidate[0]].append(r)
        for signature,group in sorted(assigned.items()):
            gamma=[float(r['gamma_inf']) for r in group]
            spread=(max(gamma)-min(gamma))/max(gamma)
            if spread>RELATIVE_THRESHOLD:
                for r in group:
                    decisions.append(dict(canonical_record_id=r['canonical_record_id'],
                        Action='HOLD_NUMERIC_CONFLICT',Reason='Same known study/full identities/T and unique compatible measurement conditions; full-group gamma spread exceeds 0.5%; no value is repaired',
                        Retained_canonical_record_id='',Condition_Signature=json.dumps(signature,ensure_ascii=False),Relative_Spread=spread))
                    stats['hold_numeric_conflict']+=1
                continue
            selected=min(group,key=literature_rank);winner=dict(selected)
            source_keys=sorted({r.get('source_record_key',r['canonical_record_id']) for r in group})
            canonical=stable('CANV2_',source_keys)
            parents=sorted(r['canonical_record_id'] for r in group)
            sources=set();original_ids=set();audit_ids=set();oldkeys=set()
            for r in group:
                sources.update(s.strip() for s in str(r.get('source_list',r.get('Source',''))).split(';') if s.strip())
                original_ids.update(_json_list(r.get('original_record_ids'),[r.get('Source','')+'::'+r.get('original_record_id','')]))
                oldkeys.update(_json_list(r.get('original_record_keys'),[r.get('source_record_key',r['canonical_record_id'])]))
                audit_ids.update(_json_list(r.get('original_audit_ids'),[r.get('audit_canonical_record_id','')]))
            winner.update(canonical_record_id=canonical,
                selected_input_record_id=selected['canonical_record_id'],
                source_count=len(sources),source_list='; '.join(sorted(sources,key=lambda v:(SOURCE_PRIORITY.index(v) if v in SOURCE_PRIORITY else 999,v))),
                original_record_ids=json.dumps(sorted(original_ids),ensure_ascii=False),
                original_record_keys=json.dumps(sorted(oldkeys),ensure_ascii=False),
                original_audit_ids=json.dumps(sorted(audit_ids-{''}),ensure_ascii=False),
                alias_parent_record_ids=json.dumps(parents,ensure_ascii=False),
                Condition_Signature=json.dumps(signature,ensure_ascii=False),
                Effective_Conditions=json.dumps(signature,ensure_ascii=False),
                Condition_History=json.dumps([_provenance(r) for r in sorted(group,key=lambda r:r['canonical_record_id'])],ensure_ascii=False),
                ln_gamma_inf=repr(math.log(float(selected['gamma_inf']))))
            kept.append(winner);stats['kept_experimental_points']+=1
            for r in group:
                rid=r['canonical_record_id'];mapping[rid]=canonical
                action='KEEP' if r is selected else 'MERGE_DUPLICATE'
                decisions.append(dict(canonical_record_id=rid,Action=action,
                    Reason=(r.get('Audited_Papers_Representative_Reason') or 'Selected actual literature-prioritized experimental value') if action=='KEEP' else 'Compatible unique condition signature; whole-group gamma spread <=0.5%; original value/history retained in audit',
                    Retained_canonical_record_id=canonical,Condition_Signature=json.dumps(signature,ensure_ascii=False),Relative_Spread=spread))
                if action=='MERGE_DUPLICATE':stats['merged_duplicate_records']+=1
            if base[0]=='UNKNOWN_RECORD':stats['kept_unknown_study_points']+=1
    if len(decisions)!=len(seen) or len(mapping)!=len(seen):raise AssertionError('Not every raw record has a decision/mapping')
    if (stats['kept_experimental_points']+stats['merged_duplicate_records']+
        stats['hold_numeric_conflict']+stats['hold_ambiguous_condition']+stats['hold_invalid_input'])!=len(seen):
        raise AssertionError('Curation accounting does not balance')
    return dict(kept=kept,decisions=decisions,mapping=mapping,stats=dict(stats))


def write_csv(path,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',required=True,type=Path)
    parser.add_argument('--registry',required=True,type=Path)
    parser.add_argument('--overrides',type=Path)
    parser.add_argument('--output-dir',required=True,type=Path)
    args=parser.parse_args()
    with args.input.open(encoding='utf-8-sig') as f:raw=list(csv.DictReader(f))
    registry=json.loads(args.registry.read_text(encoding='utf8'))
    overrides=json.loads(args.overrides.read_text(encoding='utf8')) if args.overrides else None
    result=canonicalize(raw,registry,overrides)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    write_csv(args.output_dir/'canonical_experimental_master.csv',result['kept'])
    write_csv(args.output_dir/'raw_record_decisions.csv',result['decisions'])
    (args.output_dir/'raw_to_canonical_mapping.json').write_text(json.dumps(result['mapping'],ensure_ascii=False,indent=2),encoding='utf8')
    (args.output_dir/'canonicalization_report.json').write_text(json.dumps(result['stats'],ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(result['stats'],ensure_ascii=False,indent=2),flush=True)
