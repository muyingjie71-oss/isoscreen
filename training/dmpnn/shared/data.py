"""Frozen records, whole-family splits, unbiased group-first pair sampling.

Reported atmospheric pressures are canonicalised, a merged measurement keeps all its
publication contexts for the study check, and expected counts are read from freeze.json.
"""
from pathlib import Path
import numpy as np
import csv,json,hashlib,html,math,re,unicodedata
from graphs import GraphBank

# This delivery uses ONE joint Layer A model; only 2D molecular graphs are loaded.

def csv_rows(p):
    with open(p,encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

CONDITION_FIELDS=('pressure','phase','standard_state','method','fixed')

def _norm(value):
    value=unicodedata.normalize('NFKC',html.unescape(str(value))).casefold()
    value=value.translate(str.maketrans({'−':'-','–':'-','—':'-','‐':'-','’':"'",'′':"'"}))
    return re.sub(r'\s+','',value)

def _condition_values(value,label):
    if isinstance(value,str):
        try:value=json.loads(value)
        except json.JSONDecodeError as error:raise ValueError('Invalid condition JSON: '+label) from error
    if isinstance(value,dict):
        value=[value.get('pressure',value.get('pressure_kPa','')),value.get('phase',''),
               value.get('standard_state',''),value.get('method',''),
               value.get('fixed',value.get('fixed_conditions',''))]
    if not isinstance(value,(list,tuple)) or len(value)!=5:
        raise ValueError('Condition signature must have five entries: '+label)
    result=[_norm('' if x is None else x) for x in value]
    if result[0]:
        try:pressure=float(result[0])
        except ValueError as error:raise ValueError('Nonnumeric reported pressure: '+label) from error
        if not math.isfinite(pressure) or pressure<=0:raise ValueError('Invalid reported pressure: '+label)
        # Reported nominal atmospheric pressures (101, 101.3, 101.33, 101.325 kPa)
        # are one condition; only this documented window is canonicalised.
        if abs(pressure-101.325)<=0.5:pressure=101.325
        result[0]=format(pressure,'.12g')
    method=result[3]
    if method.startswith(('chrom:','chromatog','gc')) or 'chromatography' in method:result[3]='gc'
    elif method in ('dilutor','dilutormethod','dilutortechnique'):result[3]='dilutor'
    elif method in ('gasstrippingmethod','inertgasstripping'):result[3]='gasstripping'
    result[1]={'liquidphase':'liquid','nematicphase':'nematic','isotropicphase':'isotropic',
               'supercoolednematicphase':'supercoolednematic'}.get(result[1],result[1])
    return tuple(result)

def _has_signature(row,field):
    return field in row and row[field] not in ('',None)

def _reported_conditions(record):
    # Brouwer's "other fixed conditions" field is database source metadata
    # (CAS, molecular weight, solute category), not thermodynamic conditions.
    fixed=record.get('Raw_其他固定条件','') if record.get('Source') in ('ThermoML','ILThermo') else ''
    value=[record.get('Raw_压力_kPa',''),record.get('Raw_相态',''),record.get('Raw_标准态',''),record.get('Raw_实验方法',''),fixed]
    label=record.get('canonical_record_id','')
    signature=_condition_values(value,label+' source conditions')
    # Production freezes preserve the selected source declaration under this
    # longer name. Validate both spellings and every known raw declaration.
    for field in ('Reported_Conditions','Reported_Condition_Declaration'):
        if _has_signature(record,field):
            declared=_condition_values(record[field],label+' '+field)
            _compatible(signature,declared,label+' source/reported declaration')
            signature=tuple(x or y for x,y in zip(signature,declared))
    return signature

def _compatible(left,right,label):
    for a,b,field in zip(left,right,CONDITION_FIELDS):
        if a and b and a!=b:raise ValueError('Pair condition mismatch: '+label+' '+field)

def _covers(signature,conditions,label):
    # A matched signature must state each reported/effective known condition.
    # A blank signature cannot hide a known GC/dilutor distinction.
    for target,value,field in zip(signature,conditions,CONDITION_FIELDS):
        if value and target!=value:raise ValueError('Matched signature omits or changes condition: '+label+' '+field)

def _requires_signatures(freeze):
    flag=freeze.get('requires_matched_condition_signature',False)
    if isinstance(flag,str):flag=flag.lower() in ('true','1','yes')
    try:major=int(str(freeze.get('version','1.0')).split('.')[0])
    except ValueError:major=1
    return bool(flag) or major>=2

def validate_matched_conditions(records,pairs,freeze):
    """Validate endpoint conditions and one exact condition signature per group.

    Version 2 production freezes require signatures on records and pairs.
    Old version 1 engineering fixtures may omit the new metadata, while any
    supplied metadata still receives the same consistency checks.
    """
    required=_requires_signatures(freeze)
    ids={row['canonical_record_id']:row for row in records}
    if len(ids)!=len(records):raise ValueError('Duplicate canonical_record_id')
    effective={};reported={}
    for rid,row in ids.items():
        reported[rid]=_reported_conditions(row)
        if required and not _has_signature(row,'Effective_Conditions'):
            raise ValueError('Version 2 record missing Effective_Conditions: '+rid)
        effective[rid]=_condition_values(row['Effective_Conditions'],rid+' Effective_Conditions') if _has_signature(row,'Effective_Conditions') else reported[rid]
        _compatible(effective[rid],reported[rid],rid)
        _covers(effective[rid],reported[rid],rid)
    # Legacy groups did not store a signature. Derive their one compatible
    # signature from the whole group's reported metadata, rather than from
    # individual pairs that can have different amounts of missing metadata.
    legacy_values={};legacy_signatures={}
    if not required:
        for pair in pairs:
            gid=pair.get('MatchedGroup_ID','')
            values=legacy_values.setdefault(gid,[set() for _ in CONDITION_FIELDS])
            for endpoint in ('canonical_record_id_A','canonical_record_id_B'):
                rid=pair.get(endpoint)
                if rid not in effective:raise ValueError('Pair endpoint not in frozen Task1: '+pair.get('Pair_ID',''))
                for index,value in enumerate(effective[rid]):
                    if value:values[index].add(value)
            for field in ('Matched_Conditions','Effective_Conditions'):
                if _has_signature(pair,field):
                    for index,value in enumerate(_condition_values(pair[field],pair.get('Pair_ID','')+' '+field)):
                        if value:values[index].add(value)
        for gid,values in legacy_values.items():
            if any(len(value)>1 for value in values):
                raise ValueError('MatchedGroup_ID has multiple condition signatures: '+gid)
            legacy_signatures[gid]=tuple(next(iter(value)) if value else '' for value in values)
    groups={};pair_ids=set();endpoint_signatures={}
    for pair in pairs:
        pid=pair.get('Pair_ID','')
        if not pid or pid in pair_ids:raise ValueError('Pair_ID must be nonempty and unique')
        pair_ids.add(pid)
        try:a=ids[pair['canonical_record_id_A']];b=ids[pair['canonical_record_id_B']]
        except KeyError as error:raise ValueError('Pair endpoint not in frozen Task1: '+pid) from error
        ai,bi=a['canonical_record_id'],b['canonical_record_id']
        _compatible(effective[ai],effective[bi],pid)
        _compatible(reported[ai],reported[bi],pid+' reported')
        provided=[_condition_values(pair[field],pid+' '+field) for field in ('Matched_Conditions','Effective_Conditions') if _has_signature(pair,field)]
        if required and not provided:raise ValueError('Version 2 pair missing Matched_Conditions: '+pid)
        if len(provided)==2 and provided[0]!=provided[1]:raise ValueError('Pair condition signatures disagree: '+pid)
        signature=provided[0] if provided else legacy_signatures[pair.get('MatchedGroup_ID','')]
        _covers(signature,effective[ai],pid+' endpoint A')
        _covers(signature,effective[bi],pid+' endpoint B')
        for rid in (ai,bi):
            if rid in endpoint_signatures:
                _compatible(endpoint_signatures[rid],signature,rid+' reused endpoint matched signatures')
                endpoint_signatures[rid]=tuple(x or y for x,y in zip(endpoint_signatures[rid],signature))
            else:endpoint_signatures[rid]=signature
        for field in ('Family_ID','Solvent_Mol_ID'):
            if a[field]!=b[field]:raise ValueError('Pair condition mismatch: '+pid+' '+field)
            if required and not pair.get(field):raise ValueError('Version 2 pair missing field: '+pid+' '+field)
            if pair.get(field) and pair[field]!=a[field]:raise ValueError('Pair identity metadata mismatch: '+pid+' '+field)
        # A merged measurement keeps every publication context it appeared in.
        if not pair.get('Study_ID'):raise ValueError('Pair missing Study_ID: '+pid)
        for rec in (a,b):
            if pair['Study_ID'] not in study_contexts(rec):raise ValueError('Pair study is not a context of endpoint: '+pid)
        if abs(float(a['T_K'])-float(b['T_K']))>1e-9:raise ValueError('Pair temperature mismatch: '+pid)
        if required and not pair.get('T_K'):raise ValueError('Version 2 pair missing temperature: '+pid)
        if pair.get('T_K') and abs(float(pair['T_K'])-float(a['T_K']))>1e-9:raise ValueError('Pair temperature metadata mismatch: '+pid)
        gid=pair.get('MatchedGroup_ID','')
        if not gid:raise ValueError('MatchedGroup_ID must be nonempty: '+pid)
        key=(a['Family_ID'],pair['Study_ID'],a['Solvent_Mol_ID'],format(float(a['T_K']),'.12g'),signature)
        if gid in groups and groups[gid]!=key:
            raise ValueError('MatchedGroup_ID has multiple study/system/temperature/condition signatures: '+gid)
        groups[gid]=key
    return {'records_checked':len(records),'pairs_checked':len(pairs),'matched_groups_checked':len(groups),'production_condition_signatures_required':required}

def study_contexts(record):
    raw=record.get('Study_Context_IDs','')
    ctx=set(json.loads(raw)) if raw else set()
    ctx.add(record['Study_ID'])
    return ctx

class Dataset:
    def __init__(self,path,layer='A',feature_file=None):
        if layer!='A' or feature_file is not None:
            raise ValueError('This standalone package only supports Layer A, without physical features')
        self.path=Path(path)
        self.freeze=json.loads((self.path/'freeze.json').read_text(encoding='utf8'))
        for name,digest in self.freeze['files'].items():
            if sha(self.path/name)!=digest:raise ValueError('Frozen Layer A dataset/checksum mismatch: '+name)
        if sha(self.path/'source_freeze_original.json')!=self.freeze['source_original_freeze_sha256']:
            raise ValueError('Original-source freeze provenance changed')
        self.fingerprint=sha(self.path/'freeze.json')
        self.records=csv_rows(self.path/'task1.csv');self.pairs=csv_rows(self.path/'task2.csv')
        self.splits=json.loads((self.path/'split_manifest.json').read_text(encoding='utf8'))
        self.bank=GraphBank(self.path/'graphs_layer_a_v2.npz')
        if len({r['canonical_record_id'] for r in self.records})!=len(self.records):
            raise ValueError('Duplicate experiment ID')
        self.ids={r['canonical_record_id']:i for i,r in enumerate(self.records)}
        self.matched_condition_validation=validate_matched_conditions(self.records,self.pairs,self.freeze)
        self.y64=np.asarray([float(r['ln_gamma_inf']) for r in self.records],np.float64)
        self.y=self.y64.astype(np.float32)
        self.dy64=np.asarray([float(p['D_AB']) for p in self.pairs],np.float64)
        self.t=np.asarray([1000./float(r['T_K']) for r in self.records],np.float32)
        self.sm=np.asarray([int(r['solute_mol_index']) for r in self.records],np.int32)
        self.vm=np.asarray([int(r['solvent_mol_index']) for r in self.records],np.int32)
        self.pa=np.asarray([self.ids[p['canonical_record_id_A']] for p in self.pairs],np.int32)
        self.pb=np.asarray([self.ids[p['canonical_record_id_B']] for p in self.pairs],np.int32)
        self.molecules=csv_rows(self.path/'molecules.csv')
        self.molecule_ids=[r['Mol_ID'] for r in self.molecules]
        if len(set(self.molecule_ids))!=len(self.molecule_ids):raise ValueError('Duplicate molecule identity')
        if len(self.molecule_ids)!=len(self.bank.arr['atom_ptr'])-1:
            raise ValueError('Graph cache molecule count differs from frozen molecules')
        for r,si,vi in zip(self.records,self.sm,self.vm):
            if self.molecule_ids[si]!=r['Solute_Mol_ID'] or self.molecule_ids[vi]!=r['Solvent_Mol_ID']:
                raise ValueError('Record molecule identity/index mismatch: '+r['canonical_record_id'])
        if not np.isfinite(self.y64).all() or not np.isfinite(self.dy64).all():
            raise ValueError('Nonfinite observed targets')
        for p,ai,bi in zip(self.pairs,self.pa,self.pb):
            a,b=self.records[ai],self.records[bi]
            if any(a[k]!=b[k] for k in ('Family_ID','Solvent_Mol_ID')) or p['Study_ID'] not in study_contexts(a) or p['Study_ID'] not in study_contexts(b):
                raise ValueError('Pair endpoint mismatch: '+p['Pair_ID'])
            if abs(float(a['T_K'])-float(b['T_K']))>1e-9:
                raise ValueError('Pair temperature mismatch: '+p['Pair_ID'])
            if a['Member_ID']==b['Member_ID']:
                raise ValueError('Pair endpoints represent the same isomer: '+p['Pair_ID'])
            if abs(float(p['D_AB'])-(float(a['ln_gamma_inf'])-float(b['ln_gamma_inf'])))>1e-9:
                raise ValueError('Pair differential target changed: '+p['Pair_ID'])
        expected=self.freeze['expected_counts']
        if len(self.records)!=expected['task1'] or len(self.pairs)!=expected['task2']:
            raise ValueError('Unexpected frozen sample counts: expected '+json.dumps(expected))
        self.layer='A';self.feature_fingerprint='none';self.feature_names=[]
        self.cohort=self.freeze.get('comparison_cohort','A_FULL_DATA')

    def split(self,fold):
        spec=self.splits['rounds'][str(fold)]
        # a round lists whole families (original) or whole solvents (train_solvents/...)
        key,field=(('_solvents','Solvent_Mol_ID') if 'train_solvents' in spec else ('_families','Family_ID'))
        result={}
        for role in ('train','validation','test'):
            keep=set(spec[role+key])
            result[role]=np.asarray([i for i,r in enumerate(self.records) if r[field] in keep],np.int32)
        if any(not len(v) for v in result.values()):
            raise ValueError('An original family fold has zero samples')
        if any(not set(result[a]).isdisjoint(result[b]) for a,b in [('train','validation'),('train','test'),('validation','test')]):
            raise ValueError('Family split leakage')
        if sum(len(v) for v in result.values())!=len(self.records):
            raise ValueError('Original family assignment does not cover dataset')
        return result

    def pair_indices(self,record_ids):
        mask=np.zeros(len(self.records),bool);mask[record_ids]=True
        if np.any(mask[self.pa]!=mask[self.pb]):raise ValueError('Paired endpoints split across folds')
        return np.flatnonzero(mask[self.pa]&mask[self.pb]).astype(np.int32)

    def scalers(self,train):
        s={'y_mean':float(self.y64[train].mean()),
           'y_std':float(self.y64[train].std()),
           't_mean':float(self.t[train].mean(dtype=np.float64)),
           't_std':float(self.t[train].std(dtype=np.float64))}
        if min(s['y_std'],s['t_std'])<1e-8:
            raise ValueError('Training variance is zero')
        return s

    def record_batch(self,record_ids,scalers,capacity=None):
        ids=np.asarray(record_ids,np.int32);n=len(ids);cap=capacity or n
        if n>cap:raise ValueError('Record batch overflow')
        pad=np.zeros(cap,np.int32);pad[:n]=ids
        mask=np.zeros(cap,np.float32);mask[:n]=1
        mols=np.concatenate([self.sm[pad],self.vm[pad]])
        g,lookup=self.bank.pack(mols)
        return {'graph':g,'solute_graph':lookup[:cap],'solvent_graph':lookup[cap:],
                'temperature':((self.t[pad]-scalers['t_mean'])/scalers['t_std']).astype(np.float32),
                'record_mask':mask}

    def train_batch(self,absolute_ids,pair_ids,scalers,absolute_batch,pair_batch):
        absids=np.asarray(absolute_ids,np.int32);ps=np.asarray(pair_ids,np.int32)
        joined=np.concatenate([absids,self.pa[ps],self.pb[ps]])
        uniq,loc=np.unique(joined,return_inverse=True)
        b=self.record_batch(uniq,scalers,absolute_batch+2*pair_batch)
        na=len(absids);npair=len(ps)
        def padded(x,n,cap,dtype):
            y=np.zeros(cap,dtype);y[:n]=x;return y
        b.update(absolute_map=padded(loc[:na],na,absolute_batch,np.int32),
            absolute_y=padded((self.y64[absids]-scalers['y_mean'])/scalers['y_std'],na,absolute_batch,np.float32),
            absolute_mask=padded(np.ones(na),na,absolute_batch,np.float32),
            pair_a_map=padded(loc[na:na+npair],npair,pair_batch,np.int32),
            pair_b_map=padded(loc[na+npair:],npair,pair_batch,np.int32),
            pair_y=padded(self.dy64[ps]/scalers['y_std'],npair,pair_batch,np.float32),
            pair_mask=padded(np.ones(npair),npair,pair_batch,np.float32))
        return b

class GroupSampler:
    """Group-uniform sampling as in the original combined-loss implementation."""
    def __init__(self,pairs,pair_ids):
        groups={}
        for i in pair_ids:groups.setdefault(pairs[i]['MatchedGroup_ID'],[]).append(int(i))
        self.groups=[np.asarray(v,np.int32) for _,v in sorted(groups.items())]
        if not self.groups:raise ValueError('No matched groups in training fold')
    def sample(self,rng,n):
        selected=rng.integers(len(self.groups),size=n)
        return np.asarray([self.groups[g][rng.integers(len(self.groups[g]))] for g in selected],np.int32)
