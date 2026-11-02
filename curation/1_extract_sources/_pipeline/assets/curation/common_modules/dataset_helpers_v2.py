"""Standalone deterministic I/O, matched groups, molecular graphs and fixed folds."""
from pathlib import Path
from collections import Counter,defaultdict
import csv,gzip,hashlib,itertools,json,math
import numpy as np
from rdkit import Chem
import canonicalize_v2 as canon

def read_csv(path):
    path=Path(path)
    opener=gzip.open if path.suffix=='.gz' else open
    with opener(path,'rt',encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def write_csv(path,rows,columns=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if columns is None:
        # Header discovery must not consume a one-pass record generator.
        rows=list(rows)
        columns=list(dict.fromkeys(k for r in rows for k in r if k!='Condition_History'))
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');writer.writeheader()
        for r in rows:writer.writerow({k:json.dumps(v,ensure_ascii=False,separators=(',',':')) if isinstance(v,(list,dict,tuple)) else v for k,v in r.items()})

def load_relations(assets):
    rows=json.loads((Path(assets)/'relations_catalog.json').read_text(encoding='utf8'))['rows']
    labels={}
    for r in rows:
        key=tuple(sorted((r['连接结构键A'],r['连接结构键B'])))
        label=r['主异构关系']
        if key in labels and labels[key]!=label:raise ValueError('Frozen relation label conflict '+str(key))
        labels[key]=label
    return labels

def save(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')

def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def stable(prefix, obj):
    return prefix+hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(',',':')).encode()).hexdigest()[:24]

def condition_tuple(r):
    v = r['Effective_Conditions']
    return tuple(json.loads(v) if isinstance(v,str) else v)

def unresolved_repeat_candidates(master):
    """Flag unknown-study potential reprints, without assigning a paper or method.

    These experiments remain in the master; only the molecular model view is
    held until provenance is established. Gamma is never used to infer Study_ID.
    """
    bins=defaultdict(list); evidence={}
    for r in master:
        if canon.known_study(r):
            bins[(r['Solute_Identity'],r['Solvent_Audit_ID'],format(float(r['T_K']),'.12g'))].append(r)
    for r in master:
        if canon.known_study(r):continue
        candidates=[]
        base=(r['Solute_Identity'],r['Solvent_Audit_ID'],format(float(r['T_K']),'.12g'))
        for x in bins.get(base,[]):
            g,h=float(r['gamma_inf']),float(x['gamma_inf'])
            if abs(g-h)/max(g,h)<=.005 and canon.compatible(condition_tuple(r),condition_tuple(x)):
                candidates.append({'canonical_record_id':x['canonical_record_id'],'Study_ID':x['Study_ID'],'gamma_inf':x['gamma_inf'],'Effective_Conditions':x['Effective_Conditions']})
        if candidates:evidence[r['canonical_record_id']]=candidates
    return evidence

def matched(records, relations):
    bins = defaultdict(list); pairs = []; unmatched = []
    for r in records:
        if r.get('Solute_Structure_Status')=='PENDING_AUDITED':
            unmatched.append({'canonical_record_id':r['canonical_record_id'], 'Reason':'原始溶质的具体构造或立体身份未确认；实验主表保留，暂不生成异构体配对'})
            continue
        if not canon.known_study(r):
            unmatched.append({'canonical_record_id':r['canonical_record_id'], 'Reason':'原研究未唯一确定'})
            continue
        bins[(r['Study_ID'],r['Solvent_Audit_ID'],format(float(r['T_K']),'.12g'),r['Family_ID'])].append(r)
    for base, rr in sorted(bins.items()):
        signatures = canon.maximal_signatures([condition_tuple(r) for r in rr])
        assigned = defaultdict(list)
        for r in rr:
            fits = [s for s in signatures if canon.compatible(condition_tuple(r),s)]
            if len(fits)!=1:
                unmatched.append({'canonical_record_id':r['canonical_record_id'], 'Reason':'同条件异构组存在多个实验方法或条件，缺报记录无法唯一对应'})
                continue
            assigned[fits[0]].append(r)
        for signature, group in sorted(assigned.items()):
            members = defaultdict(list)
            for r in group: members[r['Member_ID']].append(r)
            unique = []
            for member, points in sorted(members.items()):
                if len(points)!=1:
                    unmatched.extend({'canonical_record_id':r['canonical_record_id'], 'Reason':'同组同一构造成员存在多个测量点，无法唯一配对'} for r in points)
                else: unique.append(points[0])
            if len(unique)<2:
                unmatched.extend({'canonical_record_id':r['canonical_record_id'], 'Reason':'同研究及条件下缺少第二个构造异构成员'} for r in unique)
                continue
            gid = stable('MGV2_', [base,signature])
            for x,y in itertools.combinations(unique,2):
                key = tuple(sorted((x['Member_ID'],y['Member_ID'])))
                if key not in relations:
                    raise ValueError('Missing verified constitutional relation '+str(key))
                pairs.append(dict(
                    Pair_ID=stable('PAIRV2_', [gid,x['canonical_record_id'],y['canonical_record_id']]),
                    MatchedGroup_ID=gid, Family_ID=base[3], Study_ID=base[0],
                    Solvent_Audit_ID=base[1], Solvent_Mol_ID=x.get('Solvent_Mol_ID',''),
                    Solvent=x['Solvent'], T_K=x['T_K'],
                    Matched_Conditions=list(signature),
                    Condition_Signature_ID=stable('COND_',signature),
                    Isomer_A=x['Solute'], Isomer_B=y['Solute'],
                    canonical_record_id_A=x['canonical_record_id'],
                    canonical_record_id_B=y['canonical_record_id'],
                    gamma_A=x['gamma_inf'], gamma_B=y['gamma_inf'],
                    ln_gamma_A=x['ln_gamma_inf'],ln_gamma_B=y['ln_gamma_inf'],
                    D_AB=repr(float(x['ln_gamma_inf'])-float(y['ln_gamma_inf'])),
                    Isomer_Relation=relations[key],
                    Original_File_A=x['Original_Source_File'],Original_File_B=y['Original_Source_File'],
                    Original_Location_A=x['Original_Source_Location'],Original_Location_B=y['Original_Source_Location'],
                    Original_Link_A=x.get('Original_Source_Link',''),Original_Link_B=y.get('Original_Source_Link','')))
    return pairs, unmatched

def graphs_and_split(records,pairs,stage,assets,fixed_split_path=None):
    from graphs import save_graphs
    molecules=[]; indices={}; structure_cache={}
    for r in records:
        for role,col in [('solute','Solute_SMILES'),('solvent','Solvent_SMILES')]:
            original_smiles=r[col]
            if original_smiles not in structure_cache:
                m=Chem.MolFromSmiles(original_smiles)
                if m is None: raise ValueError('Invalid molecular graph '+r['canonical_record_id'])
                structure_cache[original_smiles]=(Chem.MolToInchiKey(m),Chem.MolToSmiles(m,isomericSmiles=True))
            key,sm=structure_cache[original_smiles]
            if role=='solute' and key[:14]!=r['Member_ID']:
                raise ValueError('Solute connectivity mismatch '+r['canonical_record_id'])
            if sm not in indices:
                indices[sm]=len(molecules)
                molecules.append({'Mol_ID':'MOL_'+key,'InChIKey':key,'SMILES':sm,'roles':role})
            elif role not in molecules[indices[sm]]['roles']:
                molecules[indices[sm]]['roles']+=';'+role
            r[role+'_mol_index']=indices[sm]
    out=stage/'data'
    write_csv(out/'molecules.csv',molecules)
    stats=save_graphs(molecules,out/'graphs.npz')
    folds=split(records,pairs,assets/'family_fold_manifest.json',out,fixed_split_path)
    return stats,folds

def split(records,pairs,manifest_path,out,fixed_split_path=None):
 old=json.loads(Path(manifest_path).read_text(encoding='utf8'));assignment=old['assignment']
 fams=sorted({r['Family_ID'] for r in records})
 if set(fams)-set(assignment):raise ValueError('Family absent from fixed outer manifest: '+str(sorted(set(fams)-set(assignment))))
 if len(fams)<15:raise ValueError('Five-fold validation requires at least 15 observed whole families')
 metrics={f:np.array([0,0,0,1.],np.float64) for f in fams};groups=defaultdict(set)
 for r in records:metrics[r['Family_ID']][0]+=1
 for p in pairs:metrics[p['Family_ID']][1]+=1;groups[p['Family_ID']].add(p['MatchedGroup_ID'])
 for f in fams:metrics[f][2]=len(groups[f])
 if fixed_split_path:
  fixed=json.loads(Path(fixed_split_path).read_text(encoding='utf8'))
  if fixed['outer_assignment']!=assignment:raise ValueError('Fixed complete model split disagrees with frozen outer family assignment')
  updated={};summary={}
  for rnd,d in fixed['rounds'].items():
   roles={role+'_families':list(d[role+'_families']) for role in ('train','validation','test')}
   sets=[set(roles[role+'_families']) for role in ('train','validation','test')]
   if any(sets[i]&sets[j] for i in range(3) for j in range(i+1,3)):raise ValueError('Frozen full split contains family leakage')
   if not set(fams)<=set.union(*sets):raise ValueError('Observed family absent from complete frozen split')
   if any(assignment[f]!='Fold_'+rnd for f in roles['test_families']):raise ValueError('Frozen test family differs from outer fold')
   for role in ('train','validation','test'):
    fs=set(roles[role+'_families']);observed=fs&set(fams)
    roles[role+'_counts']={'families':len(fs),'observed_families':len(observed),'task1':int(sum(metrics[f][0] for f in observed)), 'task2_pairs':int(sum(metrics[f][1] for f in observed)),'matched_groups':int(sum(metrics[f][2] for f in observed))}
   updated[rnd]=roles;summary[rnd]=roles['test_counts']
  fixed['rounds']=updated;fixed['complete_family_split_changed']=False
  fixed['outer_assignment_source']='_pipeline/assets/curation/family_fold_manifest.json'
  fixed['fixed_complete_split_sha256']=sha(fixed_split_path)
  save(Path(out)/'split_manifest.json',fixed)
  return summary
 rounds={};fold_summary={}
 for fold in range(1,6):
  test=[f for f in fams if assignment[f]==f'Fold_{fold}'];available=[f for f in fams if f not in test]
  # 10-14 entire validation families, ~15% of the non-test data.
  target=sum((metrics[f] for f in available),np.zeros(4))*.15
  target[3]=min(14,max(10,round(len(available)*.15)))
  rng=np.random.default_rng(20261007+fold);best=None;bs=float('inf')
  weights=np.array([6.,2.,1.,1.])
  def cost(s):return float(np.sum(weights*((sum((metrics[f] for f in sorted(s)),np.zeros(4))-target)/np.maximum(target,1))**2))
  for _ in range(200):
   chosen=set(rng.choice(available,size=int(target[3]),replace=False));score=cost(chosen)
   for _ in range(30):
    improve=False
    for f in sorted(chosen):
     for g in available:
      if g in chosen:continue
      new=(chosen-{f})|{g};c=cost(new)
      if c<score-1e-12:chosen=new;score=c;improve=True;break
     if improve:break
    if not improve:break
   if score<bs:bs=score;best=chosen
  roles={'train_families':sorted(set(available)-best),'validation_families':sorted(best),'test_families':test}
  for role in ('train','validation','test'):
   fs=set(roles[role+'_families']);roles[role+'_counts']={'families':len(fs),'task1':int(sum(metrics[f][0] for f in fs)),'task2_pairs':int(sum(metrics[f][1] for f in fs)),'matched_groups':int(sum(metrics[f][2] for f in fs))}
  rounds[str(fold)]=roles;fold_summary[str(fold)]=roles['test_counts']
 save(Path(out)/'split_manifest.json',{'seed':20261007,'outer_assignment':assignment,'outer_assignment_source':'_pipeline/assets/curation/family_fold_manifest.json','outer_assignment_changed':False,'validation_policy':'Each round 10-14 whole families from non-test folds; fixed before model training; aim ~15% of non-test Task1, pair/group/family counts; no target-value or relation-type stratification','rounds':rounds})
 return fold_summary
