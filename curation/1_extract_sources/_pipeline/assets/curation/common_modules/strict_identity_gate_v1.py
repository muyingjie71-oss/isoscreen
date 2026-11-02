"""Evidence-bound dual-InChI eligibility; experiments and canonical history stay intact.

This is a restriction of the already quality-screened molecular view. It does
not re-identify molecules, infer studies, promote old quality holds, merge
experiments, repair measurements, or require a second member in the same
source-specific family. A confirmed exact-value existing parent may support
the canonical experiment without validating an unresolved historical parent.
"""
from pathlib import Path
from collections import Counter
from functools import lru_cache
import csv,gzip,hashlib,json,math,sys
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

VERSION='STRICT_DUAL_INCHI_V1'
FOLDERS={'ThermoML':'01_ThermoML','ILThermo':'02_ILThermo','Brouwer':'03_Brouwer','DeepFM':'04_DeepFM','Tan':'05_Tan','Sherman':'06_Sherman','单独收集论文':'07_单独收集论文'}
FIELDS=['Strict_Identity_Version','Strict_Identity_Eligible','Strict_Identity_Reason_CN','Strict_Model_Eligible','Strict_Model_Reason_CN','Strict_Confirmed_Source','Strict_Confirmed_Source_Record_ID','Strict_Confirmed_Raw_Record_ID','Strict_Confirmed_Parent_Is_Previous_Representative','Strict_Solute_InChI','Strict_Solute_InChIKey','Strict_Solute_SMILES','Strict_Solvent_InChI','Strict_Solvent_InChIKey','Strict_Solvent_SMILES','Strict_Solute_Evidence','Strict_Solvent_Evidence','Strict_Identity_Decisions_SHA256','Strict_Original_Representative_Source','Strict_Original_Selected_Input_Record_ID']
FIELDS+=['Strict_Graph_Equivalence_CN','Strict_Physical_Graph_Override_CN','Strict_Previous_Solvent_SMILES','Strict_Previous_Solvent_Identity','Strict_Previous_Solvent_Mol_ID']

def sha(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def yes(v):return str(v).strip().casefold() in ('yes','true','1','是')

def list_value(v):
 if isinstance(v,list):return v
 return json.loads(v) if v else []

def source_name(r):return r.get('raw_source') or next((k for k,v in FOLDERS.items() if v==r.get('source')),r.get('source',''))

def load(csv_path,manifest_path):
 csv_path=Path(csv_path);manifest_path=Path(manifest_path)
 meta=json.loads(manifest_path.read_text(encoding='utf-8'));digest=sha(csv_path)
 if meta.get('decision_csv_sha256')!=digest:raise ValueError('Strict identity decision CSV does not match its locked manifest')
 decisions={}
 keep={'source','raw_source','record_id','keep_model','reason_zh','source_qc_exclude','solute_confirmed','solvent_confirmed','gamma_inf','T_K','frozen_raw_record_id','frozen_gamma_inf','frozen_T_K','frozen_gamma_inf_hex','frozen_T_K_hex','numeric_representation_review','physical_graph_override_reason'}
 keep.update(role+'_'+k for role in ('solute','solvent')for k in ('inchi','key','smiles','formula','evidence'))
 csv.field_size_limit(10*1024*1024)
 opener=gzip.open if csv_path.suffix=='.gz'else open
 with opener(csv_path,'rt',encoding='utf-8-sig',newline='') as f:
  for row in csv.DictReader(f):
   key=(source_name(row),row['record_id'])
   if key in decisions:raise ValueError('Duplicate strict identity source record '+repr(key))
   decisions[key]={k:sys.intern(v) for k,v in row.items()if k in keep}
 return {'records':decisions,'metadata':meta,'sha256':digest,'manifest_sha256':sha(manifest_path)}

@lru_cache(maxsize=20000)
def verified_structure(inchi,key,smiles,formula):
 if not inchi or not inchi.startswith('InChI=') or not key or not smiles:raise ValueError('缺少完整InChI、完整InChIKey或SMILES')
 m=Chem.MolFromSmiles(smiles)
 if m is None:raise ValueError('SMILES不能解析为具体分子结构')
 if Chem.InchiToInchiKey(inchi)!=key or Chem.MolToInchiKey(m)!=key:raise ValueError('InChI、完整InChIKey与SMILES不一致')
 if formula and rdMolDescriptors.CalcMolFormula(m)!=formula:raise ValueError('分子式与具体结构不一致')
 return key,Chem.MolToSmiles(m,isomericSmiles=True)

@lru_cache(maxsize=20000)
def current_structure(smiles):
 m=Chem.MolFromSmiles(smiles) if smiles else None
 return (Chem.MolToInchiKey(m),Chem.MolToSmiles(m,isomericSmiles=True)) if m is not None else ('','')

@lru_cache(maxsize=20000)
def resonance_set(smiles):
 m=Chem.MolFromSmiles(smiles)
 if m is None:return frozenset()
 supplier=Chem.ResonanceMolSupplier(m,flags=Chem.ResonanceFlags.KEKULE_ALL,maxStructs=64)
 values=frozenset(Chem.MolToSmiles(x,isomericSmiles=True)for x in supplier)
 return values if len(supplier)<64 else frozenset()

def equivalent_physical_graph(current_smiles,approved_smiles):
 a,b=current_structure(current_smiles),current_structure(approved_smiles)
 if a==b:return True,'完全相同的具体分子图'
 if a[0]and a[0]==b[0]:
  aa,bb=resonance_set(a[1]),resonance_set(b[1])
  if aa and aa==bb:return True,'完整InChIKey一致且穷举共振集合相同；电荷/键的共振写法不同，原模型物理图保留'
 return False,''

def approved_payload(d):
 if not yes(d.get('keep_model')) or yes(d.get('source_qc_exclude')):return None,d.get('reason_zh') or '原名称、CAS、立体构型或离子组成尚未可靠确认'
 p={}
 try:
  for role in ('solute','solvent'):
   key,smiles=verified_structure(*(d.get(role+'_'+k,'') for k in ('inchi','key','smiles','formula')))
   if not d.get(role+'_evidence'):raise ValueError('缺少'+('溶质' if role=='solute'else'溶剂')+'身份核查依据')
   p[role]={'inchi':d[role+'_inchi'],'key':key,'smiles':smiles,'evidence':d[role+'_evidence']}
 except ValueError as exc:return None,'统一身份审核载荷不完整：'+str(exc)
 return p,''

def numeric_exact(a,b):
 try:
  x,y=float(a),float(b)
  return math.isfinite(x) and math.isfinite(y) and x.hex()==y.hex()
 except (ValueError,TypeError):return False

def evaluate(record,evidence,raw_bank,allow_audited_graph_override=False):
 """Find an approved exact-value EXISTING parent with the same physical graphs."""
 selected=record.get('selected_input_record_id','');parents=list_value(record.get('alias_parent_record_ids'))
 if selected and selected not in parents:raise ValueError('Canonical representative absent from original parent set '+record['canonical_record_id'])
 ordered=([selected] if selected else [])+sorted(p for p in parents if p!=selected)
 failures=[]
 for parent_id in ordered:
  raw=raw_bank.get(parent_id)
  if raw is None:raise ValueError('Frozen canonical parent absent from original bank '+parent_id)
  key=(raw['Source'],raw['original_record_id']);d=evidence['records'].get(key)
  if d is None:
   failures.append('原记录未进入七源统一身份审核：'+key[0]+'::'+key[1]);continue
  if not numeric_exact(raw['gamma_inf'],record['gamma_inf']) or not numeric_exact(raw['T_K'],record['T_K']):
   failures.append('已审父记录与当前canonical的γ∞或温度并非完全同值，不能用舍入近值替换原目标');continue
  frozen_id=d.get('frozen_raw_record_id','')
  if frozen_id and frozen_id!=parent_id:raise ValueError('Strict identity manifest points to another frozen parent '+repr(key))
  source_gamma=d.get('frozen_gamma_inf')or d.get('gamma_inf')
  source_T=d.get('frozen_T_K')or d.get('T_K')
  if not numeric_exact(source_gamma,raw['gamma_inf']) or not numeric_exact(source_T,raw['T_K']):
   raise ValueError('Strict identity manifest scientific value differs from frozen source parent '+repr(key))
  p,why=approved_payload(d)
  if not p:
   failures.append(key[0]+'::'+key[1]+'：'+why);continue
  mismatch=[];equivalence=[];override=None
  for role,col in (('solute','Solute_SMILES'),('solvent','Solvent_SMILES')):
   current=current_structure(record.get(col,''))
   eq,basis=equivalent_physical_graph(record.get(col,''),p[role]['smiles'])
   if eq:
    if current[1]!=p[role]['smiles']:equivalence.append(('溶质'if role=='solute'else'溶剂')+'：'+basis)
   elif allow_audited_graph_override and role=='solvent'and key[0]=='ILThermo':
    name=raw.get('Solvent','').casefold();scope={
     '1-ethyl-3-methylimidazolium 4-methylbenzenesulfonate':('XLTJXCKTLAOWHR-UHFFFAOYSA-M','HXMUPILCYSJMLQ-UHFFFAOYSA-M','原ILThermo JSON原组分明确为tosylate，旧图误配含羧酸根/巯基阴离子，纠正为对甲苯磺酸根'),
     '1-ethyl-3-methyl-1h-imidazolium tetrachloroaluminate':('UYYXEZMYUOVMPT-UHFFFAOYSA-J','UYYXEZMYUOVMPT-UHFFFAOYSA-J','原研究明确单一AlCl4−盐；区别原文DEG混合物，纠正旧Al3+与四个Cl−解离写法为四氯铝酸根'),
     '1-(4-sulfobutyl)-3-methylimidazolium bis(trifluoromethanesulfonyl)imide':('PXYKADKIWNXJNF-UHFFFAOYSA-O','PXYKADKIWNXJNF-UHFFFAOYSA-O','原研究明确SO3HBMIM，保留SO3H与芳香咪唑鎓+图，纠正旧额外NH+/SO3−写法')}
    item=scope.get(name)
    if item and current[0]==item[0]and p[role]['key']==item[1]and d.get('physical_graph_override_reason'):
     override={'smiles':p[role]['smiles'],'key':p[role]['key'],'name':name,'reason':item[2]+'。仅按该原file/component/名称/分子式及出版方证据绑定更正物理图；原γ∞、T、canonical父集合、family及历史matched-group标识不改变。'+d['physical_graph_override_reason']+'；具体原记录依据：'+p[role]['evidence']}
    else:mismatch.append('溶剂')
   else:mismatch.append('溶质'if role=='solute'else'溶剂')
  if mismatch:
   failures.append('已确认'+'、'.join(mismatch)+'完整身份或物理图与当前canonical结构不一致，需另行审核，不能仅按同分子式替换');continue
  if p['solute']['key'][:14]!=record['Member_ID'] or p['solute']['key']!=record['Solute_Identity']:
   failures.append('已确认溶质的完整构造/立体身份与当前成员标识不一致');continue
  return {'eligible':True,'payload':p,'parent_id':parent_id,'source':key[0],'record_id':key[1],'reason':'溶质与溶剂的完整InChI、具体连接/立体及离子身份已按原记录确认；原γ∞、温度和canonical父集合保留。','selected':parent_id==selected,'graph_equivalence':'；'.join(equivalence),'graph_override':override}
 return {'eligible':False,'reason':'；'.join(dict.fromkeys(failures)) or '不存在可用于结构模型的完整确认原记录','selected':False}

def annotate(record,result,evidence):
 record.update(Strict_Identity_Version=VERSION,Strict_Identity_Eligible='是'if result['eligible']else'否',Strict_Identity_Reason_CN=result['reason'],Strict_Identity_Decisions_SHA256=evidence['sha256'],Strict_Original_Representative_Source=record.get('Source',''),Strict_Original_Selected_Input_Record_ID=record.get('selected_input_record_id',''))
 if result['eligible']:
  record['Strict_Graph_Equivalence_CN']=result.get('graph_equivalence','')
  override=result.get('graph_override')
  if override:
   record.update(Strict_Previous_Solvent_SMILES=record.get('Solvent_SMILES',''),Strict_Previous_Solvent_Identity=record.get('Solvent_Identity',''),Strict_Previous_Solvent_Mol_ID=record.get('Solvent_Mol_ID',''),Strict_Physical_Graph_Override_CN=override['reason'])
   record.update(Solvent_SMILES=override['smiles'],Solvent_Identity=override['key'],Solvent_Mol_ID='MOL_'+override['key'])
  record.update(Strict_Confirmed_Source=result['source'],Strict_Confirmed_Source_Record_ID=result['record_id'],Strict_Confirmed_Raw_Record_ID=result['parent_id'],Strict_Confirmed_Parent_Is_Previous_Representative='是'if result['selected']else'否')
  for role in ('solute','solvent'):
   p=result['payload'][role];prefix='Strict_'+role.capitalize()+'_'
   record.update({prefix+'InChI':p['inchi'],prefix+'InChIKey':p['key'],prefix+'SMILES':p['smiles'],prefix+'Evidence':p['evidence']})
 return record

def apply(master,previous_eligible,evidence,raw_bank):
 previous_ids={r['canonical_record_id']for r in previous_eligible};retained=[];decisions=[];audit=[];counts=Counter();source_counts={}
 for r in master:
  cid=r['canonical_record_id'];previous=cid in previous_ids
  # Structure-pending/quality-held master rows may lack physical graphs. They
  # remain excluded by the older rule and are never promoted here.
  result=evaluate(r,evidence,raw_bank,allow_audited_graph_override=previous)
  annotate(r,result,evidence);counts['experimental_master_records']+=1
  if result['eligible']:counts['dual_identity_confirmed_master_records']+=1
  if result.get('graph_override'):
   counts['audited_source_physical_graph_updates']+=1
   if result['graph_override']['name']=='1-ethyl-3-methylimidazolium 4-methylbenzenesulfonate':counts['audited_tosylate_physical_graph_updates']+=1
  if result.get('graph_equivalence'):counts['confirmed_resonance_representation_equivalences']+=1
  r['Strict_Model_Eligible']='是'if previous and result['eligible']else'否'
  r['Strict_Model_Reason_CN']=result['reason']if previous else r.get('Model_Eligibility_Reason')or'既有来源、温度QC、结构或出处条件未通过；本轮不放宽原筛选'
  if previous:
   source_counts.setdefault(r['Source'],Counter())['previous_model']+=1
   if result['eligible']:
    retained.append(r);counts['retained_model_records']+=1;source_counts[r['Source']]['retained_model']+=1
    if not result['selected']:counts['confirmed_equal_parent_supports_original_canonical']+=1
   else:
    r['Model_Eligible']='否';r['Model_Eligibility_Reason']=result['reason'];counts['newly_held_model_records']+=1;source_counts[r['Source']]['newly_held_model']+=1
    decisions.append(dict(r,Action='HOLD_MODEL_STRICT_DUAL_INCHI',Reason=result['reason']))
  audit.append({k:r.get(k,'')for k in ['canonical_record_id','Family_ID','Member_ID','Source','Solute','Solvent','T_K','gamma_inf','original_record_ids','alias_parent_record_ids','Original_Source_File','Original_Source_Location','Original_Data_Target',*FIELDS]})
 counts['previous_model_records']=len(previous_ids)
 if counts['retained_model_records']+counts['newly_held_model_records']!=len(previous_ids):raise ValueError('Previous molecular scope lost a canonical ID')
 return {'eligible':retained,'decisions':decisions,'audit':audit,'stats':dict(counts),'by_source':{k:dict(v)for k,v in source_counts.items()}}
