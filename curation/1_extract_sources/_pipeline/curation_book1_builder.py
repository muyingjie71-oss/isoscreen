"""The merged canonical experiments and complete record-level audit."""
from pathlib import Path
from collections import Counter,defaultdict
import csv,json,math,gzip
from .curation_book2_builder import original_link,native,col,load_rows,load_json

NAME='01_MERGED_DEDUPLICATED'
ACTION_CN={'KEEP':'保留代表原值','MERGE_DUPLICATE':'重复归并','HOLD_NUMERIC_CONFLICT':'冲突整组排除','HOLD_AMBIGUOUS_CONDITION':'实验条件待核','HOLD_INVALID_INPUT':'原值无效待核'}
META=[('canonical_record_id','canonical_record_id'),('Family_ID','家族ID'),('Member_ID','成员ID'),('Study_ID','原研究ID'),('Original_DOI','原始DOI'),('Reference','原始文献'),('Solute_InChI','有效溶质InChI'),('Solvent_InChI','有效溶剂InChI'),('Solute_Identity','有效溶质InChIKey'),('Solvent_Identity','有效溶剂InChIKey'),('Solute_SMILES','有效溶质SMILES'),('Solvent_SMILES','有效溶剂SMILES'),('source_list','来源列表'),('original_record_ids','原记录ID列表'),('original_record_keys','原记录键列表'),('alias_parent_record_ids','全部父记录ID'),('selected_input_record_id','保留代表原记录ID'),('Original_Source_File','原始文件'),('Original_Source_Location','原始具体位置'),('Raw_实验方法','原始实验方法'),('Raw_压力_kPa','原始压力_kPa'),('Raw_相态','原始相态'),('Raw_标准态','原始标准态'),('Raw_其他固定条件','其他固定条件'),('Effective_Conditions','确认实验条件'),('Reported_Condition_Declaration','原报实验条件声明'),('Condition_History','各来源实验条件历史'),('Source03_Gamma_XML','γ∞原始数值文本'),('Source03_T_XML','温度原始数值文本'),('QC_Status','温度质量状态'),('QC_Reason','温度质量原因'),('Duplicate_Class','重复类型'),('Fold_ID','固定Fold_ID')]

def row_state(r):
 if r.get('Model_Eligible')=='否':return 'EXCLUDE_QC' if r.get('QC_Status','').startswith('EXCLUDE') else 'HOLD_MODEL'
 return ''
def metadata(r):return [r.get(k,'') for k,_ in META]
def measured(r,which):return native(r.get('Source03_'+('T' if which=='T_K' else 'Gamma')+'_XML') or r[which])
def status(r):return '可建模' if r.get('Model_Eligible')=='是' else '排除：温度异常' if r.get('QC_Status','').startswith('EXCLUDE') else '暂缓建模'

def build_book1_descriptor(bank,directory):
 directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
 fields={k for k,_ in META}|{'Solute','Solvent','Source','T_K','gamma_inf','source_count','Model_Eligible','Model_Eligibility_Reason','Action','Reason','Reason_CN'}
 canonical=load_rows(bank['canonical'],fields);decisions=load_rows(bank['primary_decisions'],fields);holds=load_rows(bank['model_holds'],fields);report=load_json(bank['report']);sheets=[]
 def sheet(name,columns,rows):
  path=directory/(name+'.jsonl');samples=[];count=0
  with path.open('w',encoding='utf-8') as out:
   for row in rows:
    if len(row['values'])!=len(columns):raise ValueError('Column count '+name)
    out.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');count+=1
    if len(samples)<8:samples.append(row)
  sheets.append({'name':name,'columns':columns,'file':str(path),'rows':count,'samples':samples,'header_rows':1})
 columns=[col('溶质',22),col('溶剂',25),col('温度_K',11,'temperature'),col('γ∞（实验）',15,'gamma'),col('来源数',9,'integer'),col('模型资格',17),col('暂缓或排除原因',34,'reason'),col('原始数据',12,'link')]+[col(label,26,hidden=True) for _,label in META]
 sheet('保留实验',columns,({'values':[r['Solute'],r['Solvent'],measured(r,'T_K'),measured(r,'gamma_inf'),int(r['source_count']),status(r),'' if r.get('Model_Eligible')=='是' else r.get('Model_Eligibility_Reason',''),original_link(r)]+metadata(r),'row_status':row_state(r),'critical_columns':[4]} for r in canonical))
 actions=Counter(r['Action'] for r in decisions);by_source=defaultdict(Counter)
 for r in decisions:by_source[r['Source']][r['Action']]+=1
 summary_cols=[col('来源',23),col('03最终输入',17,'integer'),col('保留代表',17,'integer'),col('重复归并',17,'integer'),col('冲突排除',17,'integer'),col('条件待核',17,'integer'),col('无效原值',17,'integer')]
 sheet('来源汇总',summary_cols,({'values':[source,sum(c.values()),c['KEEP'],c['MERGE_DUPLICATE'],c['HOLD_NUMERIC_CONFLICT'],c['HOLD_AMBIGUOUS_CONDITION'],c['HOLD_INVALID_INPUT']]} for source,c in sorted(by_source.items())))
 audit_cols=[col('来源',15),col('溶质',22),col('溶剂',25),col('温度_K',11,'temperature'),col('γ∞（原值）',15,'gamma'),col('本轮处理',20),col('具体依据',44,'reason'),col('原始数据',12,'link')]+[col('原记录ID',26,hidden=True),col('归属canonical_record_id',26,hidden=True)]+[col(label,26,hidden=True) for _,label in META]
 mapping=load_json(bank['mapping'])
 sheet('去重与冲突审计',audit_cols,({'values':[r['Source'],r['Solute'],r['Solvent'],measured(r,'T_K'),measured(r,'gamma_inf'),ACTION_CN[r['Action']],r['Reason_CN'],original_link(r),r['canonical_record_id'],mapping.get(r['canonical_record_id'],'')]+metadata(r),'row_status':'EXCLUDE_CONFLICT' if r['Action']=='HOLD_NUMERIC_CONFLICT' else 'HOLD_CONDITION' if r['Action'].startswith('HOLD') else '', 'critical_columns':[5]} for r in decisions))
 hold_cols=[col('溶质',22),col('溶剂',25),col('温度_K',11,'temperature'),col('γ∞（原值）',15,'gamma'),col('建模处理',22),col('具体原因',49,'reason'),col('原始数据',12,'link')]+[col(label,26,hidden=True) for _,label in META]
 cn_hold={'EXCLUDE_QC':'温度异常排除','HOLD_MODEL_POTENTIAL_SOURCE_ESTIMATE':'实测出处待核','HOLD_MODEL_POTENTIAL_REPEAT':'原研究与转载关系待核','HOLD_MODEL_AUDITED_SCOPE':'固定家族范围待核'}
 sheet('模型待核与排除',hold_cols,({'values':[r['Solute'],r['Solvent'],measured(r,'T_K'),measured(r,'gamma_inf'),cn_hold.get(r['Action'],r['Action']),r['Reason'],original_link(r)]+metadata(r),'row_status':r['Action'],'critical_columns':[4]} for r in holds))
 n=int(report['input_raw_observations']);kept=len(canonical);excluded=sum(actions[a] for a in ('HOLD_NUMERIC_CONFLICT','HOLD_AMBIGUOUS_CONDITION','HOLD_INVALID_INPUT'));merged=actions['MERGE_DUPLICATE'];model=int(report['task1'])
 if n!=len(decisions) or n!=kept+merged+excluded or len(holds)+model!=kept:raise AssertionError('Canonical accounting mismatch')
 flow={'title':'七份来源合并、去重与冲突审计','origin':'七份当前03_ISOMERS的最终保留实验；每点溯源到各来源00_Source，原始γ∞与温度不改。','stages':[
  {'label':'七份表三保留实验（尚未跨源去重）','count':n,'detail':f'输入{n:,}条；各来源内部筛选通过，尚需统一去重与建模审查。'},
  {'label':'核对同一原研究及实验条件','count':n,'detail':'同原研究、完整溶质/溶剂结构、同温度；实验方法、压力等条件唯一兼容。','branch':{'label':'同研究重复副本归并','count':merged,'detail':'全组γ∞跨度≤0.5%；文献明确者优先，再按固定数据库优先级；原值与来源历史全部保留。'}},
  {'label':'去重／冲突审查后的审计实验','count':kept,'detail':f'{kept:,}条保留审计，含{len(holds):,}条暂缓；不是全部进入训练。','branch':{'label':'冲突或条件不明待核','count':excluded,'detail':f"冲突{actions['HOLD_NUMERIC_CONFLICT']:,}条；条件不唯一{actions['HOLD_AMBIGUOUS_CONDITION']:,}条；无效原值{actions['HOLD_INVALID_INPUT']:,}条。均留审计，不进入保留实验。"}},
  {'label':'最终实验训练资格通过','count':model,'detail':f'{model:,}条进入单点数据；实验配对还需同研究、同溶剂及同温度。','branch':{'label':'仅留审计，不进单点／配对','count':len(holds),'detail':f"{report['model_hold_counts'].get('HOLD_MODEL_POTENTIAL_REPEAT',0):,}条转载关系待核；{report['model_hold_counts'].get('HOLD_MODEL_POTENTIAL_SOURCE_ESTIMATE',0):,}条实测出处待核。原值与原因保留。"}}],
  'footer':f"{n:,} − {merged:,}重复副本 − {actions['HOLD_NUMERIC_CONFLICT']:,}冲突 − {actions['HOLD_AMBIGUOUS_CONDITION']:,}条件待核 − {actions['HOLD_INVALID_INPUT']:,}无效原值 = {kept:,}条审计实验；再暂缓{len(holds):,}条 = {model:,}条训练单点。每步条数按同一记录口径核对。"}
 result={'version':'CURATION_TWO_BOOKS_BOOK1_V1','source_folder':'08_Dataset_Curation','source_name':'七份来源合并','kind':NAME,'title':flow['title'],'flow':flow,'sheets':sheets,'science_report':bank['report'],'counts':{'input':n,'canonical':kept,'merged':merged,'excluded':excluded,'model_task1':model,'model_holds':len(holds)}}
 path=directory/'bundle.json';path.write_text(json.dumps(result,ensure_ascii=False,indent=2,default=str),encoding='utf-8');return path
