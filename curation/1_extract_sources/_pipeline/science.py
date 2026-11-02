"""Pure reconstruction from raw sources and reviewed scientific assets.

No workbook mutation, online structure lookup, or finished-workbook import is
performed here. Workbook formatting is not part of this code.
"""
from __future__ import annotations
from pathlib import Path
from collections import Counter
from types import SimpleNamespace
import csv,hashlib,importlib,json,math,re,types,sqlite3,gzip
import urllib.parse
from decimal import Decimal
from .context import SourceContext
from .bundles import RowStore,materialize_sheet,materialize_split_table,save_manifest

UNIFIED_HEADERS=['当前建模筛选结果','排除或待核原因','有效溶质InChI','有效溶质InChIKey','有效溶质SMILES','有效溶质分子式','有效溶剂InChI','有效溶剂InChIKey','有效溶剂SMILES','有效溶剂分子式','有效溶质立体核对','有效溶剂立体核对','有效溶剂阳离子SMILES','有效溶剂阴离子SMILES','已核具体结构说明','溶质身份依据','溶剂身份依据','原始数据','原始数据具体位置']
def native_value(row,field):
    if field=='当前建模筛选结果':return'可建模'if row.get('Model_Eligibility')=='PASS'else'排除'
    token=row.get('_published_number_xml',{}).get(field)
    if token is not None:return {'n':str(token)}
    return row.get(field,'')
def source_origin(ctx):
    names={'thermoml':'00_Source/ThermoML.v2020-09-30内原始ThermoML JSON物性数据块与实验点。','ilthermo':'00_Source/ILThermo.Activity.source内冻结NIST /ilset JSON、set ID清单与大小写碰撞核验。','brouwer':'00_Source/Brouwer_IDAC_Source.xlsx的Raw Data实验点与Corrections逐点更正；Data Treatment拟合结果不作为实验。','deepfm':'00_Source/DeepFM_IDAC_Source.xlsx，Table S1实验H/I列；S2/S3/S4提供原研究和结构ID，预测J/K不进入数据。','tan':'00_Source/Tan_2022_Supporting_Information.pdf的Table S1身份和Table S2实验lnγ；按原CID核对结构，γ=exp(实验lnγ)。','sherman':'00_Source/Sherman_1996_Article.pdf，Table 5（PDF第7–9页）326条已审逐点转录，核对原PDF哈希后筛选直接实验。','papers':'00_Source内逐篇PDF、HTML和XLSX原始实验；verified_experimental_records.csv是绑定这些原文件哈希的已审逐点转录。'}
    return names[ctx.module_name]
def source_screen_details(ctx,stats):
    if ctx.module_name=='sherman':return'排除：'+str(stats.get('exclude_INVSOL',195))+'条由溶解度反演INVSOL、'+str(stats.get('exclude_KHW',13))+'条KHW派生值、'+str(stats.get('exclude_interpolated',7))+'条插值；原始打印值与方法依据保留在来源审核资产。'
    if ctx.module_name=='thermoml':return'仅保留原文明确γ∞或无穷稀释条件的活度系数；混合溶剂排除'+str(stats.get('idac_mixed_solvent_skipped',0))+'条，无溶剂排除'+str(stats.get('idac_no_solvent_skipped',0))+'条，非数值排除'+str(stats.get('idac_non_numeric_skipped',0))+'条。已确认SOURCE_ANOMALY留在02并高亮。'
    if ctx.module_name=='ilthermo':
        reasons=stats.get('reason_counts',{});labels={'mixed_solvent_excluded':'混合溶剂','not_activity_coefficient':'非活度系数物性','target_not_at_infinite_dilution':'目标组分摩尔分数高于无限稀释阈值','no_target_mole_fraction':'缺少目标组分摩尔分数','confirmed':'明确无限稀释实验'}
        return'仅直接报告活度系数、明确目标摩尔分数0至10⁻⁵、单溶剂。'+ '；'.join(labels.get(k,k)+' '+str(v)+'条'for k,v in reasons.items())+'。'
    if ctx.module_name=='brouwer':return'Raw Data实验展开与更正范围硬校验；混合/非单溶剂排除'+str(stats.get('mixed_excluded',0))+'条，无效γ或温度排除'+str(stats.get('invalid_excluded',0))+'条。结构待核不会判作错误实验删除。'
    if ctx.module_name=='deepfm':return'实验γ(H)与实验lnγ(I)一致性核验；预测J/K排除，混合/未知溶剂排除'+str(stats.get('mixed_excluded',0))+'条。'
    if ctx.module_name=='tan':return'仅Table S2实验lnγ，原表预测及平方误差不作为实验；CID/身份/点数/温度范围和重复核验通过。'
    return'核对原文件及已审转录哈希、唯一记录ID、正γ和正温度、单溶剂；不引入预测、插补或推测修正值。'
class SourceRowStore(RowStore):
    def append(self,row):
        # Keep the actually parsed raw values. Reviewed output numeric XML is
        # separate metadata applied only at the authoring boundary.
        super().append(row)
    write_row=append
class DecisionStore:
    """Streaming frozen per-record decisions; large manifests stay on disk."""
    def __init__(self,path,ctx,directory):
        self.db=sqlite3.connect(str(Path(directory)/'identity_decisions.sqlite'))
        self.db.execute('DROP TABLE IF EXISTS decisions');self.db.execute('CREATE TABLE decisions(rid TEXT PRIMARY KEY,data TEXT)')
        opener=gzip.open if str(path).endswith('.gz')else open
        with opener(path,'rt',encoding='utf8')as stream:
            for line in stream:
                if not line.strip():continue
                row=json.loads(line)
                if row.get('source',ctx.source_folder)!=ctx.source_folder:continue
                rid=row.get('record_id')or row.get('IDAC记录ID')or row.get('rid')
                data=row.get('decision',row)
                if not rid:raise ValueError('Decision line has no record ID')
                self.db.execute('INSERT INTO decisions VALUES(?,?)',(rid,json.dumps(data,ensure_ascii=False,separators=(',',':'))))
        self.db.commit()
    def get(self,rid,default=None):
        r=self.db.execute('SELECT data FROM decisions WHERE rid=?',(rid,)).fetchone();return json.loads(r[0])if r else default
    def close(self):self.db.close()
def sha(path):
    with Path(path).open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
def ordered_isomer_rows(store,directory):
    """Sort the reading view through a small offset index, not a row list."""
    store.close();db=sqlite3.connect(str(Path(directory)/'reading_sort.sqlite'))
    db.execute('DROP TABLE IF EXISTS offsets');db.execute('CREATE TABLE offsets(fid TEXT,member TEXT,solvent TEXT,temp REAL,rid TEXT,pos INTEGER)')
    with store.path.open('rb')as src:
        while True:
            pos=src.tell();line=src.readline()
            if not line:break
            row=json.loads(line);db.execute('INSERT INTO offsets VALUES(?,?,?,?,?,?)',(row['异构体家族ID'],row['当前异构体名称'].casefold(),row['溶剂'].casefold(),float(row['温度_K']),row['IDAC记录ID'],pos))
    db.commit()
    try:
        with store.path.open('rb')as src:
            for(pos,)in db.execute('SELECT pos FROM offsets ORDER BY fid,solvent,member,temp,rid'):
                src.seek(pos);yield json.loads(src.readline())
    finally:db.close()
def parser_module(ctx):
    m=importlib.import_module('.sources.'+ctx.module_name,__package__)
    # Parsers have no writer. Globals refer explicitly to their asset/raw roots.
    m.ROOT=ctx.asset_dir;m.SCRIPT_DIR=ctx.source_dir
    if ctx.module_name=='sherman':m.SOURCE=ctx.raw_dir/'Sherman_1996_Article.pdf'
    if ctx.module_name=='papers':m.ROOT=ctx.source_dir
    return m
def supplement_registry(ctx):
    p=ctx.asset_dir/'audited_structure_registry.json'
    if not p.exists():return None,None
    reg=json.loads(p.read_text(encoding='utf8'));reg['project_root']=str(ctx.project_root)
    m=importlib.import_module('.sources.'+ctx.module_name+'_identity',__package__)
    for group in ('solutes','solvents'):
        for entry in reg.get(group,{}).values():
            if hasattr(m,'verified_payload'):m.verified_payload(entry)
    return reg,m
def selection_records(selection,ctx,directory=None):
    if selection is None:return None
    if isinstance(selection,(str,Path)):
        p=Path(selection)
        if '.jsonl'in p.name:
            if directory is None:raise ValueError('Streaming decisions need a work directory')
            return DecisionStore(p,ctx,directory)
        if p.suffix.lower()=='.csv':
            with p.open(encoding='utf-8-sig',newline='')as f:
                return {r['record_id']:r for r in csv.DictReader(f)if r['source']==ctx.source_folder}
        selection=json.loads(p.read_text(encoding='utf8'))
    if 'records'in selection:
        r=selection['records'];return r.get(ctx.source_folder,r if all(isinstance(v,dict)and'keep_model'in v for v in r.values())else{})
    return selection
def reviewed_supplements(ctx,row,reg,identity):
    if not reg:return {}
    shadow=dict(row)
    for h,token in row.get('_source02_number_xml',row.get('_published_number_xml',{})).items():
        if h in ('发表年份','溶剂组分数','温度_K','压力_kPa','无限稀释活度系数','活度系数不确定度','稀释值'):
            shadow[h]=float(token)
    return identity.supplements_for_row(shadow,reg)
def native_locator(link):
    """Excel location contains only a worksheet/cell reference, never prose."""
    out=dict(link);loc=str(out.get('location',''))
    target=urllib.parse.unquote(str(out.get('target','')))
    if target.startswith('['):
        end=target.index(']');loc=loc or target[end+1:].lstrip('#');target=target[1:end]
    if target.lower().startswith('file:///'):target=target[8:]
    out['target']=target;out['location']=loc
    if target and not target.split('#',1)[0].lower().endswith(('.xlsx','.xls')):
        out['location']='';return out
    if loc:
        match=re.match(r"^(.+?)!((?:\$?[A-Za-z]{1,3}\$?\d+)(?::\$?[A-Za-z]{1,3}\$?\d+)?)",loc)
        if not match:raise ValueError('Unusable Excel source location: '+loc)
        sheet=match[1].strip().strip("'").replace("''", "'").replace("'", "''")
        out['location']="'"+sheet+"'!"+match[2]
    return out
def raw_locator(ctx,row,reg=None):
    rid=str(row.get('IDAC记录ID')or row.get('数据点ID')or row.get('记录ID')or'');value=str(row.get('原始JSON文件')or'')
    target='';location='';pointer='';label='原始数据'
    if ctx.module_name=='thermoml':
        target=str(ctx.raw_dir/'ThermoML.v2020-09-30'/value)
        point=re.search(r'::POINT(\d+)',rid);pv=re.search(r'::PV(\d+)',rid)
        pointer=f'数据块{row.get("数据块编号","")} / NumValues[{int(point[1])-1}]'if point else'来源JSON'
        if pv:pointer+=f' / PropertyValue[{int(pv[1])-1}] / nPropValue'
    elif ctx.module_name=='ilthermo':
        target=str(ctx.raw_dir/'ILThermo.Activity.source'/value);point=re.search(r'::POINT(\d+)',rid)
        binding=(reg or{}).get('file_bindings',{}).get(value,{})
        index=binding.get('gamma_column')
        pointer='/data/'+str(int(point[1])-1)+('/'+str(index)+'/0'if index is not None else'')if point else'来源JSON'
    elif ctx.module_name=='brouwer':
        target=str(ctx.raw_dir/'Brouwer_IDAC_Source.xlsx');location="'Raw Data'!"+rid.split('!',1)[1];pointer=location
    elif ctx.module_name=='deepfm':
        excelrow=re.search(r'S1ROW(\d+)',rid);target=str(ctx.raw_dir/'DeepFM_IDAC_Source.xlsx');location="'Table S1'!H"+excelrow[1];pointer=location
    elif ctx.module_name=='tan':
        p=re.search(r'#page=(\d+)',value);target=str(ctx.raw_dir/'Tan_2022_Supporting_Information.pdf')+('#page='+p[1]if p else'');pointer='Table S2；'+str(row.get('数据块编号',''))
    elif ctx.module_name=='sherman':
        page=re.search(r'(?:page=|page\s*)(\d+)',value,re.I)or re.search(r'PDFp(\d+)',str(row.get('数据块编号','')))or re.search(r'PDF_page=(\d+)',str(row.get('其他固定条件','')))
        target=str(ctx.raw_dir/'Sherman_1996_Article.pdf')+('#page='+page[1]if page else'');pointer=value+'；'+str(row.get('数据块编号',''))
    elif ctx.module_name=='papers':
        relative=value.split('#',1)[0].split('::',1)[0];target=str(ctx.source_dir/relative)
        if'#page='in value:target+='#page='+value.split('#page=',1)[1].split(';',1)[0]
        elif'::'in value:location=value.split('::',1)[1]
        pointer=value+'；'+str(row.get('数据块编号',''))
    return native_locator({'v':label,'target':target,'location':location}),pointer
def applied_decision(row,d,allow_pending=False):
    if d is None:
        if not allow_pending:raise ValueError('Strict identity decision missing for '+str(row.get('IDAC记录ID')))
        d={'keep_model':'no','reason_zh':'结构审核尚未载入；仅供作者测试，不可发布'}
    keep=str(d.get('keep_model','no')).lower()in('yes','true','1')
    out={'Model_Eligibility':'PASS'if keep else'EXCLUDE_IDENTITY','排除或待核原因':d.get('reason_zh','')}
    for prefix,key in [('溶质','solute'),('溶剂','solvent')]:
        for suffix,field in [('InChI','inchi'),('InChIKey','key'),('SMILES','smiles'),('分子式','formula')]:out['有效'+prefix+suffix]=d.get(key+'_'+field,'')
        out[prefix+'身份依据']=d.get(key+'_evidence','')
        out['有效'+prefix+'立体核对']=d.get(key+'_stereo','')
    out['有效溶剂阳离子SMILES']=d.get('solvent_cation_smiles','');out['有效溶剂阴离子SMILES']=d.get('solvent_anion_smiles','');out['已核具体结构说明']=d.get('physical_graph_override_reason','')
    return out,d
def verify_decision_binding(row,d,headers):
    if d is None:return
    numeric={'发表年份','溶剂组分数','温度_K','压力_kPa','无限稀释活度系数','活度系数不确定度','稀释值'}
    def digest(values):
        out=[]
        for h in headers[:32]:
            v=values.get(h,'');v=''if v is None else v
            if h in numeric and v!='':
                try:v=float(v).hex()
                except (TypeError,ValueError):v=str(v)
            else:v=str(v)
            out.append(v)
        return hashlib.sha256(json.dumps(out,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
    tokens=d.get('numeric_fields',{});raw_tokens=d.get('raw_numeric_fields',{});proof={}
    # First verify every raw token supplied by the reviewed, source-bound asset.
    for h,token in raw_tokens.items():
        v=row.get(h,'')
        if v in ('',None):raise ValueError('Raw source numeric value is missing')
        try:actual_hex=float(v).hex();raw_hex=float(token).hex()
        except (ValueError,TypeError):
            if str(v)==str(token):continue
            raise ValueError('Exact raw uncertainty/text binding differs: '+str(row.get('IDAC记录ID'))+' '+h)
        if actual_hex!=raw_hex:
            # The old Sherman parser multiplies printed scientific notation
            # using binary floats. Prove each representation bridge against
            # the original printed decimal and legacy writer, not a tolerance.
            printed=re.search(r'raw_printed_value=([^|]+)',str(row.get('其他固定条件','')))
            parts=re.fullmatch(r'\s*([0-9.]+)\s*×\s*10\s*([-+]?\d+)\s*',printed[1])if printed else None
            exact=Decimal(parts[1])*(Decimal(10)**int(parts[2]))if parts else None
            if d.get('source')=='06_Sherman'and h=='无限稀释活度系数'and exact==Decimal(str(token))and float(format(float(v),'.16g')).hex()==float(token).hex():
                proof[h]={'raw_hex':float(v).hex(),'published_xml':str(token),'published_hex':float(token).hex(),'rule':'Sherman原PDF打印十进制'+printed[1].strip()+'精确核验；旧二进制乘法及16有效位作者出口逐点重现'}
            elif h in tokens and float(format(float(v),'.16g')).hex()==raw_hex and float(tokens[h]).hex()==raw_hex:
                proof[h]={'raw_hex':actual_hex,'published_xml':str(tokens[h]),'published_hex':raw_hex,'rule':'原源parser值经已证实旧作者16有效位序列化，逐ID/字段精确等于冻结银行及原表XML；无ULP或相对误差容许阈值'}
            else:raise ValueError('Exact raw source numeric binding differs: '+str(row.get('IDAC记录ID'))+' '+h)
    if d.get('source_row_sha256')and digest(row)!=d['source_row_sha256']:
        shadow=dict(row)
        for h,token in tokens.items():
            if h not in numeric:continue
            v=row.get(h,'')
            if v in ('',None):raise ValueError('Reviewed numeric output has no parsed source value')
            try:actual=float(v);published=float(token)
            except (ValueError,TypeError):raise ValueError('Numeric output token is not numeric')
            if actual.hex()!=published.hex():
                if h in raw_tokens:
                    rule='原源逐点数值精确核验，原表数字XML表示单独保留'
                elif float(format(actual,'.16g')).hex()==published.hex():
                    rule='旧来源作者16有效位数字序列化；逐ID/字段精确重现，未使用误差阈值'
                else:
                    explicit=d.get('numeric_bridge',{}).get(h,{})
                    if explicit.get('raw_hex')!=actual.hex()or str(explicit.get('published_xml'))!=str(token):raise ValueError('Unproved numeric representation difference: '+str(row.get('IDAC记录ID'))+' '+h)
                    rule=explicit.get('rule','已审核逐点数字表示桥')
                if h not in proof:proof[h]={'raw_hex':actual.hex(),'published_xml':str(token),'published_hex':published.hex(),'rule':rule}
            shadow[h]=published
        if digest(shadow)!=d['source_row_sha256']:raise ValueError('Non-numeric original32field binding differs after raw reconstruction: '+str(row.get('IDAC记录ID')))
    row['_source02_number_xml']=tokens;row['_published_number_xml']=tokens;row['_numeric_bridge_evidence']=proof
def parse_source_rows(ctx,kind,store):
    """Emit original source schema exactly; return scientific counts and headers."""
    m=parser_module(ctx);counts=Counter();headers=m.MASTER_HEADERS if kind=='01_MASTER'else(m.IDAC_HEADERS if hasattr(m,'IDAC_HEADERS')else m.HEADERS)
    if ctx.module_name=='thermoml':
        warnings=[];raw=ctx.raw_dir/'ThermoML.v2020-09-30'
        for p in sorted(raw.rglob('*.json')):
            doc=m.load_json_file(p);counts['files_ok']+=1
            for master,idac in m.parse_one_document(doc,p.relative_to(raw).as_posix(),counts,warnings):
                counts['master_rows']+=1
                row=master if kind=='01_MASTER'else idac
                if row is not None:store.append(row)
        counts['warnings']=len(warnings)
    elif ctx.module_name=='ilthermo':
        source=ctx.raw_dir/'ILThermo.Activity.source';inv=m.source_inventory(source);reasons=Counter()
        for sid,p in inv['items']:
            obj=json.loads(p.read_text(encoding='utf-8-sig'));counts['sets_parsed']+=1
            if not isinstance(obj.get('data',[]),list):raise ValueError('ILThermo data not list')
            for n,rawrow in enumerate(obj.get('data',[]),1):
                master,context=m.make_master_row(source,sid,p,obj,n,rawrow);counts['master_rows']+=1
                idac,reason=m.make_idac_row(context,n,m.DEFAULT_X_THRESHOLD);reasons[reason]+=1
                if idac is not None:counts['idac_rows']+=1
                row=master if kind=='01_MASTER'else idac
                if row is not None:store.append(row)
        counts['reason_counts']=dict(reasons);counts['frozen_sets']=len(inv['set_ids'])
    elif ctx.module_name=='brouwer':
        summary=m.process_source(ctx.raw_dir/'Brouwer_IDAC_Source.xlsx',None,True,1000000,row_sink=store);counts.update(summary)
    elif ctx.module_name=='deepfm':
        source=ctx.raw_dir/'DeepFM_IDAC_Source.xlsx'
        refs=m.load_table(source,'Table S2',2);svs=m.load_table(source,'Table S3',3);sos=m.load_table(source,'Table S4',2)
        refs={m.as_int(r['Ref_ID']):r for r in refs};svs={m.as_int(r['IL/DES_id']):r for r in svs};sos={m.as_int(r['Solute_id']):r for r in sos}
        svstruct={n:m.structure_from_source_smiles(r['SMILES'],'solvent ID '+str(n))for n,r in svs.items()};sostruct={n:m.structure_from_source_smiles(r['SMILES'],'solute ID '+str(n))for n,r in sos.items()}
        for excelrow,r in m.iter_table_s1(source):
            counts['source_experimental_rows']+=1;refid=m.as_int(r['Ref_id']);vid=m.as_int(r['IL_id']);sid=m.as_int(r['Solute_id']);ref=refs[refid];sv=svs[vid];so=sos[sid];ss=svstruct[vid];ts=sostruct[sid]
            if m.clean_text(r['IL_Structure'])!=m.clean_text(sv['IL/DES abbreviation']):raise ValueError('Source solvent ID/name mismatch')
            if m.clean_text(r['Solute_Name']).casefold()!=m.clean_text(so['Solute']).casefold():raise ValueError('Source solute ID/name mismatch')
            gamma=m.as_float(r['IDAC']);temp=m.as_float(r['T/K'])
            if gamma<=0 or abs(math.log(gamma)-m.as_float(r['Ln_IDAC']))>1e-10:raise ValueError('Original experimental lnγ inconsistency')
            stype=m.clean_text(sv['IL/DES']);ncomp=1 if stype=='IL'else 2 if stype=='DES'else''
            if ncomp!=1:counts['mixed_excluded']+=1;continue
            row={h:''for h in headers};row.update({'IDAC记录ID':f'DEEPFM::S1ROW{excelrow}','文献标题':m.clean_text(ref.get('name','')),'期刊':m.clean_text(ref.get('Journal','')),'发表年份':m.as_int(ref['year'])if m.clean_text(ref.get('year',''))else'','原始JSON文件':f'DeepFM_IDAC_Source.xlsx | Table S1 | row {excelrow}','数据块编号':refid,'溶质':m.clean_text(so['Solute']),'溶质分子式':ts['formula'],'溶质InChI':ts['inchi'],'溶质InChIKey':ts['inchikey'],'溶剂组分数':ncomp,'溶剂':m.clean_text(sv['Full name']),'溶剂InChI':ss['inchi'],'溶剂InChIKey':ss['inchikey'],'温度_K':temp,'无限稀释活度系数':gamma,'无限稀释判定方式':'DeepFM supplementary Table S1 explicitly reports experimental IDAC in column H; column I is Ln_IDAC and was verified to equal ln(IDAC); prediction columns J/K excluded.','其他固定条件':f"DeepFM.Ref_id={refid} | DeepFM.IL_id={vid} | DeepFM.Solute_id={sid} | SolventType={stype} | SolventAbbrev={m.clean_text(sv['IL/DES abbreviation'])} | SourceSoluteSMILES={ts['smiles']} | SourceSolventSMILES={ss['smiles']}",'IDAC判定等级':'A_来源明确IDAC','数据质量状态':'OK','建议用于建模':'是','质量备注':f"DeepFM Table S1 experimental IDAC; Sample Partitioning={m.clean_text(r['Sample Partitioning'])}; Predicted_Ln_IDAC and Predicted_IDAC columns were not imported as experimental data."});store.append(row)
    elif ctx.module_name=='tan':
        with m.fitz.open(ctx.raw_dir/'Tan_2022_Supporting_Information.pdf')as doc:
            if len(doc)!=1552:raise ValueError('Tan original SI page count changed')
            sos=m.parse_s1(doc,'solute');svs=m.parse_s1(doc,'solvent');rows=m.parse_s2(doc);m.validate(rows,sos,svs)
        for n,r in enumerate(rows,1):store.append(dict(zip(headers,m.output_row(n,r,sos,svs))))
        counts['source_experimental_rows']=len(rows)
    elif ctx.module_name=='sherman':
        if sha(m.SOURCE)!=m.EXPECTED_SHA256:raise ValueError('Sherman original PDF SHA changed')
        for row in m.rows():store.append(row)
        for r in m.RAW_ROWS:
            method=m.method_of(r)
            if method=='INVSOL':counts['exclude_INVSOL']+=1
            elif method=='KHW':counts['exclude_KHW']+=1
            elif r.get('footnote')=='e':counts['exclude_interpolated']+=1
        counts.update({'source_table_rows':len(m.RAW_ROWS),'non_direct_excluded':counts['exclude_INVSOL']+counts['exclude_KHW']+counts['exclude_interpolated']})
    elif ctx.module_name=='papers':
        # The curated CSV is the checked transcription of the original papers.
        # It is recalculated, not copied from the previously authored workbook.
        manifest=json.loads((ctx.asset_dir/'verified_sources.json').read_text(encoding='utf8'));bank=ctx.asset_dir/'verified_experimental_records.csv'
        if sha(bank)!=manifest['curation_sha256']:raise ValueError('Reviewed transcription CSV changed')
        for rel,expected in manifest['original_sources'].items():
            if sha(ctx.source_dir/rel)!=expected:raise ValueError('Original paper source changed: '+rel)
        numeric={'溶剂组分数','温度_K','压力_kPa','无限稀释活度系数','稀释值','发表年份'}
        with bank.open(encoding='utf-8-sig',newline='')as f:
            for row in csv.DictReader(f):
                for h in numeric:
                    if row[h]!='':row[h]=float(row[h])
                if row['溶剂组分数']!=1 or row['无限稀释活度系数']<=0 or row['温度_K']<=0:raise ValueError('Invalid reviewed experimental point')
                if re.search(r'(?:pred|prediction|square_error)\s*=',row['其他固定条件'],re.I):raise ValueError('Predicted value not eligible')
                store.append(row)
        if store.count!=manifest['expected_records']:raise ValueError('Reviewed transcription count changed')
    store.close();counts['table_rows']=store.count;return list(headers),dict(counts)

def build_source_table(ctx,kind,selection=None,*,allow_pending=False):
    from .audit import EvidenceViews
    ctx.output_path(kind);directory=ctx.work_dir/ctx.source_folder/kind;directory.mkdir(parents=True,exist_ok=True)
    original=SourceRowStore(directory/'original_records.jsonl');headers,statistics=parse_source_rows(ctx,'02_IDAC'if kind=='03_ISOMERS'else kind,original)
    if kind=='03_ISOMERS':return build_isomer_bundle(ctx,directory,original,headers,statistics,selection)
    reg,identity=supplement_registry(ctx)if kind=='02_IDAC'else(None,None);records=selection_records(selection,ctx,directory);extra=list(reg['supplement_headers'])if reg and'supplement_headers'in reg else list(identity.HEADERS)if identity else[]
    output_headers=headers+extra+(UNIFIED_HEADERS if kind=='02_IDAC'else['原始数据','原始数据具体位置']);decisions=Counter();evidence=EvidenceViews()
    def data():
        for row in original:
            d=None
            if kind=='02_IDAC':
                d=(records or{}).get(row['IDAC记录ID']);verify_decision_binding(row,d,headers);decisions['original32_binding_rows']+=int(d is not None);decisions['published_numeric_cells']+=len(row.get('_published_number_xml',{}));added,d=applied_decision(row,d,allow_pending);row.update(added);decisions[row['Model_Eligibility']]+=1
            if reg:row.update(reviewed_supplements(ctx,row,reg,identity))
            link,pointer=raw_locator(ctx,row,reg);row['原始数据']=link;row['原始数据具体位置']=pointer
            if d and d.get('raw_target'):
                target=d['raw_target']
                if target.lower().endswith('.pdf')and'#page='in link['target']:target+='#page='+link['target'].split('#page=',1)[1]
                row['原始数据']=native_locator({'v':'原始数据','target':target,'location':d.get('raw_location','')});row['原始数据具体位置']=d.get('raw_pointer',pointer)
            critical=[]
            if row.get('数据质量状态')=='SOURCE_ANOMALY':
                if'温度字段异常'in str(row.get('质量备注','')):critical.append(headers.index('温度_K')+1)
                if row.get('无限稀释活度系数')is not None and isinstance(row.get('无限稀释活度系数'),(int,float))and row['无限稀释活度系数']<=0:critical.append(headers.index('无限稀释活度系数')+1)
            if kind=='02_IDAC':evidence.add(row,d)
            yield {'values':[native_value(row,h)for h in output_headers],'source_record_id':str(row.get('IDAC记录ID')or row.get('数据点ID')or row.get('记录ID')or''),'row_status':'SOURCE_ANOMALY'if row.get('数据质量状态')=='SOURCE_ANOMALY'else row.get('Model_Eligibility',''),'critical_columns':critical}
    sheets=materialize_split_table(directory,'DATA'if kind=='01_MASTER'else'IDAC',output_headers,data(),freeze_columns=3)
    total_rows=sum(s['rows']for s in sheets)
    if kind=='02_IDAC':sheets.extend(evidence.sheets(ctx,directory,statistics))
    if hasattr(records,'close'):records.close()
    statistics.update({'structure_eligible':decisions.get('PASS',0),'structure_excluded':decisions.get('EXCLUDE_IDENTITY',0),'original32_binding_rows':decisions.get('original32_binding_rows',0),'published_numeric_cells':decisions.get('published_numeric_cells',0),'numeric_representation_bridges':len(evidence.numeric)})
    source_count=statistics.get('source_table_rows',statistics.get('unfolded_source_points',statistics.get('source_experimental_rows',statistics.get('master_rows',statistics['table_rows']))))
    source_excluded=source_count-total_rows
    flow={'title':ctx.source_name+' '+('原始母表'if kind=='01_MASTER'else'实验 IDAC'),'origin':source_origin(ctx),'stages':[{'label':'读取原始数据','count':source_count,'unit':'条','detail':'原始实验字段及原研究/源位置明确。'},{'label':'来源实验筛选','count':total_rows,'unit':'条','detail':source_screen_details(ctx,statistics),'branch':{'label':'未进入本表的原始记录','count':source_excluded,'unit':'条','detail':source_screen_details(ctx,statistics)}},{'label':'双方结构及源质量通过','count':decisions.get('PASS',0),'unit':'条','detail':'完整InChI/Key、名称、具体连接结构、位号、立体和离子身份核实；原建议字段不覆盖。','branch':{'label':'当前神经网络排除','count':decisions.get('EXCLUDE_IDENTITY',0),'unit':'条','detail':'保留实验原值与每行具体理由；不确认的结构不猜填。'}}],'footer':'本表保留原实验记录；建模资格另列。原始链接直接指向00_Source。'+('JSON具体指针另列，Excel打开JSON不保证自动定位指针。'if ctx.module_name in('thermoml','ilthermo')else'')}
    if kind=='01_MASTER':
        flow['stages']=[{'label':'核验原始来源','count':statistics.get('files_ok',statistics.get('frozen_sets',0)),'unit':'个文件/数据块','detail':'原始文件、原研究与原始点位置保留。'},{'label':'读取全部原物性点（含非IDAC）','count':source_count,'unit':'条','detail':'这里统计全部物性记录，不是IDAC点数或训练点数。'},{'label':'母表全量保留（本步不筛模型）','count':total_rows,'unit':'条','detail':'本步排除0条；IDAC性质及单溶剂由02筛选，构造家族由03筛选；最终训练资格见08。'}]
        flow['footer']='原始范围完整保留；所有DATA分页合计'+str(total_rows)+'条。原始链接直接指向00_Source，具体JSON指针另列。'
    return save_manifest(ctx,kind,directory,statistics,flow,sheets,{'original_records_file':str(original.path),'original_schema':headers,'pending_audit_pilot':bool(allow_pending)})

def classification_for_source(source):
    """Pure reviewed classification API for downstream relation audits."""
    from . import classification
    # Clone functions into one explicit policy namespace; common code remains
    # one implementation, while reviewed ThermoML relation policy is preserved.
    ns=dict(vars(classification))
    for name,fn in list(ns.items()):
        if isinstance(fn,types.FunctionType)and fn.__module__==classification.__name__:
            ns[name]=types.FunctionType(fn.__code__,ns,fn.__name__,fn.__defaults__,fn.__closure__)
    if source in ('01_ThermoML','thermoml')or getattr(source,'module_name',None)=='thermoml':
        from .sources import thermoml_classification_policy as policy
        for name in ('structure_features','classify_pair_relation'):
            fn=getattr(policy,name);ns[name]=types.FunctionType(fn.__code__,ns,fn.__name__,fn.__defaults__,fn.__closure__)
    return SimpleNamespace(**ns)

def build_isomer_bundle(ctx,directory,original,headers,statistics,selection):
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors
    from .audit import EvidenceViews
    c=classification_for_source(ctx);decisions=selection_records(selection,ctx,directory)
    if decisions is None:raise ValueError('03 requires a complete strict identity decision manifest')
    reg,identity=supplement_registry(ctx);extra=list(reg['supplement_headers'])if reg and'supplement_headers'in reg else list(identity.HEADERS)if identity else[]
    original_headers=headers+extra
    qualified=RowStore(directory/'qualified_records.jsonl');excluded=RowStore(directory/'excluded_records.jsonl')
    observations={};resolutions={};reason_counts=Counter();stats=Counter();evidence=EvidenceViews()
    def annotated(row,d):
        if reg:row.update(reviewed_supplements(ctx,row,reg,identity))
        added,d=applied_decision(row,d);row.update(added)
        link,pointer=raw_locator(ctx,row,reg)
        if d.get('raw_target'):
            target=d['raw_target']
            if target.lower().endswith('.pdf')and'#page='in link['target']:target+='#page='+link['target'].split('#page=',1)[1]
            link=native_locator({'v':'原始数据','target':target,'location':d.get('raw_location','')});pointer=d.get('raw_pointer',pointer)
        row['原始数据']=link;row['原始数据具体位置']=pointer;return row
    for row in original:
        rid=row['IDAC记录ID'];d=decisions.get(rid)
        if d is None:raise ValueError('Strict identity scope missing '+rid)
        verify_decision_binding(row,d,headers)
        stats['original32_binding_rows']+=1;stats['published_numeric_cells']+=len(d.get('source03_numeric_fields',d.get('numeric_fields',{})))
        row=annotated(row,d);keep=str(d.get('keep_model','no')).lower()in('yes','true','1')
        stats['source_qc_excluded']+=int(str(d.get('source_qc_exclude','no')).lower()in('yes','true','1'))
        if row.get('数据质量状态')=='SOURCE_ANOMALY':keep=False;row['Model_Eligibility']='EXCLUDE_SOURCE_QC';row['排除或待核原因']=str(row.get('质量备注')or'已确认来源异常')
        evidence.add(row,d)
        # Original 03 numeric cells have their own frozen presentation tokens;
        # source02 tokens remain separate for reviewed supplement binding.
        if d.get('source03_numeric_fields'):row['_published_number_xml']=dict(d['source03_numeric_fields'])
        if not keep:
            excluded.append(row);reason_counts[row['排除或待核原因']]+=1;stats['identity_or_source_excluded']+=1;continue
        smi=d.get('solute_smiles','');inchi=d.get('solute_inchi','');key=d.get('solute_key','');mol=Chem.MolFromSmiles(smi)
        if mol is None or not inchi or not key or Chem.MolToInchiKey(mol)!=key or Chem.InchiToInchiKey(inchi)!=key:raise ValueError('Strict full identity inconsistent: '+rid)
        formula=rdMolDescriptors.CalcMolFormula(mol)
        if d.get('solute_formula')and d['solute_formula']!=formula:raise ValueError('Strict formula mismatch '+rid)
        okey=(c.norm_name(row['溶质']),formula,inchi,key)
        if okey not in observations:
            observations[okey]={'key':okey,'name_norm':okey[0],'source_formula':formula,'source_inchi':inchi,'source_inchikey':key,'names':Counter(),'source_files':set(),'source_labels':set(),'row_count':0}
            resolutions[okey]={'mol':mol,'formula':formula,'inchi':inchi,'inchikey':key,'canonical_smiles':smi,'connectivity_key':key.split('-')[0],'method':'STRICT_REVIEWED_IDENTITY','confidence':'已核实','evidence':d.get('solute_evidence','')}
        obs=observations[okey];obs['names'][row['溶质']]+=1;obs['source_files'].add('02_IDAC.xlsx');obs['source_labels'].add(ctx.source_name);obs['row_count']+=1
        row['_observation_key']=list(okey);row['_reviewed_source03_keep']=d.get('source03_keep');qualified.append(row);stats['identity_and_source_qualified']+=1
    qualified.close()
    structures=c.build_structure_index(observations,resolutions)
    families,formulas,positions,relations,relation_counts=c.assign_isomer_families_and_relations(structures)
    main_headers=c.output_headers(original_headers)+UNIFIED_HEADERS
    # Remove accidental duplicate columns without renaming original fields.
    main_headers=list(dict.fromkeys(main_headers));family_stats={};members=[]
    for (formula,conn),s in structures.items():
        fam=families.get((formula,conn))
        if not fam:continue
        members.append({'异构体家族ID':fam['family_id'],'分子式':formula,'当前异构体名称':s['rep_name'],'连接结构键':conn,'标准化InChI':s['rep_inchi'],'标准化InChIKey':s['rep_inchikey'],'标准化CanonicalSMILES':s['canonical_smiles'],'实验记录数':s['row_count'],'异构关系概览':fam['relation_overview']})
        family_stats.setdefault(fam['family_id'],{'formula':formula,'members':fam['family_count'],'member_names':fam['family_names'],'rows':0,'solvents':set(),'temps':set()})
    eligible_output=RowStore(directory/'isomer_records.jsonl')
    for row in qualified:
        okey=tuple(row.pop('_observation_key'));res=resolutions[okey];skey=(res['formula'],res['connectivity_key']);fam=families.get(skey)
        if not fam:
            if row.get('_reviewed_source03_keep')=='yes':raise ValueError('Reviewed surviving isomer point lacks rebuilt family: '+row['IDAC记录ID'])
            row['Model_Eligibility']='EXCLUDE_NO_ISOMER_PARTNER';row['排除或待核原因']='严格结构筛选后，本来源同分子式仅剩一个已确认连接结构；没有构造异构伙伴。';excluded.append(row);stats['no_confirmed_isomer_partner']+=1;reason_counts[row['排除或待核原因']]+=1;continue
        out=c.build_output_row(row,ctx.source_name,'02_IDAC.xlsx',res,structures[skey],fam)
        if row.get('_reviewed_source03_keep')=='no':raise ValueError('Rebuilt isomer point disagrees with independent eligibility manifest: '+row['IDAC记录ID'])
        # Physical structures remain from decisions; generated standardization
        # never overwrites any original source field or source supplement.
        eligible_output.append(out);st=family_stats[fam['family_id']];st['rows']+=1;st['solvents'].add(row['溶剂']);st['temps'].add(row['温度_K'])
    eligible_output.close();excluded.close()
    # Relations are rebuilt after strict eligibility, and every endpoint must
    # reference an actual surviving structure, never an excluded family member.
    allowed={r['连接结构键']for r in members}
    if any(r['连接结构键A']not in allowed or r['连接结构键B']not in allowed for r in relations):raise ValueError('Relation references an absent surviving structure')
    sheets=[]
    def dictsheet(name,hs,rows,header_rows=1,kind='table',freeze=0):
        return materialize_sheet(directory,name,hs,({'values':[native_value(r,h)for h in hs],'source_record_id':r.get('IDAC记录ID',''),'row_status':r.get('Model_Eligibility',''),'critical_columns':[]}for r in rows),header_rows,kind,freeze_columns=freeze)
    family_rows=[{'异构体家族ID':fid,'分子式':st['formula'],'家族成员数':st['members'],'家族成员列表':' | '.join(st['member_names']),'实验记录数':st['rows'],'溶剂数':len(st['solvents']),'温度点数':len(st['temps'])}for fid,st in sorted(family_stats.items())]
    family_headers=['异构体家族ID','分子式','家族成员数','家族成员列表','实验记录数','溶剂数','温度点数']
    relation_headers=list(relations[0])if relations else['异构体家族ID','分子式','成员A','成员B','连接结构键A','连接结构键B','主异构关系','伴随结构差异','置信度','分类依据','成员A_CanonicalSMILES','成员B_CanonicalSMILES','成员A_IDAC记录数','成员B_IDAC记录数','成员A_来源数','成员B_来源数']
    # Family/member reading groups link directly to source data, with a genuine
    # grouping index instead of instructions or separate positional-subfamily tab.
    reading_headers=['异构体家族ID','分子式','当前异构体名称','异构关系概览','溶剂','温度_K','无限稀释活度系数','原始数据','原始数据具体位置','IDAC记录ID']
    reading_index={};catalog=[]
    def reading_rows():
        for n,row in enumerate(ordered_isomer_rows(eligible_output,directory),2):
            key=(row['异构体家族ID'],row['溶剂'])
            if key not in reading_index:
                reading_index[key]=len(catalog);catalog.append({'异构体家族ID':key[0],'分子式':row['分子式'],'溶剂':key[1],'实际实验异构体':'','实验记录数':0,'阅读链接':{'v':'查看实验','target':'','location':"'分组阅读'!A"+str(n)},'_names':set()})
            item=catalog[reading_index[key]];item['实验记录数']+=1;item['_names'].add(row['当前异构体名称']);yield row
    reading= dictsheet('分组阅读',reading_headers,reading_rows(),freeze=3)
    for item in catalog:item['实际实验异构体']=' | '.join(sorted(item.pop('_names'),key=str.casefold))
    catalog_headers=['异构体家族ID','分子式','溶剂','实际实验异构体','实验记录数','阅读链接']
    sheets.append(dictsheet('组目录',catalog_headers,catalog,freeze=3))
    sheets.append(reading)
    sheets.append(dictsheet('异构关系',relation_headers,relations,freeze=3))
    sheets.append(dictsheet('家族汇总',family_headers,family_rows,freeze=2))
    member_headers=list(members[0])if members else['异构体家族ID','分子式','当前异构体名称','连接结构键','标准化InChI','标准化InChIKey','标准化CanonicalSMILES','实验记录数','异构关系概览']
    sheets.append(dictsheet('家族成员',member_headers,members,freeze=3))
    sheets.append(dictsheet('异构体_IDAC',main_headers,eligible_output,2,freeze=3))
    audit_headers=original_headers+UNIFIED_HEADERS
    sheets.append(dictsheet('结构与筛选审计',audit_headers,excluded,kind='audit',freeze=3))
    sheets.extend(evidence.sheets(ctx,directory,statistics))
    statistics.update(dict(stats));statistics.update({'input_idac_rows':original.count,'table_rows':eligible_output.count,'families':len(formulas),'members':len(members),'relations':len(relations),'excluded_from03':excluded.count,'relation_counts':dict(relation_counts),'exclusion_reasons':dict(reason_counts),'numeric_representation_bridges':len(evidence.numeric),'strict_scope_recomputed_and_verified':True})
    if hasattr(decisions,'close'):decisions.close()
    flow={'title':ctx.source_name+' 构造异构体实验','origin':source_origin(ctx)+'本表按照表二相同原始处理及本轮身份筛选实际重建。','stages':[{'label':'原始实验 IDAC','count':original.count,'unit':'条','detail':source_screen_details(ctx,statistics)},{'label':'双方结构及来源质量通过','count':qualified.count,'unit':'条','detail':'InChI/完整Key/具体SMILES一致，名称、位号、立体与离子身份已核实。','branch':{'label':'结构不确定或来源异常','count':stats['identity_or_source_excluded'],'unit':'条','detail':'排除出表三主数据；原值与具体理由保留在筛选审计。'}},{'label':'仍有构造异构伙伴','count':eligible_output.count,'unit':'条','detail':'同一分子式至少2个不同连接结构；位置、骨架、官能团等关系按实际成员重算。','branch':{'label':'仅剩1个确认连接结构','count':stats['no_confirmed_isomer_partner'],'unit':'条','detail':'没有确认的构造异构伙伴，排除出表三主数据并记录原因。'}}],'footer':'家族'+str(len(formulas))+'个；成员'+str(len(members))+'个；关系'+str(len(relations))+'对。原始链接直接指向00_Source，关系和实验均基于本轮实际保留成员。'}
    return save_manifest(ctx,'03_ISOMERS',directory,statistics,flow,sheets,{'original_records_file':str(original.path),'qualified_records_file':str(qualified.path),'isomer_records_file':str(eligible_output.path),'excluded_records_file':str(excluded.path),'original_schema':headers,'family_policy':'same molecular formula and at least two confirmed distinct connectivity graphs','legacy_policy_preserved':'ThermoML source-specific reviewed classification functions; other six shared reviewed policy'})
