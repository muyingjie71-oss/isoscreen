"""Pre-canonical source/quality checks. Every experimental value is preserved.

``load_source_context(project, records)`` reads Brouwer's actual raw headings
and the saved ThermoML comparison. ``inspect(record, context)`` returns a list
of action/reason dictionaries; it never mutates a record. Source Data Type is
a source metadata code, never an experimental-versus-predicted flag.
"""
from pathlib import Path
from zipfile import ZipFile
import hashlib
import json
import math
import re
import sys

FIELDS = {
    'quality': ('Raw_数据质量状态','Raw_质量标记'),
    'use': ('Raw_建议用于建模','Raw_建议建模'),
    'components': ('Raw_溶剂组分数','Raw_溶剂数量'),
}
ASSOCIATION_RECORDS = ('BROUWER::RawData!GV120','BROUWER::RawData!GV119','BROUWER::RawData!GV117')
ASSOCIATION_DOI = '10.1021/je0503554'


def _first(record, names, default=''):
    for name in names:
        if record.get(name) not in ('',None):
            return record[name]
    return default


def _number(value):
    try:
        value=float(value)
        return value if math.isfinite(value) else None
    except (TypeError,ValueError):
        return None


def _raw_row(record):
    if record.get('Source')!='Brouwer':
        return None
    rid=_first(record,('original_record_id','Raw_IDAC记录ID'))
    match=re.search(r'Raw\s*Data!\$?[A-Z]+\$?(\d+)$',str(rid),re.I)
    return int(match[1]) if match else None


def load_source_context(project, records):
    project=Path(project)
    records=list(records)
    wanted={number for record in records if (number:=_raw_row(record)) is not None}
    context={'brouwer_by_raw_row':{},'associations':{},'source_sha256':{}}
    if not wanted:
        return context
    source=project/'03_Brouwer/00_Source/Brouwer_IDAC_Source.xlsx'
    # The parser is shipped beside this module; no old 08 entrypoint is needed.
    import source_overlap_audit as audit
    citation='';anchor=0
    with ZipFile(source) as archive:
        strings=audit.strings(archive)
        target=dict(audit.sheet_targets(archive))['Raw Data']
        for number,cells in audit.rows(archive,target,strings,{'A','B','C','D','E','F'}):
            if number>max(wanted):
                break
            if number<=3:
                continue
            reference=audit.val(cells,'A')
            if reference:
                citation=reference;anchor=number
            if number not in wanted:
                continue
            # Only A is a carried bibliography heading in the source layout.
            # CAS/name/type/F are literal fields on the actual raw data row.
            context['brouwer_by_raw_row'][str(number)]={
                'source_file':str(source),'raw_row':number,'reference':citation,'reference_anchor':anchor,
                'year':audit.val(cells,'B'),'source_data_type':audit.val(cells,'C'),
                'source_solvent_CAS':audit.val(cells,'D'),'source_solvent_name':audit.val(cells,'E'),
                'source_field_F':audit.val(cells,'F'),
            }
    missing=wanted-set(map(int,context['brouwer_by_raw_row']))
    if missing:
        raise ValueError('Brouwer source context rows not found: '+str(sorted(missing)[:10]))
    context['source_sha256'][str(source)]=audit.file_hash(source)
    reference_file=project/'01_ThermoML/00_Source/ThermoML.v2020-09-30'/(ASSOCIATION_DOI+'.json')
    data=json.loads(reference_file.read_text(encoding='utf-8'))
    block=next(block for block in data['PureOrMixtureData'] if block['nPureOrMixtureDataNumber']==43)
    points=[(float(point['VariableValue'][0]['nVarValue']),float(point['PropertyValue'][0]['nPropValue'])) for point in block['NumValues']]
    context['association_reference']={'DOI':ASSOCIATION_DOI,'source_file':str(reference_file),'block':43,
                                      'reported_method':'Chromatography','points':points}
    context['source_sha256'][str(reference_file)]=audit.file_hash(reference_file)
    for record in records:
        rid=_first(record,('original_record_id','Raw_IDAC记录ID'))
        if record.get('Source')!='Brouwer' or rid not in ASSOCIATION_RECORDS:
            continue
        temperature,gamma=_number(record.get('T_K')),_number(record.get('gamma_inf'))
        if temperature is None or gamma is None:
            continue
        at_temperature=[value for t,value in points if abs(t-temperature)<1e-9]
        value_at=[t for t,value in points if abs(value-gamma)<1e-9]
        if not at_temperature:
            continue
        relative=abs(gamma-at_temperature[0])/max(gamma,at_temperature[0])
        reproduced=relative>.005 and len(value_at)==1 and abs(value_at[0]-temperature)>1e-9
        context['associations'][rid]={
            'reproduced':reproduced,'DOI':ASSOCIATION_DOI,'T_K':temperature,'Brouwer_gamma':gamma,
            'ThermoML_gamma_same_T':at_temperature[0],'ThermoML_T_same_gamma':value_at,
            'relative_difference':relative,'reference_file':str(reference_file),'reference_block':43,
            'reported_Brouwer_method':record.get('Raw_实验方法',''),'reported_ThermoML_method':'Chromatography',
            'original_paper_error_confirmed':False,
        }
    return context


def _context(record, context):
    number=_raw_row(record)
    return (context or {}).get('brouwer_by_raw_row',{}).get(str(number),{}) if number is not None else {}


def _issue(action,reason,**evidence):
    severity='EXCLUDE' if action.startswith('EXCLUDE_') else 'REVIEW'
    return {'action':action,'status':action,'code':action,'severity':severity,'reason':reason,**evidence}


def _uncertainty_text_issue(text):
    # Relative percentage ranges and explicit +/- descriptions are legitimate
    # source uncertainty metadata; they are not malformed gamma measurements.
    value=str(text).strip()
    lower=value.casefold()
    if re.search(r'未明确|not\s+specified|basis\s+not|unclear',lower):
        return _issue('REVIEW_UNCERTAINTY','不确定度表示或基准未明确；实验点保留，不据此设置不确定度权重')
    if '%' in value and re.search(r'\d',value):
        if re.search(r'(^|\s)-\s*\d+(?:\.\d+)?\s*%',value):
            return _issue('EXCLUDE_DATA_ERROR','文字记录的不确定度百分比为负')
        return None
    return _issue('REVIEW_UNCERTAINTY','不确定度不是可解析的有限数值；不据此替换或删除实验值')


def inspect(record, context=None):
    issues=[]
    quality=str(_first(record,FIELDS['quality'])).strip()
    use=str(_first(record,FIELDS['use'])).strip().casefold()
    if quality not in ('OK','OK_CORRECTED'):
        issues.append(_issue('EXCLUDE_ORIGINAL_QUALITY','原始质量状态不符合建模条件：'+(quality or '未报告')))
    if use not in ('是','yes','true','1','y'):
        issues.append(_issue('EXCLUDE_ORIGINAL_RECOMMENDATION','原表未推荐用于建模：'+(use or '未报告')))
    temperature,gamma=_number(record.get('T_K')),_number(record.get('gamma_inf'))
    if temperature is None or temperature<=0:
        issues.append(_issue('EXCLUDE_INVALID_VALUE','温度不是正的有限数值'))
    if gamma is None or gamma<=0:
        issues.append(_issue('EXCLUDE_INVALID_VALUE','γ∞不是正的有限数值'))
    components=_number(_first(record,FIELDS['components']))
    if components!=1:
        issues.append(_issue('EXCLUDE_COMPOSITION','原表溶剂组分数不等于1或未确定'))
    for source_field,field,label in [('Raw_温度_K','T_K','温度'),('Raw_无限稀释活度系数','gamma_inf','γ∞')]:
        saved=_number(record.get(source_field));value=_number(record.get(field))
        if record.get(source_field) not in ('',None) and saved is None:
            issues.append(_issue('EXCLUDE_RAW_VALUE_UNRESOLVED','保存的原始'+label+'不能解析为有限数值'))
        elif saved is not None and value is not None and not math.isclose(saved,value,rel_tol=1e-12,abs_tol=1e-12):
            issues.append(_issue('EXCLUDE_RAW_MISMATCH','实验主值与保存的原始'+label+'不一致',raw_value=record.get(source_field),main_value=record.get(field)))
    uncertainty_text=record.get('Raw_活度系数不确定度','')
    uncertainty=_number(uncertainty_text)
    if uncertainty is not None and uncertainty<0:
        issues.append(_issue('EXCLUDE_DATA_ERROR','报告的活度系数不确定度为负'))
    elif uncertainty_text not in ('',None) and uncertainty is None:
        uncertainty_issue=_uncertainty_text_issue(uncertainty_text)
        if uncertainty_issue:issues.append(uncertainty_issue)
    if temperature is not None and temperature>0 and (temperature<200 or temperature>700):
        issues.append(_issue('REVIEW_T_RANGE','温度超出200–700 K核查范围；不据此排除'))
    if gamma is not None and gamma>0 and (gamma>1e5 or gamma<1e-6):
        issues.append(_issue('REVIEW_MAGNITUDE','γ∞绝对量级较极端；不据此排除'))
    context_row=_context(record,context)
    original_name=str(context_row.get('source_solvent_name') or record.get('Source_Solvent_Name') or record.get('Raw_溶剂') or record.get('Solvent',''))
    cas=str(context_row.get('source_solvent_CAS') or record.get('Source_Solvent_CAS',''))
    if not cas:
        cas=str(record.get('Raw_其他固定条件',''))
    distinct_cas=set(re.findall(r'\b\d{2,7}-\d{2}-\d\b',cas))
    if '/' in original_name and len(distinct_cas)>=2:
        issues.append(_issue('EXCLUDE_COMPOSITION','源数据原溶剂名称明确列出两种溶剂及两种不同CAS，按单一溶剂范围排除',source_solvent_name=original_name,source_solvent_CAS=cas,raw_source_row=context_row.get('raw_row')))
    elif re.search(r'\b(?:mixture|eutectic|des)\b|混合|共晶',original_name,re.I):
        issues.append(_issue('REVIEW_SOLVENT_COMPOSITION','源名称提示混合/共晶体系，需回原研究核对实际组成'))
    identity=str(record.get('Solute_Identity',''))
    if not re.fullmatch(r'[A-Z]{14}-[A-Z]{10}-[A-Z]',identity):
        issues.append(_issue('EXCLUDE_SOLUTE_IDENTITY','溶质完整结构身份未确定'))
    raw_key=str(record.get('Raw_溶质InChIKey',''))
    if raw_key and identity and raw_key[:14]!=identity[:14]:
        issues.append(_issue('REVIEW_SOLUTE_IDENTITY','原始溶质InChIKey与建模连接结构不一致，需核对结构更正依据'))
    rid=_first(record,('original_record_id','Raw_IDAC记录ID'))
    association=(context or {}).get('associations',{}).get(rid)
    if record.get('Source')=='Brouwer' and rid in ASSOCIATION_RECORDS:
        if association and association.get('reproduced'):
            issues.append(_issue('EXCLUDE_SOURCE_CONFLICT','同研究ThermoML与Brouwer的1-pentene温度—γ∞反序对应仍可复现；按已授权的高度可疑来源对应冲突排除，源值保留，未声称原论文错误已证实',evidence=association))
        else:
            issues.append(_issue('REVIEW_SOURCE_ASSOCIATION','旧来源对应排除的当前证据未完整重现，需复核；不直接继承旧排除'))
    return issues


def prepare_temperature_record(record, context=None):
    """Recover bibliographic context solely for a provisional temperature series."""
    output={key:record.get(key,'') for key in ('Study_ID','Solute_Identity','Solvent_Audit_ID','T_K','gamma_inf','canonical_record_id')}
    if not output['Solvent_Audit_ID']:
        output['Solvent_Audit_ID']=record.get('Solvent_Mol_ID') or record.get('Solvent_Identity','')
    output['Effective_Conditions']=record.get('Effective_Conditions',['','','','',''])
    output['Effective_Conditions_Ambiguous']=record.get('Effective_Conditions_Ambiguous',False)
    study=str(output['Study_ID'])
    known=study.startswith(('DOI:','BIB:','TITLE:','RAW_BIB_','10.'))
    output['Study_Confirmed']=known
    output['Temperature_Study_Basis']='CURATED_STUDY_ID' if known else 'UNRESOLVED'
    if not known and record.get('Source')=='Brouwer':
        row=_context(record,context)
        citation=str(row.get('reference') or record.get('Source_Original_Reference',''))
        anchor=row.get('reference_anchor') or record.get('Source_Reference_Anchor_Row','')
        if anchor and len(citation)>25 and re.search(r'\b(?:18|19|20)\d{2}\b',citation):
            parts=json.dumps([citation,anchor],ensure_ascii=False,separators=(',',':'))
            output['Study_ID']='RAW_BIB_'+hashlib.sha256(parts.encode('utf-8')).hexdigest()[:20]
            output['Study_Confirmed']=True
            output['Temperature_Study_Basis']='ORIGINAL_RAW_CITATION_PROVISIONAL'
            output['Temperature_Original_Reference']=citation
            output['Temperature_Reference_Anchor_Row']=anchor
    return output
