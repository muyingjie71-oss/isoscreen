"""Small evidence views calculated from reconstructed points and review assets."""
from collections import Counter
import json,re
from .bundles import materialize_sheet

def yes(value):return str(value).lower() in ('yes','true','1','是')

class EvidenceViews:
    def __init__(self):
        self.solvents={};self.quality={};self.numeric=[]
    def add(self,row,decision):
        d=decision or {};study=str(row.get('DOI')or row.get('文献标题')or row.get('_original_study_reference')or'')
        if not study:
            ref=re.search(r';ref=(.*)$',str(row.get('其他固定条件','')))
            study=('原参考'+('（原表截断）'if'REF_CLIPPED'in str(row.get('质量备注',''))else'')+'：'+ref[1]+'；年份'+str(row.get('发表年份','')))if ref else'原研究待核；原名/结构汇总，完整依据按源位置逐点保留'
        name=str(row.get('溶剂',''));structure=str(d.get('solvent_key',''));scope=(name,study,structure)
        if scope not in self.solvents:
            self.solvents[scope]={'溶剂原名':name,'原研究范围':study,'已核溶剂InChIKey':structure,'原实验行数':0,'溶剂身份确认行数':0,'溶剂仍待核行数':0,'当前可建模行数':0,'当前排除行数':0,'具体原因':'','核查依据':'','原始数据':row.get('原始数据',''),'原始数据具体位置':row.get('原始数据具体位置',''),'_reasons':Counter(),'_evidence':''}
        r=self.solvents[scope];r['原实验行数']+=1;r['溶剂身份确认行数']+=int(yes(d.get('solvent_confirmed')));r['溶剂仍待核行数']+=int(not yes(d.get('solvent_confirmed')))
        keep=row.get('Model_Eligibility')=='PASS';r['当前可建模行数']+=int(keep);r['当前排除行数']+=int(not keep)
        reason=str(row.get('排除或待核原因',''));evidence=str(d.get('solvent_evidence',''))
        if reason:r['_reasons'][reason]+=1
        if evidence and not r['_evidence']:r['_evidence']=evidence
        qstate=str(row.get('数据质量状态',''));qreason=str(row.get('质量备注',''));source_qc=yes(d.get('source_qc_exclude'))
        if source_qc or qstate not in ('','OK') or str(row.get('建议用于建模',''))=='否':
            key=(study,qstate,qreason,str(d.get('source_qc_reason','')))
            if key not in self.quality:self.quality[key]={'审核事项':'原来源质量或既有质量控制','原研究范围':study,'涉及实验行数':0,'原数据质量状态':qstate,'原建议用于建模':str(row.get('建议用于建模','')),'具体原因':str(d.get('source_qc_reason',''))or qreason,'原始记录ID示例':row.get('IDAC记录ID',''),'原始数据':row.get('原始数据',''),'原始数据具体位置':row.get('原始数据具体位置','')}
            self.quality[key]['涉及实验行数']+=1
        for field,proof in row.get('_numeric_bridge_evidence',{}).items():
            self.numeric.append({'原始记录ID':row.get('IDAC记录ID',''),'原字段':field,'原源浮点值':proof['raw_hex'],'原表数字XML':proof['published_xml'],'原表浮点值':proof['published_hex'],'表示差异核验依据':proof['rule'],'原始数据':row.get('原始数据',''),'原始数据具体位置':row.get('原始数据具体位置','')})
    def solvent_rows(self):
        for _,r in sorted(self.solvents.items(),key=lambda p:p[0]):
            reasons=r.pop('_reasons');examples=list(reasons.items())[:3]
            r['具体原因']='；'.join(reason+'（'+str(n)+'条）'for reason,n in examples)or'双方结构及来源质量已通过本轮核查。'
            if len(reasons)>3:r['具体原因']+='；另有'+str(len(reasons)-3)+'种逐点原因，完整原文保留于主表各行。'
            representative=r.pop('_evidence')
            r['核查依据']=('代表实验的完整溶剂核查依据：'+representative+'；本组每个实验的具体核查依据和源位置均保留于主表对应行。')if representative else'本组尚缺可确认结构证据；具体原因及已有原始信息保留于主表对应行。'
            yield r
    def sheets(self,ctx,directory,statistics):
        results=[]
        representative=dict(next(iter(self.solvents.values()),{}).get('原始数据')or{})
        if representative:representative['v']='原始来源代表数据'
        def sheet(name,headers,rows):
            return materialize_sheet(directory,name,headers,({'values':[r.get(h,'')for h in headers],'source_record_id':r.get('原始记录ID',r.get('原始记录ID示例','')),'row_status':'','critical_columns':[]}for r in rows),kind='audit',freeze_columns=2)
        hs=['溶剂原名','原研究范围','原实验行数','溶剂身份确认行数','溶剂仍待核行数','当前可建模行数','当前排除行数','已核溶剂InChIKey','具体原因','核查依据','原始数据','原始数据具体位置']
        results.append(sheet('结构核查汇总',hs,self.solvent_rows()))
        qs=['审核事项','原研究范围','涉及实验行数','原数据质量状态','原建议用于建模','具体原因','原始记录ID示例','原始数据','原始数据具体位置']
        quality=list(self.quality.values())
        # Original scope decisions are visible even when excluded points do not
        # have a corresponding IDAC row. Counts come from actual source parsers.
        mapping={'thermoml':[('混合溶剂','idac_mixed_solvent_skipped'),('无可确定溶剂','idac_no_solvent_skipped'),('非数值γ','idac_non_numeric_skipped')], 'ilthermo':[], 'brouwer':[('混合/非单溶剂','mixed_excluded'),('非正或无效γ/温度','invalid_excluded'),('逐点Corrections实际更正','corrected_points')], 'deepfm':[('DES等非单溶剂','mixed_excluded')], 'sherman':[('INVSOL溶解度反演值','exclude_INVSOL'),('KHW派生值','exclude_KHW'),('原文插值点','exclude_interpolated')]}
        for label,key in mapping.get(ctx.module_name,[]):
            if statistics.get(key):quality.append({'审核事项':'原始范围筛选：'+label,'涉及实验行数':statistics[key],'具体原因':'该计数由本次原始处理及已审方法标记实际生成；不修改原始实验文件。链接为本来源原始文件的代表数据，逐点排除按来源审核资产保留。','原始数据':representative or''})
        if ctx.module_name=='ilthermo':
            labels={'mixed_solvent_excluded':'混合溶剂','not_activity_coefficient':'非活度系数','target_not_at_infinite_dilution':'高于无限稀释摩尔分数阈值','no_target_mole_fraction':'缺少目标组分摩尔分数'}
            for key,n in statistics.get('reason_counts',{}).items():
                if key!='confirmed':quality.append({'审核事项':'原始范围筛选：'+labels.get(key,key),'涉及实验行数':n,'具体原因':'由原始set JSON物性/目标组分及组成条件逐点判定；链接为该来源原文件的代表数据。','原始数据':representative or''})
        if not quality:quality.append({'审核事项':'原始实验一致性核查','涉及实验行数':statistics.get('source_experimental_rows',statistics.get('table_rows',0)),'具体原因':'原始物性、单溶剂、原研究/点ID、正温度及实验γ核验通过。预测不作为实验，未按结构缺失删除原实验。','原始数据':representative or''})
        results.append(sheet('来源审计',qs,quality))
        if self.numeric:
            nh=['原始记录ID','原字段','原源浮点值','原表数字XML','原表浮点值','表示差异核验依据','原始数据','原始数据具体位置'];results.append(sheet('数字表示核验',nh,self.numeric))
        return results
