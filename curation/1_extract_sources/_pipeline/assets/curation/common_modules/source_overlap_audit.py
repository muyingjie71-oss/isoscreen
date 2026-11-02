#!/usr/bin/env python3
"""Audit the seven experimental 03_ISOMERS workbooks without changing them.

The audit keeps every original observation. A pairs alone may share a canonical
record; B and D observations retain separate canonical records and share a
leakage group for later review. Different proven original studies are C, even
when their measured values agree. A compilation DOI is never an experiment DOI.

Default: python source_overlap_audit.py
Optional: --project PATH --output PATH --gamma-relative-tolerance 0.005
Dependencies: pypdf for local PDF titles.
The data reader, matcher and streaming XLSX expansion use Python stdlib only.
"""
from __future__ import annotations
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from decimal import Decimal, InvalidOperation
import argparse, hashlib, html, json, math, os, re, sqlite3, subprocess
import tempfile, unicodedata, xml.etree.ElementTree as E, itertools, datetime
from urllib.parse import unquote

VERSION='1.0'
S='http://schemas.openxmlformats.org/spreadsheetml/2006/main'
R='http://schemas.openxmlformats.org/officeDocument/2006/relationships'
T=lambda n:'{'+S+'}'+n
ROW=re.compile(br'<(?:x:)?row\b[^>]*(?:/>|>.*?</(?:x:)?row>)',re.S)
CELL=re.compile(br'<(?:x:)?c\b[^>]*r="([A-Z]+)(\d+)"[^>]*?(?:/>|>.*?</(?:x:)?c>)',re.S)
TEXT=re.compile(br'<(?:x:)?t(?:\s[^>]*)?>(.*?)</(?:x:)?t>',re.S)
VALUE=re.compile(br'<(?:x:)?v>(.*?)</(?:x:)?v>',re.S)
SOURCES=['01_ThermoML','02_ILThermo','03_Brouwer','04_DeepFM','05_Tan','06_Sherman','07_单独收集论文']
LABELS=['ThermoML','ILThermo','Brouwer','DeepFM','Tan','Sherman','单独收集论文']
RAW_COLS=['AC','X','AA','Z','Y','BJ','BK','AY','AZ','BA','BB','J','I','BC','BD','K','L','M','N','O','P','Q','BE','BF','BG','BH','BI','BL','T','R','S','U']
SELECTED=set(RAW_COLS)|{'C','AS','AT','AU','AV','AW','AX','AH'}
SECONDARY_DOIS={'10.1021/ie950281x'}
CLASSES={'A':'完全重复','B':'高度疑似重复','C':'同条件独立实验','D':'冲突'}

def dumps(v):return json.dumps(v,ensure_ascii=False,separators=(',',':'))
def digest(v,prefix=''):return prefix+hashlib.sha256(dumps(v).encode()).hexdigest()[:24]
def file_hash(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def colname(n):
    s=''
    while n:n,r=divmod(n-1,26);s=chr(65+r)+s
    return s
def sheet_targets(z):
    wb=E.fromstring(z.read('xl/workbook.xml'));rels=E.fromstring(z.read('xl/_rels/workbook.xml.rels'))
    refs={r.get('Id'):r.get('Target') for r in rels}
    return [(e.get('name'),refs[e.get('{'+R+'}id')].lstrip('/') if refs[e.get('{'+R+'}id')].startswith('/') else 'xl/'+refs[e.get('{'+R+'}id')]) for e in wb.find(T('sheets'))]
def strings(z):
    if 'xl/sharedStrings.xml' not in z.namelist():return []
    return [''.join(e.itertext()) for e in E.fromstring(z.read('xl/sharedStrings.xml'))]
def rows(z,target,ss,selected=None):
    with z.open(target) as f:
        buf=b''
        while True:
            chunk=f.read(524288);buf+=chunk;pos=0
            for rm in ROW.finditer(buf):
                pos=rm.end();n=int(re.search(br'\br="(\d+)"',rm[0])[1]);c={}
                for cm in CELL.finditer(rm[0]):
                    col=cm[1].decode()
                    if selected is not None and col not in selected:continue
                    raw=cm[0]
                    if b't="inlineStr"' in raw:c[col]=('s',html.unescape(b''.join(TEXT.findall(raw)).decode()))
                    else:
                        vm=VALUE.search(raw);v=html.unescape(vm[1].decode()) if vm else ''
                        if b't="s"' in raw:c[col]=('s',ss[int(v)] if v else '')
                        else:c[col]=('s' if b't="str"' in raw else 'n',v)
                yield n,c
            buf=buf[pos:]
            if not chunk:break
def val(c,k):return c.get(k,('s',''))[1]
def number(s):
    try:
        v=float(s);return v if math.isfinite(v) else None
    except (ValueError,TypeError):return None
def norm(s):
    s=unicodedata.normalize('NFKC',html.unescape(str(s))).casefold()
    s=s.translate(str.maketrans({'−':'-','–':'-','—':'-','‐':'-','’':"'",'′':"'"}))
    return re.sub(r'\s+','',s)
def title_key(s):return re.sub(r'[^\w]','',norm(s))
def doi(s):
    found=re.findall(r'10\.\d{4,9}/[^\s<>"|]+',unquote(str(s)),re.I)
    if len(set(found))!=1:return ''
    result=found[0].lower().rstrip('.,;')
    while result.endswith(')') and result.count(')')>result.count('('):result=result[:-1]
    return result
def year(s):
    m=re.search(r'\b((?:19|20)\d{2})[a-z]?\b',str(s));return m[1] if m else ''
def journal(s):
    s=title_key(re.sub(r'^\s*the\s+','',str(s),flags=re.I))
    aliases={
      'journalofchemicalandengineeringdata':'jchemengdata','journalofchemicalengineeringdata':'jchemengdata',
      'fluidphaseequilibria':'fluidphaseequilib','industrialandengineeringchemistryresearch':'indengchemres',
      'industrialengineeringchemistryresearch':'indengchemres','journalofchemicalthermodynamics':'jchemthermodyn',
      'jchemthermodynamics':'jchemthermodyn','journalofsolutionchemistry':'jsolchem','jsolutionchem':'jsolchem','solutionchem':'jsolchem',
      'journalofmolecularliquids':'jmolliq','journalofphysicalchemistry':'jphyschem','journalofphysicalchemistrya':'jphyschema',
      'journalofphysicalchemistryb':'jphyschemb','journalofphysicalchemistryc':'jphyschemc','journalofchemicalphysics':'jchemphys',
      'journalofchromatography':'jchromatogr','journalofchromatographya':'jchromatogra',
      'chemicalengineeringjournal':'chemengj','chemicalengineeringscience':'chemengsci','chemicalengineeringcommunications':'chemengcommun',
      'chemicalengineeringtechnology':'chemengtechnol','separationandpurificationtechnology':'seppuriftechnol',
      'analyticalchemistry':'analchem','aichejournal':'aichej','internationaljournalofthermophysics':'intjthermophys',
      'journalofappliedchemistrybiotechnology':'japplchembiotechnol','journaloffluorinechemistry':'jfluorinechem',
      'industrialengineeringchemistryprocessdesignanddevelopment':'indengchemprocessdesdev',
      'internationaljournalofmolecularsciences':'intjmolsci','transactionsofthefaradaysociety':'transfaradaysoc',
      'journalofliquidchromatography':'jliqchromatogr','journalofcolloidandinterfacescience':'jcolloidinterfacesci',
      'rscadvances':'rscadv','thermochimicaacta':'thermochimacta',
    }
    return aliases.get(s,s)
def bibliographic(s):
    # Require a journal, volume and first page. Author/year alone is insufficient.
    patterns=[r'([A-Za-z][A-Za-z .&−-]{3,}?)\s*(?:\(?(?:19|20)\d{2}\)?\s*[,;:]?\s*)?(\d+)\s*(?:\([^)]*\))?\s*[,;:]\s*([A-Za-z]?\d+)',r'([A-Za-z][A-Za-z .&−-]{3,}?)\s+(\d+)\s*\((?:19|20)\d{2}\)\s+([A-Za-z]?\d+)']
    for pattern in patterns:
        for m in re.finditer(pattern,s):
            j=journal(m[1])
            if len(j)>=6 and j not in ['volume','number']:return (j,m[2],m[3].lower())
    return None
def valid_key(s):return bool(re.fullmatch(r'[A-Z]{14}-[A-Z]{10}-[A-Z]',s))
def numcell(s):return {'n':str(s)} if str(s) else ''

@dataclass(slots=True)
class Record:
    i:int;key:str;source:int;sheet:str;row:int;original_id:str
    solute_name:str;solvent_name:str;t:float|None;g:float|None;p:float|None
    solute_key:str='';solvent_key:str='';identity_basis:str='';identity_strong:bool=True
    reporting_doi:str='';original_doi:str='';study:str='';study_basis:str='';study_title:str=''
    citation:str='';study_year:str='';year_reported:str='';phase:str='';standard:str='';method:str='';fixed:str=''
    original_use:str='';quality:str='';flags:list=field(default_factory=list)
    classes:set=field(default_factory=set);canonical:str='';leakage:str=''

class Union:
    def __init__(self,n):self.parent=list(range(n));self.size=[1]*n
    def find(self,i):
        while self.parent[i]!=i:self.parent[i]=self.parent[self.parent[i]];i=self.parent[i]
        return i
    def join(self,a,b):
        a=self.find(a);b=self.find(b)
        if a==b:return
        if self.size[a]<self.size[b]:a,b=b,a
        self.parent[b]=a;self.size[a]+=self.size[b]
    def groups(self):
        g=defaultdict(list)
        for i in range(len(self.parent)):g[self.find(i)].append(i)
        return list(g.values())

def load_inputs(project,db):
    records=[];alias=defaultdict(set);metadata=[];raw_headers=None
    db.execute('CREATE TABLE raw(i INTEGER PRIMARY KEY,data TEXT)')
    for si,folder in enumerate(SOURCES):
        path=project/folder/'03_ISOMERS.xlsx';checksum=file_hash(path);batch=[];start=len(records)
        with ZipFile(path) as z:
            ss=strings(z)
            for sheet,target in sheet_targets(z):
                if not sheet.startswith('异构体_IDAC'):continue
                for n,c in rows(z,target,ss,SELECTED):
                    if n==2:
                        h=[val(c,k) for k in RAW_COLS]
                        if raw_headers is None:raw_headers=h
                        assert h==raw_headers,(folder,'experimental schema differs');continue
                    if n<3:continue
                    rid=val(c,'AC');assert rid,(folder,sheet,n,'missing original id')
                    i=len(records);r=Record(i,digest([folder,rid,sheet,n],'SR_'),si,sheet,n,rid,val(c,'C') or val(c,'AY'),val(c,'I'),number(val(c,'K')),number(val(c,'M')),number(val(c,'L')))
                    r.solute_key=val(c,'BB') or val(c,'AU');r.solvent_key=val(c,'BD')
                    r.reporting_doi=doi(val(c,'X'));r.study_title=val(c,'AA');r.year_reported=year(val(c,'Y'))
                    r.phase=norm(val(c,'P'));r.standard=norm(val(c,'Q'));r.method=norm(val(c,'O'));r.fixed=val(c,'BI')
                    r.original_use=val(c,'S');r.quality=val(c,'R')
                    r.citation=r.fixed.split('reference=',1)[-1].split(' | ',1)[0] if folder=='06_Sherman' and 'reference=' in r.fixed else ''
                    for name,key in [(r.solute_name,r.solute_key),(val(c,'AY'),r.solute_key),(r.solvent_name,r.solvent_key)]:
                        if name and valid_key(key):alias[norm(name)].add(key)
                    if r.t is None or r.t<=0 or r.g is None or r.g<=0:r.flags.append('实验数值无效，未进入比较')
                    if val(c,'J') and number(val(c,'J'))!=1:r.flags.append('非单一溶剂，未进入比较')
                    records.append(r);batch.append((i,dumps(c)))
                    if len(batch)>=1000:db.executemany('INSERT INTO raw VALUES (?,?)',batch);batch=[]
            # Existing registry aliases are structure-grounded and preserve full stereo keys.
            registry=dict(sheet_targets(z)).get('化合物注册表')
            if registry:
                for n,c in rows(z,registry,ss,{'C','E','M'}):
                    if n>1 and valid_key(val(c,'E')):
                        for name in (val(c,'C')+' | '+val(c,'M')).split(' | '):
                            if name:alias[norm(name)].add(val(c,'E'))
        if batch:db.executemany('INSERT INTO raw VALUES (?,?)',batch)
        metadata.append({'source':LABELS[si],'path':str(path.relative_to(project)).replace('\\','/'),'sha256':checksum,'rows':len(records)-start})
        print('READ',folder,len(records)-start,flush=True)
    db.commit();assert len({r.key for r in records})==len(records)
    for r in records:
        basis=[]
        for attr,name in [('solute_key',r.solute_name),('solvent_key',r.solvent_name)]:
            current=getattr(r,attr)
            if valid_key(current):basis.append('完整InChIKey');continue
            choices=alias.get(norm(name),set())
            if len(choices)==1:setattr(r,attr,next(iter(choices)));basis.append('本地唯一结构别名')
            else:
                setattr(r,attr,'NAME:'+norm(name));basis.append('名称匹配')
                r.identity_strong=False
                if len(choices)>1:r.flags.append('名称对应多个完整结构标识')
                else:r.flags.append('缺少可唯一对应的结构标识')
        r.identity_basis=' / '.join(basis)
    return records,metadata,raw_headers

def resolve_studies(project,records,db):
    auxiliary={};papers={};title_index=defaultdict(set);biblio_index=defaultdict(set)
    def read_json(path):
        if not path.exists():return None
        auxiliary[str(path.relative_to(project)).replace('\\','/')]=file_hash(path)
        return json.loads(path.read_text(encoding='utf-8'))
    # Read the original Brouwer reference anchors instead of treating the database as a study.
    bpath=project/'03_Brouwer/00_Source/Brouwer_IDAC_Source.xlsx';brefs={}
    if bpath.exists():
        auxiliary[str(bpath.relative_to(project)).replace('\\','/')]=file_hash(bpath)
        with ZipFile(bpath) as z:
            target=dict(sheet_targets(z))['Raw Data'];current=''
            for n,c in rows(z,target,strings(z),{'A'}):
                if n<4:continue
                if val(c,'A'):current=val(c,'A')
                brefs[n]=current
    ilrefs={}
    for r in records:
        if r.reporting_doi and r.reporting_doi not in SECONDARY_DOIS:
            papers.setdefault(r.reporting_doi,{'doi':r.reporting_doi,'title':r.study_title,'year':r.year_reported,'biblio':None,'location':''})
        if r.source==2:
            m=re.search(r'!\D+(\d+)$',r.original_id)
            r.citation=brefs.get(int(m[1]),'') if m else ''
    for d,p in list(papers.items()):
        path=project/'01_ThermoML/00_Source/ThermoML.v2020-09-30'/Path(d+'.json')
        data=read_json(path)
        if data:
            c=data.get('Citation',{});p.update(title=c.get('sTitle',p['title']),year=str(c.get('yrPubYr',p['year'])),location=str(path.relative_to(project)).replace('\\','/'))
            first=re.match(r'[A-Za-z]?\d+',str(c.get('sPage','')))
            if c.get('sPubName') and c.get('sVol') and first:p['biblio']=(journal(c['sPubName']),str(c['sVol']),first[0].lower())
    # Read only the first page of each local primary PDF. The source row's path
    # connects its DOI to the PDF; no DOI is guessed from file names or authors.
    pdf_dois=defaultdict(set)
    for r in records:
        if r.source!=6 or not r.reporting_doi:continue
        raw=json.loads(db.execute('SELECT data FROM raw WHERE i=?',(r.i,)).fetchone()[0])
        location=val(raw,'BJ').replace('\\','/')
        for fragment in location.split(' | '):
            if fragment.lower().endswith('.pdf'):pdf_dois[fragment.strip()].add(r.reporting_doi)
    if pdf_dois:
        try:from pypdf import PdfReader
        except ImportError:PdfReader=None
        for location,ds in pdf_dois.items():
            path=project/'07_单独收集论文'/location
            if len(ds)!=1 or not path.exists() or PdfReader is None:continue
            text=PdfReader(path).pages[0].extract_text() or ''
            b=bibliographic(text)
            auxiliary[str(path.relative_to(project)).replace('\\','/')]=file_hash(path)
            if b:
                p=papers[next(iter(ds))];p['biblio']=b;p['location']=str(path.relative_to(project)).replace('\\','/')
    for r in records:
        if r.source!=1:continue
        setid=r.original_id.split('::')[1]
        if setid not in ilrefs:
            data=read_json(project/f'02_ILThermo/00_Source/ILThermo.Activity.source/ilset/{setid}.json')
            ilrefs[setid]=(data or {}).get('ref',{}).get('full','')
        r.citation=ilrefs[setid]
    for d,p in papers.items():
        if p['title']:title_index[title_key(p['title'])].add(d)
        if p['biblio']:biblio_index[p['biblio']].add(d)
    # Titles without DOI can still form proven study identities. Bibliography connects
    # alternative titles; a DOI/title link is used only when it is unique.
    title_biblio=defaultdict(set)
    for r in records:
        if r.study_title and r.reporting_doi not in SECONDARY_DOIS:
            b=bibliographic(r.citation)
            if b:title_biblio[title_key(r.study_title)].add(b)
    for r in records:
        if r.reporting_doi in SECONDARY_DOIS:
            r.flags.append('表内DOI属于汇编论文，未作为原研究DOI')
            if r.citation and 'unresolved' not in r.citation:
                r.study='REF:'+title_key(r.citation);r.study_basis='汇编内原始参考文献';r.study_year=year(r.citation)
            else:r.flags.append('汇编中的原始参考文献未解决')
            continue
        direct=r.reporting_doi
        tk=title_key(r.study_title);bk=bibliographic(r.citation);candidates=set()
        if direct:r.original_doi=direct;r.study='DOI:'+direct;r.study_basis='原表DOI'
        else:
            if tk and len(title_index[tk])==1:candidates|=title_index[tk]
            if bk and len(biblio_index[bk])==1:candidates|=biblio_index[bk]
            # An exact local bibliography can connect ILThermo/DeepFM titles to DOI.
            for tb in title_biblio.get(tk,()):
                if len(biblio_index[tb])==1:candidates|=biblio_index[tb]
            if len(candidates)==1:
                r.original_doi=next(iter(candidates));r.study='DOI:'+r.original_doi;r.study_basis='本地文献标题/卷页唯一对应DOI'
            elif len(candidates)>1:r.flags.append('原研究元数据对应多个DOI，未合并')
            elif bk:r.study='BIB:'+dumps(bk);r.study_basis='原始参考文献的期刊卷页'
            elif tk and len(tk)>=30 and r.source!=2:r.study='TITLE:'+tk;r.study_basis='完整原研究标题'
            else:r.flags.append('缺少可确定原研究的DOI/完整题录')
        if r.original_doi:r.study_year=papers.get(r.original_doi,{}).get('year','') or r.year_reported
        else:r.study_year=year(r.citation) or r.year_reported
        if r.source==4 and 'ref=' in r.fixed:
            r.citation=r.fixed.split('ref=',1)[1]
            r.flags.append('Tan原参考文献字段截断，未按作者年份推断DOI')
    # Alias equal full titles to a bibliography even if neither has a DOI.
    for r in records:
        if r.study.startswith('TITLE:'):
            bs=title_biblio.get(title_key(r.study_title),set())
            if len(bs)==1:r.study='BIB:'+dumps(next(iter(bs)));r.study_basis='完整标题对应唯一原始卷页'
    return auxiliary

def method_group(s):
    if s in ['gc','gc1','gc2','chromatography','gas-liquidchromatography','gasliquidchromatography']:return 'gc'
    return s
def fixed_signature(r):
    # Database IDs, molecular weights, reference text, ln-gamma, source locations and
    # printed values are provenance, not additional experimental conditions.
    if r.source in [2,3,4,5,6]:return ''
    s=r.fixed
    if not s:return ''
    return norm(s)
def conditions(a,b,args):
    notes=[];near_pressure=False
    if a.p is not None and b.p is not None:
        diff=abs(a.p-b.p)
        if diff>args.pressure_tolerance:
            if diff<=args.pressure_rounding_tolerance:near_pressure=True;notes.append('压力记载存在小差异')
            else:return False,False,'不同压力'
    elif (a.p is None)!=(b.p is None):notes.append('一侧压力未报告')
    for av,bv,label in [(a.phase,b.phase,'相态'),(a.standard,b.standard,'标准态'),(method_group(a.method),method_group(b.method),'实验方法'),(fixed_signature(a),fixed_signature(b),'其他固定条件')]:
        if av and bv and av!=bv:return False,False,'不同'+label
        if bool(av)!=bool(bv):notes.append('一侧'+label+'未报告')
    return True,near_pressure,'；'.join(notes)
def pair_class(a,b,args):
    if abs(a.t-b.t)>args.temperature_tolerance:return None
    compatible,near_pressure,condition_note=conditions(a,b,args)
    if not compatible:return None
    same=bool(a.study and a.study==b.study)
    different=bool(a.study and b.study and a.study!=b.study)
    delta=abs(a.g-b.g);relative=delta/max(abs(a.g),abs(b.g))
    exact=math.isclose(a.g,b.g,rel_tol=1e-12,abs_tol=0)
    close=delta<=max(args.gamma_absolute_tolerance,args.gamma_relative_tolerance*max(abs(a.g),abs(b.g)))
    if different:
        # A differently formatted, unmapped reference must not establish independence.
        if a.study.startswith('REF:') or b.study.startswith('REF:'):return ('B','原研究引用格式未能唯一对应，实验值接近',condition_note,relative) if close else None
        if a.study.split(':',1)[0]!=b.study.split(':',1)[0]:return ('B','原研究标识体系尚未对应，不能据此确认独立实验',condition_note,relative) if close else None
        if near_pressure:return None
        return ('C','原研究身份明确且不同',condition_note,relative)
    if same:
        if exact and not near_pressure and a.identity_strong and b.identity_strong:
            if a.source!=b.source or a.original_id==b.original_id:return ('A','同一原研究、结构与温度匹配，γ∞一致',condition_note,relative)
            return ('B','同一来源的不同记录ID，需排除独立重复测量',condition_note,relative)
        if close:return ('B','同一原研究，数值/条件存在微小记载差异',condition_note,relative)
        if not near_pressure and a.identity_strong and b.identity_strong:return ('D','同一原研究和实验条件，γ∞超出近值阈值',condition_note,relative)
        return None
    if close:return ('B','原研究证据不足，仅条件与实验值高度相似',condition_note,relative)
    return None

def audit(records,args):
    by_entity=defaultdict(list);pairs=[];skipped=0
    for r in records:
        if any('未进入比较' in f for f in r.flags):skipped+=1;continue
        by_entity[(r.solute_key,r.solvent_key)].append(r.i)
    for indices in by_entity.values():
        ordered=sorted(indices,key=lambda i:(records[i].t,records[i].key))
        for at,ia in enumerate(ordered):
            a=records[ia]
            for ib in ordered[at+1:]:
                b=records[ib]
                if b.t-a.t>args.temperature_tolerance:break
                result=pair_class(a,b,args)
                if result:pairs.append([ia,ib,*result])
    # Different publications can reprint one original series. Keep these candidates
    # under B; independent-study status requires more than a different citation.
    series=defaultdict(lambda: {'points':set(),'temperatures':defaultdict(set)})
    for a,b,cat,*rest in pairs:
        if cat!='C':continue
        ra,rb=records[a],records[b]
        if not math.isclose(ra.g,rb.g,rel_tol=1e-12,abs_tol=0):continue
        studies=tuple(sorted((ra.study,rb.study)));key=(ra.solute_key,ra.solvent_key)
        series[studies]['points'].add((key,round(ra.t,6),format(ra.g,'.12g')))
        series[studies]['temperatures'][key].add(round(ra.t,6))
    suspect={studies for studies,s in series.items() if len(s['points'])>=10 and any(len(ts)>=3 for ts in s['temperatures'].values())}
    for pair in pairs:
        a,b=pair[:2];studies=tuple(sorted((records[a].study,records[b].study)))
        if pair[2]=='C' and studies in suspect:
            pair[2]='B';pair[3]='文献不同但至少10个点、含3温度序列的值完全一致，需核对转载关系'
    # Missing conditions must never transitively bridge different pressure/phase series.
    exact=Union(len(records))
    for a,b,cat,*rest in pairs:
        if cat=='A':exact.join(a,b)
    ambiguous={}
    for group in exact.groups():
        if len(group)<2:continue
        for a,b in itertools.combinations(group,2):
            ok,near,_=conditions(records[a],records[b],args)
            if not ok or near:ambiguous[exact.find(a)]='缺失条件使原实验对应不唯一，未建立共用canonical';break
    for a,b,cat,*rest in pairs:
        if cat in ['B','D'] and exact.find(a)==exact.find(b):
            ambiguous[exact.find(a)]='重复对应存在多条不同测量或疑似/冲突记录，未建立共用canonical'
    for pair in pairs:
        if pair[2]=='A' and exact.find(pair[0]) in ambiguous:
            pair[2]='B';pair[3]=ambiguous[exact.find(pair[0])]
    canonical=Union(len(records));leakage=Union(len(records))
    for a,b,cat,*rest in pairs:
        records[a].classes.add(cat);records[b].classes.add(cat)
        if cat=='A':canonical.join(a,b)
        if cat in ['A','B','D']:leakage.join(a,b)
    for group in leakage.groups():
        lid=digest(sorted(records[i].key for i in group),'LEAK_')
        for i in group:records[i].leakage=lid
    groups=canonical.groups()
    for group in groups:
        cid=digest(min(records[i].key for i in group),'CAN_')
        for i in group:records[i].canonical=cid
    assert sum(len(g) for g in groups)==len(records)
    print('MATCH',len(records),'records;',len(pairs),'classified pairs;',len(groups),'canonical records',flush=True)
    return pairs,groups,skipped

def column(label,width=18,kind='text',hidden=False):return {'label':label,'width':width,'kind':kind,'hidden':hidden}
def write_table(work,name,columns,data,freeze=1):
    path=work/(name+'.ndjson');count=0;samples=[]
    with path.open('w',encoding='utf-8',newline='') as f:
        for row in data:
            assert len(row)==len(columns),(name,len(row),len(columns))
            for value in row:
                if isinstance(value,str) and len(value)>32767:raise RuntimeError(f'{name}: source history exceeds Excel cell limit; no truncation performed')
            f.write(dumps(row)+'\n');count+=1
            if len(samples)<8:samples.append(row)
    if count+1>1048576:raise RuntimeError(f'{name}: Excel row limit exceeded')
    return {'name':name,'columns':columns,'rows':count,'file':str(path.resolve()),'samples':samples,'freeze':freeze}

def prepare_tables(work,records,pairs,groups,metadata,headers,auxiliary,args,db,skipped):
    tables=[];counts=Counter(p[2] for p in pairs);involved={c:{i for p in pairs if p[2]==c for i in p[:2]} for c in CLASSES}
    review_records={r.i for r in records if 'B' in r.classes or 'D' in r.classes or not r.study or not r.identity_strong}
    summary=[['原始实验记录数',len(records)],['canonical记录数',len(groups)],['共用canonical的原始记录数',sum(len(g) for g in groups if len(g)>1)],['原记录删除数',0],['A 完全重复（记录对）',counts['A']],['B 高度疑似重复（记录对）',counts['B']],['C 同条件独立实验（记录对）',counts['C']],['D 冲突（记录对）',counts['D']],['原研究未识别的记录数',sum(not r.study for r in records)],['结构仅名称对应的记录数',sum(not r.identity_strong for r in records)],['待核对原始记录数',len(review_records)],['γ∞近值相对阈值',args.gamma_relative_tolerance],['γ∞近值绝对阈值',args.gamma_absolute_tolerance],['温度匹配容差_K',args.temperature_tolerance],['压力匹配容差_kPa',args.pressure_tolerance],['小压力差复核上限_kPa',args.pressure_rounding_tolerance],['未进入比较的原始记录数',skipped]]
    tables.append(write_table(work,'审计汇总',[column('指标',38),column('数值',20,'number')],summary))
    stats=defaultdict(Counter);sets=defaultdict(lambda:defaultdict(set))
    for a,b,cat,*rest in pairs:
        x,y=sorted((records[a].source,records[b].source));stats[(x,y)][cat]+=1;sets[(x,y)][cat]|={a,b}
    matrix=[]
    for x,y in itertools.combinations_with_replacement(range(7),2):
        stat=stats[(x,y)];matrix.append([LABELS[x],LABELS[y],*(stat[c] for c in CLASSES),len(set().union(*sets[(x,y)].values())) if sets[(x,y)] else 0])
    tables.append(write_table(work,'来源重叠',[column('来源 A',17),column('来源 B',17),*[column(c+' '+CLASSES[c],15,'integer') for c in CLASSES],column('涉及原始记录数',16,'integer')],matrix))
    def pair_rows():
        for a,b,cat,reason,note,relative in sorted(pairs,key=lambda p:('DBAC'.index(p[2]),records[p[0]].solute_name,records[p[0]].t,records[p[0]].key,records[p[1]].key)):
            ra,rb=records[a],records[b]
            yield [cat+' '+CLASSES[cat],ra.solute_name,ra.solvent_name,ra.t,ra.g,rb.g,LABELS[ra.source],LABELS[rb.source],relative,reason,note,ra.study,rb.study,ra.original_doi,rb.original_doi,ra.original_id,rb.original_id,ra.key,rb.key,ra.canonical,rb.canonical,ra.leakage,ra.p,rb.p,ra.identity_basis,rb.identity_basis]
    pcols=[column('类别',17),column('溶质',24),column('溶剂',28),column('温度_K',11,'temperature'),column('γ∞ A',12,'gamma'),column('γ∞ B',12,'gamma'),column('来源 A',13),column('来源 B',13),column('相对差',12,'percent',True),column('判定依据',52,hidden=True),column('条件缺失/差异',52,hidden=True)]
    pcols+=[column(name,42,hidden=True) for name in ['original_study_A','original_study_B','original_DOI_A','original_DOI_B','original_record_id_A','original_record_id_B','source_record_key_A','source_record_key_B','canonical_record_id_A','canonical_record_id_B','leakage_group_id']]
    pcols+=[column('压力_A_kPa',16,'number',True),column('压力_B_kPa',16,'number',True),column('结构对应依据 A',38,hidden=True),column('结构对应依据 B',38,hidden=True)]
    tables.append(write_table(work,'重叠明细',pcols,pair_rows()))
    def canonical_rows():
        for group in sorted(groups,key=lambda g:(records[g[0]].solute_name,records[g[0]].solvent_name,records[g[0]].t or -1,records[g[0]].canonical)):
            # Select one original observation; never average or otherwise alter gamma.
            representative=min((records[i] for i in group),key=lambda r:(r.original_use!='是',not r.original_doi,r.p is None,r.source,r.key))
            cats=set().union(*(records[i].classes for i in group));status='冲突待原文复核' if 'D' in cats else '疑似待核对' if 'B' in cats else '完全重复映射' if len(group)>1 else '同条件独立实验' if 'C' in cats else '保留原始记录'
            sl=sorted({records[i].source for i in group});history=[LABELS[records[i].source]+'::'+records[i].original_id for i in sorted(group,key=lambda i:records[i].key)]
            eligible=all(records[i].original_use=='是' for i in group) and not (cats&{'B','D'}) and bool(representative.study) and all(records[i].identity_strong for i in group)
            yield [representative.solute_name,representative.solvent_name,representative.t,representative.g,len(sl),status,representative.canonical,'; '.join(LABELS[s] for s in sl),dumps(history),len(group),dumps(sorted(records[i].key for i in group)),representative.key,representative.original_doi,representative.study,representative.leakage,'可进入下一轮筛选' if eligible else '待核对/保留原质量限制',representative.p,representative.solute_key,representative.solvent_key]
    ccols=[column('溶质',24),column('溶剂',28),column('温度_K',11,'temperature'),column('γ∞',12,'gamma'),column('source_count',13,'integer'),column('状态',20)]
    ccols+=[column(name,44,hidden=True) for name in ['canonical_record_id','source_list','original_record_ids']]
    ccols+=[column('original_record_count',22,'integer',True)]
    ccols+=[column(name,44,hidden=True) for name in ['original_record_keys','representative_source_record_key','original_DOI','original_study_id','leakage_group_id','复核状态']]
    ccols+=[column('压力_kPa',16,'number',True),column('solute_identity',44,hidden=True),column('solvent_identity',44,hidden=True)]
    tables.append(write_table(work,'canonical_records',ccols,canonical_rows()))
    raw_kind={'Y':'integer','J':'integer','K':'temperature','L':'number','M':'gamma','N':'number','BH':'number'}
    raw_columns=[column(h,28,raw_kind.get(col,'text'),True) for h,col in zip(headers,RAW_COLS)]
    def source_rows():
        for r in records:
            raw=json.loads(db.execute('SELECT data FROM raw WHERE i=?',(r.i,)).fetchone()[0]);values=[]
            for col in RAW_COLS:
                kind,v=raw.get(col,['s','']);values.append(numcell(v) if kind=='n' and v else v)
            yield [LABELS[r.source],r.solute_name,r.solvent_name,r.t,r.g,','.join(sorted(r.classes)) or '未发现可分类重叠',r.canonical,r.key,r.original_id,r.sheet,r.row,r.leakage,r.study,r.original_doi,r.study_basis,r.identity_basis,'；'.join(dict.fromkeys(r.flags)),*values]
    rcols=[column('来源',13),column('溶质',24),column('溶剂',28),column('温度_K',11,'temperature'),column('γ∞',12,'gamma'),column('重叠类别',22)]
    rcols+=[column(name,42,hidden=True) for name in ['canonical_record_id','source_record_key','original_record_id','original_sheet']]
    rcols+=[column('original_row',15,'integer',True)]
    rcols+=[column(name,45,hidden=True) for name in ['leakage_group_id','original_study_id','original_DOI','原研究对应依据','结构对应依据','待核对事项']]
    tables.append(write_table(work,'原始记录',rcols+raw_columns,source_rows()))
    study_groups=defaultdict(list)
    for r in records:
        if r.study:study_groups[r.study].append(r.i)
    def study_rows():
        for study,ids in sorted(study_groups.items()):
            rs=[records[i] for i in ids];r=rs[0]
            yield [study,r.original_doi,r.study_title or r.citation,len(ids),'; '.join(sorted({LABELS[s.source] for s in rs})),'; '.join(sorted({s.study_basis for s in rs if s.study_basis})),r.citation,r.study_year,'; '.join(sorted({s.reporting_doi for s in rs if s.reporting_doi}))]
    tables.append(write_table(work,'研究对应',[column('original_study_id',48,hidden=True),column('original_DOI',28),column('原研究题录',55),column('记录数',12,'integer'),column('来源',28),column('对应依据',30),column('原始参考文献',70,hidden=True),column('原研究年份',16,'text',True),column('原表/汇编DOI',36,hidden=True)],study_rows()))
    reg=[]
    for s,m in enumerate(metadata):
        rs=[r for r in records if r.source==s]
        reg.append([m['source'],m['rows'],sum(bool(r.reporting_doi) for r in rs),sum(bool(r.original_doi) for r in rs),sum(bool(r.study) for r in rs),sum(r.identity_strong for r in rs),m['path'],m['sha256'],digest(sorted(auxiliary.items()))])
    tables.append(write_table(work,'来源登记',[column('来源',17),column('原始记录数',16,'integer'),column('原表DOI记录数',18,'integer'),column('原研究DOI记录数',20,'integer'),column('原研究已识别记录数',22,'integer'),column('结构已对应记录数',22,'integer'),column('原始工作簿',70,hidden=True),column('sha256',68,hidden=True),column('本地引用元数据快照',44,hidden=True)],reg))
    schema={'version':VERSION,'tables':tables,'parameters':vars(args)|{'project':str(args.project),'output':str(args.output),'workdir':str(args.workdir) if args.workdir else None},'input_metadata':metadata,'auxiliary_metadata':auxiliary,'summary':{'input_records':len(records),'canonical_records':len(groups),'pair_counts':dict(counts),'unresolved_study_records':sum(not r.study for r in records),'exact_represented_records':sum(len(g) for g in groups if len(g)>1),'nondeleted_original_records':len(records),'class_involved_records':{c:len(involved[c]) for c in CLASSES},'maximum_canonical_group':max(map(len,groups))}}
    (work/'schema.json').write_text(dumps(schema),encoding='utf-8')
    (work/'id_map.json').write_text(dumps({r.key:[r.canonical,r.leakage] for r in records}),encoding='utf-8')
    return schema

def create_template(work,args):
    raise RuntimeError('The formatted report workbook is not part of this code; the analysis tables are kept in the work folder.')

def xml_value(value,ref,style):
    attrs=f'r="{ref}" s="{style}"'
    if value is None or value=='':return f'<c {attrs}/>'
    if isinstance(value,dict) and 'n' in value:return f'<c {attrs}><v>{html.escape(value["n"])}</v></c>'
    if isinstance(value,(int,float)):
        if not math.isfinite(value):raise ValueError('Invalid numeric export')
        return f'<c {attrs}><v>{value}</v></c>'
    value=str(value)
    if re.search(r'[\x00-\x08\x0B\x0C\x0E-\x1F]',value):raise ValueError('Invalid XML source text')
    return f'<c {attrs} t="inlineStr"><is><t xml:space="preserve">{html.escape(value)}</t></is></c>'
def expand_template(template,output,schema):
    # Artifact Tool authors the workbook/column styles. Streaming its worksheets
    # keeps hundreds of thousands of raw rows out of the JS object graph.
    with ZipFile(template) as z,ZipFile(output,'w',ZIP_DEFLATED,compresslevel=4) as w:
        targets=dict(sheet_targets(z));replacements={targets[t['name']]:t for t in schema['tables']}
        for info in z.infolist():
            table=replacements.get(info.filename)
            if table is None:w.writestr(info,z.read(info.filename));continue
            root=E.fromstring(z.read(info.filename));data=root.find(T('sheetData'));header=data[0]
            body=data[1] if len(data)>1 else data[0];styles={re.sub(r'\d+$','',c.get('r')):c.get('s','0') for c in body}
            rowstyles={c.get('r'):c.get('s','0') for row in data for c in row} if table['name']=='审计汇总' else {}
            whole_style=next(c.get('s','0') for row in data for c in row if c.get('r')=='ZZ100')
            E.register_namespace('',S);opening=E.Element(T('worksheet'))
            sheetpr=E.SubElement(opening,T('sheetPr'));E.SubElement(sheetpr,T('outlinePr'),{'summaryRight':'0'})
            E.SubElement(opening,T('dimension'),{'ref':f'A1:{colname(len(table["columns"]))}{table["rows"]+1}'})
            views=E.SubElement(opening,T('sheetViews'));sv=E.SubElement(views,T('sheetView'),{'workbookViewId':'0','showGridLines':'0','zoomScale':'100'})
            E.SubElement(sv,T('pane'),{'ySplit':'1','topLeftCell':'A2','state':'frozen','activePane':'bottomLeft'})
            E.SubElement(sv,T('selection'),{'pane':'bottomLeft','activeCell':'A2','sqref':'A2'})
            E.SubElement(opening,T('sheetFormatPr'),{'defaultRowHeight':'22','outlineLevelCol':'1'})
            cols=E.SubElement(opening,T('cols'));first_hidden=next((i for i,c in enumerate(table['columns'],1) if c['hidden']),None)
            for i,c in enumerate(table['columns'],1):
                attrs={'min':str(i),'max':str(i),'width':str(c['width']+.71),'customWidth':'1'}
                if c['hidden']:attrs.update(hidden='1',outlineLevel='1')
                elif first_hidden and i==first_hidden-1:attrs['collapsed']='1'
                E.SubElement(cols,T('col'),attrs)
            prefix=E.tostring(opening,encoding='utf-8',xml_declaration=True);prefix=prefix[:prefix.rfind(b'</')]+b'<sheetData>'
            with w.open(info.filename,'w') as sink:
                sink.write(prefix);sink.write(E.tostring(header,encoding='utf-8'))
                with Path(table['file']).open(encoding='utf-8') as f:
                    for n,line in enumerate(f,2):
                        values=json.loads(line);visible=[(v,c['width']) for v,c in zip(values,table['columns']) if not c['hidden'] and c['kind']=='text']
                        lines=max([1]+[math.ceil(sum(1.9 if ord(ch)>127 else .85 for ch in str(v or ''))/width) for v,width in visible])
                        height=min(100,max(22,lines*13+5))
                        cells=[]
                        for i,(value,c) in enumerate(zip(values,table['columns']),1):
                            ref=f'{colname(i)}{n}';style=rowstyles.get(ref,styles.get(colname(i),'0'))
                            numeric=number(value.get('n')) if isinstance(value,dict) else float(value) if isinstance(value,(int,float)) else None
                            if table['name']!='审计汇总' and c['kind'] in ['gamma','number'] and numeric is not None and numeric.is_integer() and abs(numeric)<100000:style=whole_style
                            cells.append(xml_value(value,ref,style))
                        cells=''.join(cells)
                        sink.write(f'<row r="{n}" ht="{height}" customHeight="1">{cells}</row>'.encode())
                end=colname(len(table['columns']));sink.write(f'</sheetData><autoFilter ref="A1:{end}{table["rows"]+1}"/><pageMargins left="0.3" right="0.3" top="0.5" bottom="0.5" header="0.2" footer="0.2"/></worksheet>'.encode())

def run_analysis(project,work,args):
    db=sqlite3.connect(work/'audit.sqlite')
    records,metadata,headers=load_inputs(project,db)
    auxiliary=resolve_studies(project,records,db)
    pairs,groups,skipped=audit(records,args)
    schema=prepare_tables(work,records,pairs,groups,metadata,headers,auxiliary,args,db,skipped)
    for m in metadata:assert file_hash(project/m['path'])==m['sha256'],'Input changed during audit'
    for p,h in auxiliary.items():assert file_hash(project/p)==h,'Reference metadata changed during audit'
    db.close();print(dumps(schema['summary']),flush=True);return schema
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sibling=Path(__file__).resolve().parent.parent
    default_project=next((p for p in [sibling,Path.home()/'Desktop/Surrogate_Modelling'] if all((p/f/'03_ISOMERS.xlsx').exists() for f in SOURCES)),sibling)
    parser.add_argument('--project',type=Path,default=default_project)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--gamma-relative-tolerance',type=float,default=.005)
    parser.add_argument('--gamma-absolute-tolerance',type=float,default=0)
    parser.add_argument('--temperature-tolerance',type=float,default=1e-6)
    parser.add_argument('--pressure-tolerance',type=float,default=.01)
    parser.add_argument('--pressure-rounding-tolerance',type=float,default=2)
    parser.add_argument('--workdir',type=Path);parser.add_argument('--analysis-only',action='store_true')
    parser.add_argument('--export-analysis',type=Path,help='Export a previously completed analysis directory')
    args=parser.parse_args();args.project=args.project.resolve()
    if args.analysis_only and not (args.workdir or args.export_analysis):parser.error('--analysis-only requires --workdir so the analysis is retained')
    args.output=(args.output or args.project/'08_Dataset_Curation/01_SOURCE_OVERLAP.xlsx').resolve()
    if any(args.output==(args.project/f/'03_ISOMERS.xlsx').resolve() for f in SOURCES):parser.error('The audit output must not replace a source workbook')
    for value in [args.gamma_relative_tolerance,args.gamma_absolute_tolerance,args.temperature_tolerance,args.pressure_tolerance,args.pressure_rounding_tolerance]:
        if value<0 or not math.isfinite(value):parser.error('Tolerances must be finite and non-negative')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.workdir or args.export_analysis:
        work=(args.export_analysis or args.workdir).resolve();work.mkdir(parents=True,exist_ok=True)
        schema=json.loads((work/'schema.json').read_text(encoding='utf-8')) if args.export_analysis else run_analysis(args.project,work,args)
        if not args.analysis_only:
            template=create_template(work,args);staged=work/'final.xlsx';expand_template(template,staged,schema)
            os.replace(staged,args.output)
            from workbook_provenance_notes import add_notes
            add_notes(args.output)
            print('SAVED',args.output,flush=True)
    else:
        with tempfile.TemporaryDirectory(prefix='overlap_audit_',dir=args.output.parent) as tmp:
            work=Path(tmp);schema=run_analysis(args.project,work,args)
            if not args.analysis_only:
                template=create_template(work,args);staged=work/'final.xlsx';expand_template(template,staged,schema)
                os.replace(staged,args.output)
                from workbook_provenance_notes import add_notes
                add_notes(args.output)
                print('SAVED',args.output,flush=True)
if __name__=='__main__':
    raise SystemExit('Import-only scientific utilities; use the two paired 08 workbook entrypoints.')
