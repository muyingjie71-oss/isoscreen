"""Unify study identifiers and cluster repeated listings of one measurement. Reads the extracted tables only."""
import sys, re, json, itertools, collections
import pandas as pd, numpy as np
sys.path.insert(0,__import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from pairing import sig, compat
TOL=0.005   # |dln gamma| <= 0.005 == the extracted 0.5% merge rule
PTOL=0.5    # kPa: 101.3 vs 101.33 vs 101.325 treated as the same nominal pressure
# Manually reviewed same-publication aliases (each verified by title / journal-volume-page; see review table)
ALIAS=[('BIB:["jmolliq","289","111"]','BIB:["jmolliq","289","111169"]'),
('DOI:10.1063/1.4864182','TITLE:understandingtheimpactofthecentralatomontheionicliquidbehaviorphosphoniumvsammoniumcations'),
('DOI:10.1016/j.fluid.2018.09.024','TITLE:ammoniumionicliquidsinextractionofbiobutan1olfromwaterphaseusingactivitycoefficientsatinfinitedilutio'),
('DOI:10.1016/j.seppur.2016.02.015','TITLE:gammainfinitydatafortheseparationofwaterbutan1olmixturesusingionicliquids'),
('BIB:["indengchemres","50","10755"]','TITLE:activitycoefficientsoforganicsolutesatinfinitedilutioninionicliquids11hexyl3methylimidazoliumhexafluorophosphateand1octyl3methylimidazoliumhexafluorophosphateandtheirapplicationtoalkanearomaticandaromaticaromatichydrocarbonseparation'),
('BIB:["jphyschem","115","8207"]','BIB:["jphyschemb","115","8207"]'),
('BIB:["jchemthermodyn","42","1361"]','TITLE:measurementsofactivitycoefficientsatinfinitedilutionoforganicsolutesandwaterin1propyl1methylpiperidiniumbistrifluoromethylsulfonylimideionicliquidusingglc'),
('BIB:["rscadv","7","7092"]','TITLE:assessmentofcertainionicliquidsforseparationofbinarymixturesbasedongammainfinitydatameasurements'),
('DOI:10.1016/j.jct.2005.04.010','TITLE:systemswithionicliquidsmeasurementofvleandγdataandpredictionoftheirthermodynamicbehaviorusingoriginalunifacmodunifacdoandcosmorsol'),
('DOI:10.1021/je101008y','TITLE:measurementsofactivitycoefficientsatinfinitedilutionfororganicsolutesandwaterintheionicliquid1butyl1methylpiperidiniumthiocyanate'),
('BIB:["indengchemres","51","12479"]','TITLE:activitycoefficientsoforganicsolutesatinfinitedilutionintheionicliquids2organicsolutesin1hexyl3methylimidazoliumnitrateandgasliquidpartitioningandinterfacialadsorptionusinggasliquidchromatography'),
('BIB:["jchemengdata","46","1526"]','TITLE:thermodynamicpropertiesofmixturescontainingionicliquids1activitycoefficientsatinfinitedilutionofalkanesalkenesandalkylbenzenesin4methylnbutylpyridiniumtetrafluoroborateusinggasliquidchromatography'),
('DOI:10.1016/j.fluid.2008.11.013','TITLE:measurementandpredictionofactivitycoefficientsatinfinitedilutionγvaporliquidequilibriavleandexcessenthalpiesheofbinarysystemswith11dialkylpyrrolidiniumbistrifluoromethylsulfonylimideusingmodunifacdortmund'),
('BIB:["jchromatogr","1102","256"]','BIB:["jchromatogra","1102","256"]'),
('BIB:["frontiersofchemicalengineeringinchina","1","190"]','TITLE:measurementofactivitycoefficientsatinfinitedilutionforhydrocarbonsinimidazoliumbasedionicliquidsandqsprmodel'),
('BIB:["analchem","65","3212"]','REF:lietal1993a')]
COMPILATIONS={'BIB:["fluidphaseequilib","131","145"]','BIB:["indengchemres","35","1044"]'}
def is_compilation(s): return s in COMPILATIONS or str(s).startswith('REF:')
def prefix_rank(s):
    return {'DOI':0,'BIB':1,'TITLE':2,'REF':3}.get(str(s).split(':')[0],4)
class UF:
    def __init__(s): s.p={}
    def f(s,x):
        s.p.setdefault(x,x)
        while s.p[x]!=x: s.p[x]=s.p[s.p[x]]; x=s.p[x]
        return x
    def u(s,a,b): s.p[s.f(a)]=s.f(b)
def year_of(meta,sid):
    y=meta.get(sid,{}).get('year')
    try: return int(float(y))
    except Exception: pass
    m=re.search(r'\((1[89]\d\d|20\d\d)\)',str(meta.get(sid,{}).get('reference','')))
    return int(m.group(1)) if m else 9999
def run(d, links, meta):
    d=d.copy()
    # 1. alias unification of study identity
    uf=UF()
    for a,b in ALIAS: uf.u(a,b)
    canon={}
    for comp in set(uf.f(x) for x in uf.p):
        mem=[x for x in uf.p if uf.f(x)==comp]
        rep=sorted(mem,key=lambda s:(prefix_rank(s),s))[0]
        for x in mem: canon[x]=rep
    d['Study_ID_Merged']=d.Study_ID.map(lambda s: canon.get(s,s) if isinstance(s,str) else s)
    # 2. study pairs whose identical values are treated as copies
    strong={tuple(sorted((canon.get(a,a),canon.get(b,b)))) for a,b,n in zip(links.s1,links.s2,links.shared) if n>=3}
    # 3. measurement-copy clustering among model-eligible extracted records
    m=d[d.Model_Eligible=='是'].copy()
    m['lg']=np.log(m.gamma_inf.astype(float)); m['Tr']=m.T_K.round(2); m['sg']=m.Effective_Conditions.map(sig)
    uf2=UF(); reason={}
    for k,x in m.groupby(['Solute_Mol_ID','Solvent_Audit_ID','Tr']):
        if len(x)<2: continue
        r=x[['canonical_record_id','Study_ID_Merged','lg','sg']].values
        for (i1,s1,l1,g1),(i2,s2,l2,g2) in itertools.combinations(r,2):
            if abs(l1-l2)>TOL: continue
            k1,k2=isinstance(s1,str),isinstance(s2,str)
            why=None
            if k1 and k2 and s1==s2:
                if compat(g1,g2,PTOL): why='SAME_STUDY_COMPATIBLE_CONDITION'
            elif k1 and k2:
                if tuple(sorted((s1,s2))) in strong: why='LINKED_STUDIES_IDENTICAL_VALUE'
                elif is_compilation(s1) or is_compilation(s2): why='COMPILATION_COPY'
            else:
                why='UNRESOLVED_STUDY_IDENTICAL_VALUE'
            if why:
                uf2.u(i1,i2); reason[frozenset((i1,i2))]=why
    clusters=collections.defaultdict(list)
    for x in uf2.p: clusters[uf2.f(x)].append(x)
    mi=m.set_index('canonical_record_id')
    rows=[]
    for c,ids in clusters.items():
        if len(ids)<2: continue
        sub=mi.loc[ids]
        def key(i):
            r=mi.loc[i]; s=r.Study_ID_Merged; known=isinstance(s,str)
            srank=0 if 'ThermoML' in r.source_list else 1 if 'ILThermo' in r.source_list else 2
            return (0 if known else 1, 1 if (known and is_compilation(s)) else 0, year_of(meta,s) if known else 9999, prefix_rank(s) if known else 9, srank, i)
        rep=sorted(ids,key=key)[0]
        whys=sorted({reason[p] for p in reason if p<=set(ids)})
        span=sub.lg.max()-sub.lg.min()
        for i in ids:
            rows.append(dict(canonical_record_id=i,copy_cluster=rep,is_representative=i==rep,cluster_size=len(ids),cluster_reasons='|'.join(whys),cluster_lng_span=span))
    C=pd.DataFrame(rows)
    return d, canon, C
