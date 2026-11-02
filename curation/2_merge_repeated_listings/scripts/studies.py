import pandas as pd, numpy as np, re, json, collections, itertools
def norm(s): return re.sub(r'[^a-z0-9]','',str(s).lower())
def study_meta(d):
    c=d[d.Study_ID.notna()]
    f=lambda col: c.groupby('Study_ID')[col].agg(lambda x: x.dropna().iloc[0] if x.notna().any() else '')
    M=pd.DataFrame({'title':f('Raw_文献标题'),'reference':f('Reference'),'year':f('Raw_发表年份'),'n':c.groupby('Study_ID').size()})
    M['srcs']=c.groupby('Study_ID').source_list.agg(lambda x:'; '.join(sorted(set('; '.join(x).split('; ')))))
    return M
def bib(sid):
    if sid.startswith('BIB:'):
        try: j,v,p=json.loads(sid[4:]); return norm(j),str(v),str(p)
        except Exception: return None
    return None
def title_key(sid,M):
    if sid.startswith('TITLE:'): return sid[6:]
    t=M.at[sid,'title'] if sid in M.index else ''
    if not t: t=M.at[sid,'reference'] if sid in M.index else ''
    return norm(t)
COMPILATION_PAT=re.compile(r'compilation|review|data ?base|critical evaluation',re.I)
def classify_link(a,b,M):
    ba,bb=bib(a),bib(b)
    if ba and bb and ba[1]==bb[1] and (ba[2].startswith(bb[2]) or bb[2].startswith(ba[2])) and (ba[0].startswith(bb[0]) or bb[0].startswith(ba[0])):
        return 'ALIAS_SAME_PUBLICATION','BIB journal/volume/first page consistent'
    ta,tb=title_key(a,M),title_key(b,M)
    if len(ta)>=30 and len(tb)>=30 and (ta[:40]==tb[:40]):
        return 'ALIAS_SAME_PUBLICATION','normalized title identical (first 40 chars)'
    # BIB vs TITLE/DOI: match by reference text containing journal vol page
    for x,y in ((a,b),(b,a)):
        bx=bib(x)
        if bx and len(title_key(y,M))>=30:
            ry=norm(M.at[y,'reference']) if y in M.index else ''
            rx=norm(M.at[x,'reference']) if x in M.index else ''
            if title_key(y,M)[:40] and title_key(y,M)[:40] in norm(M.at[x,'title'] if x in M.index else ''):
                return 'ALIAS_SAME_PUBLICATION','title of one equals title recorded under the other'
    ra,rb=str(M.at[a,'reference']) if a in M.index else '',str(M.at[b,'reference']) if b in M.index else ''
    if COMPILATION_PAT.search(ra) or COMPILATION_PAT.search(rb) or COMPILATION_PAT.search(str(M.at[a,'title'] if a in M.index else '')) or COMPILATION_PAT.search(str(M.at[b,'title'] if b in M.index else '')):
        return 'COMPILATION_COPY','one study is a data compilation/review'
    return 'REPUBLICATION_OR_REUSE','different publications reporting identical values'
