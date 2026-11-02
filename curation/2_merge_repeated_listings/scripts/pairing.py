import json, itertools
def sig(s):
    try: v=json.loads(s)
    except Exception: v=['']*5
    return [str(x).strip().lower() for x in v]
def compat(a,b,ptol=None):
    for i,(x,y) in enumerate(zip(a,b)):
        if x=='' or y=='' or x==y: continue
        if i==0 and ptol is not None:
            try:
                if abs(float(x)-float(y))<=ptol: continue
            except ValueError: pass
        return False
    return True
def build_pairs(m, study_col='Study_ID', ptol=None, ambiguity='drop'):
    """m: DataFrame of pairable records. Returns set of frozenset id pairs."""
    out=set()
    for k,x in m.groupby(['Family_ID',study_col,'Solvent_Audit_ID','T_K'],dropna=False):
        if len(x)<2: continue
        ids=x.canonical_record_id.values; mem=x.Member_ID.values; sg=[sig(s) for s in x.Effective_Conditions.values]
        full=[s for s in sg if '' not in s]
        amb=set()
        if ambiguity=='drop':
            # a record with blanks compatible with >1 mutually-incompatible signatures is ambiguous
            distinct=[]
            for s in sg:
                if s not in distinct: distinct.append(s)
            for i,s in enumerate(sg):
                cands=[t for t in distinct if t!=s and compat(s,t,ptol)]
                for t1,t2 in itertools.combinations(cands,2):
                    if not compat(t1,t2,ptol): amb.add(i); break
            if ambiguity=='qc':
                pass
        # a member with >1 compatible record in this study/solvent/T is a repeat ambiguity -> not paired
        rep=set()
        for i,j in itertools.combinations(range(len(x)),2):
            if mem[i]==mem[j] and compat(sg[i],sg[j],ptol): rep.update((i,j))
        for i,j in itertools.combinations(range(len(x)),2):
            if mem[i]==mem[j] or i in amb or j in amb or i in rep or j in rep: continue
            if compat(sg[i],sg[j],ptol): out.add(frozenset((ids[i],ids[j])))
    return out
