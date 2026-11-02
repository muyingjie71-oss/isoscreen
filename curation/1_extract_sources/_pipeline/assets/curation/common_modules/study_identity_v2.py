"""Normalize reported study identifiers from unique original bibliography only.

No target values, temperature, solute, solvent or measurement methods contribute
to study identity. Unknown studies are upgraded only when a complete original
bibliography uniquely links them to an already confirmed DOI.
"""
from pathlib import Path
from collections import Counter,defaultdict
import json,re,sys
import source_overlap_audit as a


def _doi_study(record):
    study=str(record.get('Study_ID',''))
    if not study.startswith('DOI:'):return ''
    doi=a.doi(study[4:])
    original=a.doi(record.get('Original_DOI',''))
    return doi if doi and original==doi and doi not in a.SECONDARY_DOIS else ''


def _bibliographies(record):
    """Parse complete journal/volume/first-page declarations, preserving evidence."""
    result=[]
    for field in ('Source_Original_Reference','Raw_其他固定条件','Reference'):
        text=str(record.get(field,'') or '')
        if not text:continue
        # Fixed-condition strings may contain explicit reference=... provenance;
        # a complete bibliography is required even when such a marker exists.
        bib=a.bibliographic(text)
        if bib:result.append((tuple(bib),field,text))
    return result


def _reported_bib(study):
    if not study.startswith('BIB:'):return None
    try:
        j,vol,page=json.loads(study[4:])
        return (a.journal(j),str(vol),str(page).lower())
    except (ValueError,TypeError):return None


def normalize(records,promote_unknown=True):
    title_to_dois=defaultdict(set);bib_to_dois=defaultdict(set)
    title_evidence=defaultdict(list);bib_evidence=defaultdict(list)
    for r in records:
        doi=_doi_study(r)
        if not doi:continue
        if a.doi(r.get('Raw_DOI',''))==doi:
            title=str(r.get('Raw_文献标题','') or '')
            key=a.title_key(title)
            if len(key)>=30:
                title_to_dois[key].add(doi)
                title_evidence[key].append(dict(source_record_key=r.get('source_record_key',''),
                    DOI=doi,field='Raw_文献标题',text=title,
                    evidence='Raw_DOI == confirmed Original_DOI == DOI Study_ID'))
        for bib,field,text in _bibliographies(r):
            bib_to_dois[bib].add(doi)
            bib_evidence[bib].append(dict(source_record_key=r.get('source_record_key',''),DOI=doi,field=field,text=text))
    normalized=[];updates=[];counts=Counter()
    for original in records:
        r=dict(original);study=str(r.get('Study_ID',''));reported=r.get('Reported_Study_ID',study)
        r['Reported_Study_ID']=reported
        r['Study_Identity_Basis']=r.get('Study_Identity_Basis','Original reported study identifier; no value-based inference')
        if study.startswith('DOI:'):
            normalized.append(r);counts['unchanged_doi_records']+=1;continue
        candidate=set();bases=[];evidence=[]
        if study.startswith('TITLE:'):
            key=a.title_key(study[6:])
            possible=title_to_dois.get(key,set())
            if len(possible)==1:
                candidate.update(possible);bases.append('Exact full original title matches a unique DOI-backed original Raw title')
                evidence.extend(title_evidence[key][:3])
            elif len(possible)>1:counts['ambiguous_title_records']+=1
        bib=_reported_bib(study)
        if bib:
            possible=bib_to_dois.get(bib,set())
            if len(possible)==1:
                candidate.update(possible);bases.append('Exact original journal/volume/first-page bibliography matches a unique confirmed DOI')
                evidence.extend(bib_evidence[bib][:3])
            elif len(possible)>1:counts['ambiguous_bibliography_records']+=1
        if promote_unknown and not study.startswith(('DOI:','TITLE:','BIB:')):
            for bib,field,text in _bibliographies(r):
                possible=bib_to_dois.get(bib,set())
                if len(possible)==1:
                    candidate.update(possible);bases.append('Complete original bibliography in '+field+' uniquely matches a confirmed DOI')
                    evidence.extend(bib_evidence[bib][:3])
        if len(candidate)==1:
            doi=next(iter(candidate))
            # An existing contradictory confirmed Original_DOI prevents rewrite.
            olddoi=a.doi(r.get('Original_DOI',''))
            if olddoi and olddoi!=doi:
                counts['contradictory_original_doi_records']+=1
            else:
                r['Study_ID']='DOI:'+doi;r['Original_DOI']=doi
                r['Study_Identity_Basis']='; '.join(bases)
                r['Study_Identity_Evidence']=json.dumps(evidence,ensure_ascii=False)
                update=dict(canonical_record_id=r['canonical_record_id'],source_record_key=r.get('source_record_key',''),
                    Reported_Study_ID=reported,Study_ID=r['Study_ID'],Study_Identity_Basis=r['Study_Identity_Basis'],evidence=evidence)
                updates.append(update);counts['normalized_'+study.split(':',1)[0].lower()+'_records']+=1
        elif len(candidate)>1:counts['incompatible_title_bibliography_candidates']+=1
        normalized.append(r)
    counts['input_records']=len(records);counts['updates']=len(updates)
    counts['unique_title_keys_with_doi']=len(title_to_dois)
    counts['unique_bibliography_keys_with_doi']=len(bib_to_dois)
    return dict(records=normalized,updates=updates,stats=dict(counts))


if __name__=='__main__':
    import argparse,csv
    p=argparse.ArgumentParser();p.add_argument('--input',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--promote-unknown',action='store_true');args=p.parse_args()
    with args.input.open(encoding='utf-8-sig',newline='') as f:records=list(csv.DictReader(f))
    result=normalize(records,promote_unknown=args.promote_unknown)
    fields=list(dict.fromkeys(k for r in result['records'] for k in r))
    with args.output.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(result['records'])
    args.output.with_suffix('.study_updates.json').write_text(json.dumps(result['updates'],ensure_ascii=False,indent=2),encoding='utf8')
    args.output.with_suffix('.study_report.json').write_text(json.dumps(result['stats'],ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(result['stats'],ensure_ascii=False,indent=2))
