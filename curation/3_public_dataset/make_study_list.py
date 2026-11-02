"""Write data/studies.csv: one reference per original study of the dataset.

Usage: python make_study_list.py

A study can be listed under several identifiers and reference strings in the sources
(../bibliography_all_listings.csv). For each study of data/activity_coefficients.csv, the
reference is taken from the listings of its identifier and of the identifiers merged into it,
preferring the NIST databases, then individual papers, then compilations.
"""
import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / 'data'


def read(path):
    with open(path, encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def rank(sources):
    if 'ThermoML' in sources or 'ILThermo' in sources:
        return 0
    return 1 if sources in ('Individual papers', 'Sherman') else 2


listings = [r for r in read(HERE.parent / 'bibliography_all_listings.csv') if r['Reference']]
merged, doi = {}, {}
for r in read(DATA / 'activity_coefficients.csv'):
    sid = r['Study_ID']
    if not sid:
        continue
    merged.setdefault(sid, set())
    if r['Study_ID_Before_Merge']:
        merged[sid].add(r['Study_ID_Before_Merge'])
    if r['Original_DOI'] and sid not in doi:
        doi[sid] = r['Original_DOI']

rows = []
for sid in sorted(merged):
    ids = {sid} | merged[sid]
    found = sorted((x for x in listings if x['Study_ID'] in ids), key=lambda x: rank(x['Sources']))
    refs = list(dict.fromkeys(x['Reference'].strip() for x in found))
    d = doi.get(sid, sid[4:] if sid.startswith('DOI:') else '')
    rows.append({'Study_ID': sid, 'DOI': d, 'Reference': refs[0] if refs else '', 'Other_reference_strings': ' || '.join(refs[1:])})

with open(DATA / 'studies.csv', 'w', encoding='utf-8', newline='') as f:
    w = csv.DictWriter(f, list(rows[0]), lineterminator='\n')
    w.writeheader()
    w.writerows(rows)
print(len(rows), 'studies')
