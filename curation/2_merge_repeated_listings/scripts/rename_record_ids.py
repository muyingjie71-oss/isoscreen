"""Give the record IDs in the tables written by this step the prefix REC_.

Step 1 names each record by a prefix and a hash of the record. The hash part is kept, so every record
can still be found in the step 1 tables in inputs/. Pair and matched-group IDs are computed from the step 1
record IDs (export_tables.py, make_training_tables.py) before this renaming, and carry the prefixes PAIR_
and GROUP_.
"""
import csv
import sys
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from paths import O, P

COLUMNS = ('canonical_record_id', 'copy_cluster', 'canonical_record_id_A', 'canonical_record_id_B')


def rec(x):
    return 'REC_' + x.split('_', 1)[1] if x else x


def rename(path):
    with open(path, 'rb') as f:
        encoding = 'utf-8-sig' if f.read(3) == b'\xef\xbb\xbf' else 'utf-8'
    with open(path, encoding=encoding, newline='') as f:
        rows = list(csv.reader(f))
    head = rows[0]
    idx = [head.index(c) for c in COLUMNS if c in head]
    dec = head.index('Decision') if 'Decision' in head else None
    for r in rows[1:]:
        for i in idx:
            r[i] = rec(r[i])
        if dec is not None and r[dec].startswith('MERGED_INTO_'):
            r[dec] = 'MERGED_INTO_' + rec(r[dec][len('MERGED_INTO_'):])
    with open(path, 'w', encoding=encoding, newline='') as f:
        csv.writer(f, lineterminator='\n').writerows(rows)


for name in ('record_decisions.csv', 'task1_merged.csv', 'task2_merged.csv'):
    rename(O + name)
for name in ('task1.csv', 'task2.csv'):
    rename(P + 'data/' + name)
print('record IDs renamed to REC_')
