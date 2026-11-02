"""Write data/isomer_selectivity_dataset.xlsx, a readable version of the CSV files in data/.

Usage: python make_workbook.py

One workbook with a README sheet that defines every column, structures next to each value
and references written out. The CSV files remain the reference version.
"""
import collections
import json
import math
import zipfile
from pathlib import Path

import pandas as pd
from openpyxl import Workbook
from openpyxl.cell import WriteOnlyCell
from openpyxl.styles import Alignment, Font, PatternFill

DATA = Path(__file__).resolve().parents[2] / 'data'
OUT = DATA / 'isomer_selectivity_dataset.xlsx'
read = lambda name: pd.read_csv(DATA / name, dtype=str, keep_default_na=False)
t1 = read('activity_coefficients.csv')
t2 = read('isomer_pairs.csv')
mol = read('molecules.csv').set_index('Mol_ID')
solv = read('solvents.csv').set_index('Solvent_Mol_ID')
corr = read('structure_corrections.csv')
refs = read('studies.csv').set_index('Study_ID')
solvent_fold = json.loads((DATA / 'folds_new_solvent_test.json').read_text())['outer_assignment_solvent']

FLAG = {'': '', 'DERIVED_FROM_OTHER_PROPERTY': 'Converted from another property',
        'SOLVENT_LIQUID_CRYSTAL_PHASE': 'Solvent in liquid-crystal phase',
        'SOLUTE_GAS_PHASE_LABEL': 'Solute labelled gaseous in source'}


def reference(study, doi):
    if not study and not doi:
        return 'Original study not identified (listed only in a compilation)', ''
    if study in refs.index:
        r = refs.loc[study]
        d = r.DOI or doi
        text = r.Reference if r.Reference and r.Reference != d else (f'doi:{d}' if d else '')
        if text:
            return text, (f'https://doi.org/{d}' if d else '')
    if doi:
        return f'doi:{doi}', f'https://doi.org/{doi}'
    return study, ''


names = collections.defaultdict(collections.Counter)
for a, b, c, d in zip(t1.Solute_Mol_ID, t1.Solute, t1.Solvent_Mol_ID, t1.Solvent):
    names[a][b] += 1
    names[c][d] += 1
name = {k: v.most_common(1)[0][0] for k, v in names.items()}
kind = lambda sid: 'Ionic liquid' if solv.loc[sid, 'Is_Ionic_Liquid'] == 'True' else 'Molecular solvent'
fold = lambda f: f.replace('Fold_', '')
sig = lambda x, n=6: float(f'{x:.{n}g}')

BOLD = Font(name='Arial', size=10, bold=True, color='FFFFFF')
FILL = PatternFill('solid', fgColor='2C2C2A')
wb = Workbook(write_only=True)


def sheet(title, cols, rows, widths):
    ws = wb.create_sheet(title)
    for i, w in enumerate(widths):
        ws.column_dimensions[chr(65 + i)].width = w
    ws.freeze_panes = 'A2'
    ws.auto_filter.ref = f'A1:{chr(64 + len(cols))}{len(rows) + 1}'
    head = []
    for c in cols:
        x = WriteOnlyCell(ws, value=c)
        x.font, x.fill, x.alignment = BOLD, FILL, Alignment(wrap_text=True, vertical='center')
        head.append(x)
    ws.append(head)
    for r in rows:
        ws.append(r)


rows1 = []
for r in t1.itertuples(index=False):
    ref, url = reference(r.Study_ID, r.Original_DOI)
    ln = float(r.ln_gamma_inf)
    rows1.append([r.canonical_record_id, r.Family_ID.replace('ISO_', ''), r.Solute, mol.loc[r.Solute_Mol_ID, 'SMILES'],
                  r.Solvent, kind(r.Solvent_Mol_ID), mol.loc[r.Solvent_Mol_ID, 'SMILES'], float(r.T_K), sig(math.exp(ln)), ln,
                  ref, url, r.Source, r.Reported_Method, r.Reported_Phase, r.Reported_Standard_State,
                  FLAG[r.Special_Record_Flag], int(r.Merged_Copy_Count), fold(r.Fold_ID), fold(solvent_fold.get(r.Solvent_Mol_ID, '')),
                  r.Solute_Mol_ID, r.Solvent_Mol_ID])
cols1 = ['Record ID', 'Isomer family (formula)', 'Solute', 'Solute SMILES', 'Solvent', 'Solvent type', 'Solvent SMILES', 'T (K)',
         'γ∞', 'ln γ∞', 'Reference', 'Link', 'Source database', 'Method (as reported)', 'Phase (as reported)',
         'Standard state (as reported)', 'Flag', 'Listings merged', 'Fold, new-isomer test', 'Fold, new-solvent test',
         'Solute Mol ID', 'Solvent Mol ID']

record = t1.set_index('canonical_record_id')
rows2 = []
for r in t2.itertuples(index=False):
    ga, gb, d = float(r.gamma_A), float(r.gamma_B), float(r.D_AB)
    ref, url = reference(r.Study_ID, record.loc[r.canonical_record_id_A].Original_DOI)
    rows2.append([r.Pair_ID, r.Family_ID.replace('ISO_', ''), r.Isomer_Relation, r.Isomer_A, r.Isomer_B, r.Solvent,
                  kind(r.Solvent_Mol_ID), float(r.T_K), ga, gb, sig(math.exp(abs(d)), 5),
                  r.Isomer_A if ga < gb else (r.Isomer_B if gb < ga else 'Neither'), ref, url,
                  r.canonical_record_id_A, r.canonical_record_id_B, fold(r.Fold_ID), fold(solvent_fold.get(r.Solvent_Mol_ID, ''))])
cols2 = ['Pair ID', 'Isomer family (formula)', 'Type of isomerism', 'Isomer A', 'Isomer B', 'Solvent', 'Solvent type', 'T (K)',
         'γ∞ of A', 'γ∞ of B', 'Selectivity S∞', 'Isomer retained more strongly', 'Reference', 'Link',
         'Record ID of A', 'Record ID of B', 'Fold, new-isomer test', 'Fold, new-solvent test']

as_solute, as_solvent = t1.Solute_Mol_ID.value_counts(), t1.Solvent_Mol_ID.value_counts()
family = dict(zip(t1.Solute_Mol_ID, t1.Family_ID.str.replace('ISO_', '')))
rows3 = []
for mid, r in mol.iterrows():
    role = ' and '.join(x for x, n in (('solute', as_solute.get(mid, 0)), ('solvent', as_solvent.get(mid, 0))) if n)
    rows3.append([mid, name.get(mid, ''), r.SMILES, r.InChIKey, role, kind(mid) if mid in solv.index else 'Molecular solute',
                  family.get(mid, ''), int(as_solute.get(mid, 0)), int(as_solvent.get(mid, 0))])
rows3.sort(key=lambda x: (x[5], x[1].lower()))
cols3 = ['Mol ID', 'Name', 'SMILES', 'InChIKey', 'Role', 'Type', 'Isomer family (solutes)', 'Records as solute', 'Records as solvent']

pairs_per_study = t2.Study_ID.value_counts()
rows4 = []
for sid, x in t1.groupby('Study_ID'):
    ref, url = reference(sid, next((d for d in x.Original_DOI if d), ''))
    rows4.append([sid or '(not identified)', ref, url, len(x), int(pairs_per_study.get(sid, 0)), x.Solute_Mol_ID.nunique(),
                  x.Solvent_Mol_ID.nunique(), '; '.join(sorted(set(x.Source)))])
rows4.sort(key=lambda x: -x[3])
cols4 = ['Study ID', 'Reference', 'Link', 'Records', 'Isomer pairs', 'Solutes', 'Solvents', 'Listed in']

rows5 = [[r.Mol_ID, name.get(r.Mol_ID, ''), r.Problem, r.SMILES_As_Analysed, r.SMILES_Corrected, r.InChIKey_As_Analysed,
          r.InChIKey_Corrected] for r in corr.itertuples(index=False)]
cols5 = ['Mol ID', 'Name', 'Problem', 'SMILES used in the analyses', 'Corrected SMILES', 'InChIKey used in the analyses',
         'Corrected InChIKey']

n_studies = t1.Study_ID[t1.Study_ID != ''].nunique()
n_unknown = int((t1.Study_ID == '').sum())
readme = [
    'Infinite-dilution activity coefficients and isomer selectivities',
    '',
    'Dataset of the paper "Resolution limits of predicted selectivities in isomer separation" (submitted to AIChE Journal).',
    'Repository: https://github.com/muyingjie71-oss/isoscreen    Archive: https://doi.org/10.5281/zenodo.23877328    Web tool: https://muyingjie71-oss.github.io/isoscreen/',
    'Licence: CC BY 4.0. The CSV files in data/ of the repository are the reference version, and this workbook shows the same data.',
    '',
    'Contents',
    f'Activity coefficients: {len(rows1):,} measured γ∞ of {t1.Solute_Mol_ID.nunique()} solutes in {t1.Solvent_Mol_ID.nunique()} solvents '
    f'from {n_studies} identified studies, one row per measurement. For {n_unknown:,} records, listed only in a compilation, the original study is not identified.',
    f'Isomer pairs: {len(rows2):,} pairs of isomers measured in the same study, solvent and temperature, with their selectivity.',
    f'Molecules: {len(rows3):,} solutes and solvents with structures.',
    f'Studies: the {n_studies} identified original studies with their references and numbers of records, and one row for records whose study is not identified.',
    f'Corrections: {len(rows5)} solvent structures corrected after the analyses. Values, records, pairs and folds are not affected.',
    '',
    'Definitions',
    'γ∞: activity coefficient of the solute at infinite dilution in the solvent, as reported. ln γ∞ is the value in the source, and γ∞ is rounded to six significant digits.',
    'Selectivity S∞: ratio of the larger to the smaller γ∞ of the two isomers, so S∞ ≥ 1. The solvent retains the isomer with the smaller γ∞ more strongly.',
    'Isomer family: solutes with the same molecular formula. Isomer pair: two records of different isomers of one family from the same study, solvent, temperature and conditions.',
    'Reference: the original study as cited in the source databases (authors, year, journal, volume, pages), with a DOI link where known.',
    'Source database: where the record was taken from. When several databases listed the same measurement, one record was kept, and Listings merged gives how many listings it represents.',
    'Method, phase and standard state are given as reported by the source and are empty when the source does not report them.',
    'Flag: records kept but marked for sensitivity analyses (values converted from another property, solvent in a liquid-crystal phase, solute labelled gaseous in the source).',
    'Fold, new-isomer test: cross-validation fold (1–5) of the isomer family. Fold, new-solvent test: fold (1–5) of the solvent. Records of one fold are tested by models trained on the other folds.',
    'Type of isomerism: skeletal (chain branching), positional, ring-chain, metameric, functional group, skeletal (ring system) or unsaturation type.',
    'Mol ID: identifier assigned from the InChIKey at curation. It is kept for corrected structures, so for one corrected solvent it differs from the current InChIKey.',
    '',
    'Sources',
    'NIST ThermoML archive (version 2020-09-30). NIST Ionic Liquids Database, ILThermo. Brouwer et al., Sep. Purif. Technol. 2021, 272, 118727. Tan et al., AIChE J. 2022, 68, e17789.',
    'Li et al., ACS Sustainable Chem. Eng. 2022, 10, 13927. Sherman et al., Ind. Eng. Chem. Res. 1996, 35, 1044. Individual papers collected by the authors.',
]

ws = wb.create_sheet('README')
ws.column_dimensions['A'].width = 160
for i, line in enumerate(readme):
    c = WriteOnlyCell(ws, value=line)
    c.font = Font(name='Arial', size=12 if i == 0 else 10,
                  bold=i == 0 or line in ('Contents', 'Definitions', 'Sources'))
    ws.append([c])

sheet('Activity coefficients', cols1, rows1, [27, 12, 24, 30, 34, 16, 34, 9, 10, 10, 30, 14, 22, 18, 12, 18, 18, 9, 9, 9, 30, 30])
sheet('Isomer pairs', cols2, rows2, [33, 12, 24, 24, 24, 34, 16, 9, 10, 10, 11, 24, 30, 14, 27, 27, 9, 9])
sheet('Molecules', cols3, rows3, [32, 34, 40, 30, 20, 18, 14, 10, 10])
sheet('Studies', cols4, rows4, [40, 40, 14, 10, 10, 10, 10, 40])
sheet('Corrections', cols5, rows5, [32, 34, 46, 50, 50, 30, 30])
tmp = OUT.with_suffix('.tmp.xlsx')
wb.save(tmp)

# Arial as the default font (write-only workbooks keep Calibri as the Normal style)
with zipfile.ZipFile(tmp) as zi, zipfile.ZipFile(OUT, 'w', zipfile.ZIP_DEFLATED) as zo:
    for item in zi.infolist():
        data = zi.read(item.filename)
        if item.filename == 'xl/styles.xml':
            data = data.replace(b'Calibri', b'Arial')
        zo.writestr(item, data)
tmp.unlink()
print(OUT.name, len(rows1), len(rows2), len(rows3), len(rows4))
