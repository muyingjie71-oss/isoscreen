"""Write the public dataset (data/) from the tables used in the analyses.

Usage: python make_public_dataset.py [merge step output folder] [output folder]

Reads the training tables written by ../2_merge_repeated_listings (task1.csv, task2.csv, molecules.csv,
split_manifest.json and cross_study_replicates.csv) and, from inputs/, the solvent table and the folds
of the new-solvent test as used in the analyses.

Changes made for the public dataset:
- English file names, column names and labels, and source file paths with forward slashes.
- Corrected structures of eight solvents (list in structure_corrections.csv). Measured values, records,
  pairs, Mol_IDs and folds are unchanged. The models in the paper used the original structures,
  which training/inputs/molecules_as_trained.csv keeps.
"""
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path

from rdkit import Chem, RDLogger

RDLogger.DisableLog('rdApp.*')
HERE = Path(__file__).resolve().parent
MERGE = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / '2_merge_repeated_listings' / 'outputs'
SRC = MERGE / 'training_tables' / 'data'
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE.parents[1] / 'data'
OUT.mkdir(parents=True, exist_ok=True)

SOURCE = {'ThermoML': 'NIST ThermoML', 'ILThermo': 'NIST ILThermo', 'Brouwer': 'Brouwer et al. 2021',
          'Tan': 'Tan et al. 2022', 'DeepFM': 'Li et al. 2022', 'Sherman': 'Sherman et al. 1996',
          '单独收集论文': 'Individual papers'}
RELATION = {'骨架/链异构': 'skeletal (chain branching)', '位置异构': 'positional', '环-链异构': 'ring-chain',
            '连接/分配异构': 'metameric', '官能团异构': 'functional group', '骨架/环系异构': 'skeletal (ring system)',
            '不饱和类型异构': 'unsaturation type'}
COLUMNS = {'Raw_压力_kPa': 'Reported_Pressure_kPa', 'Raw_相态': 'Reported_Phase', 'Raw_标准态': 'Reported_Standard_State',
           'Raw_实验方法': 'Reported_Method', 'Raw_其他固定条件': 'Reported_Other_Conditions'}
SOLVENT_COLUMNS = {'smiles': 'SMILES', 'cation': 'Cation_SMILES', 'anion': 'Anion_SMILES', 'is_IL': 'Is_Ionic_Liquid',
                   'group_solvent': 'Group_Solvent', 'group_cation': 'Group_Cation', 'n_records': 'Records'}
REPLICATE_COLUMNS = {'pair': 'Isomer_Pair', 'solvent': 'Solvent_Mol_ID', 'T': 'T_K', 's1': 'Study_1', 's2': 'Study_2',
                     'diff': 'Difference', 'D1': 'D_Study_1', 'D2': 'D_Study_2'}

# Structures corrected after the analyses: Mol_ID -> (problem, SMILES, is ionic liquid, cation, anion)
P66614 = 'CCCCCCCCCCCCCC[P+](CCCCCC)(CCCCCC)CCCCCC'
CS = 'CC1(C)[C@@H]2CC[C@@]1(CS(=O)(=O)[O-])C(=O)C2'  # (1S)-camphor-10-sulfonate, as in the other measured ionic liquids
C4MIM, C3MIM, C3MMIM = 'CCCCn1cc[n+](C)c1', 'CCCn1cc[n+](C)c1', 'CCCn1cc[n+](C)c1C'
HSO3C4MIM = 'C[n+]1ccn(CCCCS(=O)(=O)O)c1'
FIX = {
    'MOL_MUVYIRZYHQNTHV-KBSFAJRQSA-N': ('anion given as the neutral ethyl ester of camphorsulfonic acid',
                                        f'{P66614}.{CS}', True, P66614, CS),
    'MOL_SBOSGIJGEHWBKV-UHFFFAOYSA-L': ('molecular organotin compound given as ions and radical fragments and labelled ionic liquid',
                                        'CCCCCCCC[Sn](Cl)(Cl)CCCCCCCC', False, '', ''),
    'MOL_JTGNPNLBCGBCMP-UHFFFAOYSA-N': ('molecular organotin compound given as radical fragments',
                                        'CCCCCCCC[Sn](CCCCCCCC)(CCCCCCCC)CCCCCCCC', False, '', ''),
    'MOL_CSHAOLBENQQIBR-UHFFFAOYSA-H': ('hexafluoroantimonate given as six F- and Sb5+',
                                        f'{C4MIM}.F[Sb-](F)(F)(F)(F)F', True, C4MIM, 'F[Sb-](F)(F)(F)(F)F'),
    'MOL_PZMFVKDFXFYHCI-UHFFFAOYSA-J': ('tetrabromocobaltate given as four Br- and Co2+',
                                        f'{C4MIM}.{C4MIM}.Br[Co-2](Br)(Br)Br', True, C4MIM, 'Br[Co-2](Br)(Br)Br'),
    'MOL_CDWUIWLQQDTHRA-UHFFFAOYSA-N': ('cation charge placed on the other imidazolium nitrogen', None, True, C3MIM, None),
    'MOL_IPYQZLGLLJNFAT-UHFFFAOYSA-M': ('cation charge placed on the other imidazolium nitrogen', None, True, C3MMIM, None),
    'MOL_PXYKADKIWNXJNF-UHFFFAOYSA-O': ('cation written as a protonated zwitterion', None, True, HSO3C4MIM, None),
}
OLD_CATION = {'MOL_CDWUIWLQQDTHRA-UHFFFAOYSA-N': 'CCC[n+]1ccn(C)c1', 'MOL_IPYQZLGLLJNFAT-UHFFFAOYSA-M': 'CCC[n+]1ccn(C)c1C',
              'MOL_PXYKADKIWNXJNF-UHFFFAOYSA-O': 'C[N+]1=C[NH+](CCCCS(=O)(=O)[O-])C=C1'}


def read(name):
    with open(SRC / name if isinstance(name, str) else name, encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    return rows, list(rows[0])


def write(name, rows, columns):
    with open(OUT / name, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, columns, lineterminator='\n')
        w.writeheader()
        w.writerows(rows)


def renamed(rows, columns, mapping):
    return [{mapping.get(k, k): v for k, v in r.items()} for r in rows], [mapping.get(c, c) for c in columns]


# activity coefficients
t1, c1 = read('task1.csv')
for r in t1:
    r['Source'] = SOURCE[r['Source']]
    r['Original_Source_File'] = r['Original_Source_File'].replace('\\', '/')
write('activity_coefficients.csv', *renamed(t1, c1, COLUMNS))

# isomer pairs
t2, c2 = read('task2.csv')
for r in t2:
    r['Isomer_Relation'] = RELATION[r['Isomer_Relation']]
write('isomer_pairs.csv', t2, c2)

# molecules and solvents with corrected structures
mol, cm = read('molecules.csv')
solv, cs = read(HERE / 'inputs' / 'solvent_table.csv')
log = []
for r in mol:
    if r['Mol_ID'] not in FIX:
        continue
    problem, smiles, *_ = FIX[r['Mol_ID']]
    if smiles is None:
        smiles = r['SMILES'].replace(OLD_CATION[r['Mol_ID']], FIX[r['Mol_ID']][3])
    m = Chem.MolFromSmiles(smiles)
    assert m and Chem.GetFormalCharge(m) == 0 and not any(a.GetNumRadicalElectrons() for a in m.GetAtoms()), smiles
    key = Chem.MolToInchiKey(m)
    log.append({'Mol_ID': r['Mol_ID'], 'Problem': problem, 'SMILES_As_Analysed': r['SMILES'], 'SMILES_Corrected': smiles,
                'InChIKey_As_Analysed': r['InChIKey'], 'InChIKey_Corrected': key})
    r['SMILES'], r['InChIKey'] = smiles, key
corrected = {x['Mol_ID']: x['SMILES_Corrected'] for x in log}
for r in solv:
    k = r['Solvent_Mol_ID']
    if k in FIX:
        _, _, ionic, cation, anion = FIX[k]
        r['smiles'], r['is_IL'], r['cation'] = corrected[k], str(ionic), cation
        if anion is not None:
            r['anion'] = anion
write('molecules.csv', mol, cm)
write('solvents.csv', *renamed(solv, cs, SOLVENT_COLUMNS))
write('structure_corrections.csv', log, list(log[0]))

# folds and replicates
shutil.copyfile(SRC / 'split_manifest.json', OUT / 'folds_new_isomer_test.json')
shutil.copyfile(HERE / 'inputs' / 'folds_new_solvent_test.json', OUT / 'folds_new_solvent_test.json')
shutil.copyfile(HERE / 'inputs' / 'folds_new_solvent_test_summary.csv', OUT / 'folds_new_solvent_test_summary.csv')
rep, cr = read(MERGE / 'cross_study_replicates.csv')
write('cross_study_replicates.csv', *renamed(rep, cr, REPLICATE_COLUMNS))

# checksums
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
counts = {'activity_coefficients': len(t1), 'isomer_pairs': len(t2), 'molecules': len(mol), 'solvents': len(solv)}
files = {p.name: sha(p) for p in sorted(OUT.iterdir()) if p.suffix in ('.csv', '.json') and p.name != 'checksums.json'}
(OUT / 'checksums.json').write_text(json.dumps({'counts': counts, 'sha256': files}, indent=2) + '\n')
print(json.dumps(counts))
