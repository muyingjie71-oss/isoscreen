"""Build the training input folders from the public dataset.

Usage: python training/prepare_data.py

Writes training/data/new_isomer_test/ and training/data/new_solvent_test/ in the format read by the
D-MPNN and GH-GNN code: task1.csv (activity coefficients), task2.csv (isomer pairs), molecules.csv,
molecule_graphs.npz, split_manifest.json (folds) and dataset.json (expected counts). Labels and column
names are those of the training code. The molecules are those used to train the models in the paper
(inputs/molecules_as_trained.csv), not the corrected structures of the public dataset.
"""
import csv
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / 'data'
INPUTS = HERE / 'inputs'

SOURCE = {'NIST ThermoML': 'ThermoML', 'NIST ILThermo': 'ILThermo', 'Brouwer et al. 2021': 'Brouwer',
          'Tan et al. 2022': 'Tan', 'Li et al. 2022': 'DeepFM', 'Sherman et al. 1996': 'Sherman',
          'Individual papers': '单独收集论文'}
RELATION = {'skeletal (chain branching)': '骨架/链异构', 'positional': '位置异构', 'ring-chain': '环-链异构',
            'metameric': '连接/分配异构', 'functional group': '官能团异构', 'skeletal (ring system)': '骨架/环系异构',
            'unsaturation type': '不饱和类型异构'}
COLUMNS = {'Reported_Pressure_kPa': 'Raw_压力_kPa', 'Reported_Phase': 'Raw_相态', 'Reported_Standard_State': 'Raw_标准态',
           'Reported_Method': 'Raw_实验方法', 'Reported_Other_Conditions': 'Raw_其他固定条件'}

def read(path):
    with open(path, encoding='utf-8-sig', newline='') as f:
        rows = list(csv.reader(f))
    return rows[0], rows[1:]


def write(path, header, rows):
    with open(path, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(header)
        w.writerows(rows)


def build(work):
    work.mkdir(parents=True, exist_ok=True)
    h, rows = read(DATA / 'activity_coefficients.csv')
    s = h.index('Source')
    for r in rows:
        r[s] = SOURCE[r[s]]
    h = [COLUMNS.get(c, c) for c in h]
    write(work / 'task1.csv', h, rows)

    h, rows = read(DATA / 'isomer_pairs.csv')
    k = h.index('Isomer_Relation')
    for r in rows:
        r[k] = RELATION[r[k]]
    write(work / 'task2.csv', h, rows)

    shutil.copyfile(INPUTS / 'molecules_as_trained.csv', work / 'molecules.csv')
    shutil.copyfile(INPUTS / 'molecule_graphs.npz', work / 'molecule_graphs.npz')


def info(folder, cohort):
    doc = {'cohort': cohort, 'requires_matched_condition_signature': True,
           'expected_counts': {'task1': 68763, 'task2': 31541, 'matched_groups': 14075}}
    (folder / 'dataset.json').write_text(json.dumps(doc, indent=2) + '\n')


def main():
    out = HERE / 'data'
    base = out / 'new_isomer_test'
    shutil.rmtree(base, ignore_errors=True)
    build(base)
    shutil.copyfile(DATA / 'folds_new_isomer_test.json', base / 'split_manifest.json')
    info(base, 'new_isomer_test')

    other = out / 'new_solvent_test'
    shutil.rmtree(other, ignore_errors=True)
    shutil.copytree(base, other)
    shutil.copyfile(DATA / 'folds_new_solvent_test.json', other / 'split_manifest.json')
    info(other, 'new_solvent_test')
    print('written:', base, other)


if __name__ == '__main__':
    main()
