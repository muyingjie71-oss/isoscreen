"""Build the training input folders from the public dataset.

Usage: python training/prepare_data.py

Writes training/data/new_isomer_test/ and training/data/new_solvent_test/ in the format read by
the D-MPNN and GH-GNN code (task1.csv, task2.csv, molecules.csv, split_manifest.json, freeze.json and
two auxiliary files). The molecules are those used to train the models in the paper
(inputs/molecules_as_trained.csv), not the corrected structures of the public dataset.

The script checks that the result equals the training data of the paper. The only difference is the
column Original_Source_File, which gives paths relative to the source archive.
"""
import csv
import hashlib
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

# Checksums of the training data used in the paper
EXPECTED = {
    'task1_without_source_path': '665c57cc3d12e9f3a4da80fcc28f65e6663f0ae1339ed996ee6b3f8c9867c4a8',
    'task2.csv': 'e64240098799593219340e89ea699a7606ed46ff5120a4956efdf074ed77f5cb',
    'molecules.csv': '1110abff6b529de8cdaca77a6f1287fcba830bcd7602ee9941afb0c356c36e07',
    'split_new_isomer_test': '12bb0c7840b406eaebe4aa3e9d81d256cd7b5e525fda47270f4359e40b11fc2f',
    'split_new_solvent_test': '80e8285993e473d162ae41886ffe9a12a858ddb9af916830b41d3701cb429b89',
    'graphs_layer_a_v2.npz': '125807e4e9e4f51c723be76068f2038553ed023c76f48f233487d806d6f200a3',
    'source_freeze_original.json': '47864710934ab8d7c3ba5853388710b6bd0a3c2bc35ed3a5d75e8a53bd8b7adc',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    with open(path, encoding='utf-8-sig', newline='') as f:
        rows = list(csv.reader(f))
    return rows[0], rows[1:]


def write(path, header, rows):
    with open(path, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(header)
        w.writerows(rows)


def check(name, value):
    if value != EXPECTED[name]:
        raise ValueError(f'{name} differs from the training data of the paper')


def build(work):
    work.mkdir(parents=True, exist_ok=True)
    h, rows = read(DATA / 'activity_coefficients.csv')
    s = h.index('Source')
    for r in rows:
        r[s] = SOURCE[r[s]]
    h = [COLUMNS.get(c, c) for c in h]
    write(work / 'task1.csv', h, rows)
    p = h.index('Original_Source_File')
    digest = hashlib.sha256()
    for r in rows:
        digest.update(('\x1f'.join(r[:p] + r[p + 1:]) + '\n').encode())
    check('task1_without_source_path', digest.hexdigest())

    h, rows = read(DATA / 'isomer_pairs.csv')
    k = h.index('Isomer_Relation')
    for r in rows:
        r[k] = RELATION[r[k]]
    write(work / 'task2.csv', h, rows)
    check('task2.csv', sha(work / 'task2.csv'))

    shutil.copyfile(INPUTS / 'molecules_as_trained.csv', work / 'molecules.csv')
    shutil.copyfile(INPUTS / 'molecule_graphs.npz', work / 'graphs_layer_a_v2.npz')
    shutil.copyfile(INPUTS / 'source_checksums.json', work / 'source_freeze_original.json')
    for name in ('molecules.csv', 'graphs_layer_a_v2.npz', 'source_freeze_original.json'):
        check(name, sha(work / name))


def freeze(folder, cohort):
    files = {n: sha(folder / n) for n in ('task1.csv', 'task2.csv', 'molecules.csv', 'split_manifest.json',
                                          'graphs_layer_a_v2.npz', 'source_freeze_original.json')}
    doc = {'version': '2.0', 'requires_matched_condition_signature': True, 'requires_strict_dual_inchi': True,
           'comparison_cohort': cohort, 'expected_counts': {'task1': 68763, 'task2': 31541, 'matched_groups': 14075},
           'files': files, 'source_original_freeze_sha256': files['source_freeze_original.json'],
           'report': {'split_manifest_sha256': files['split_manifest.json']}}
    (folder / 'freeze.json').write_text(json.dumps(doc, indent=2) + '\n')


def main():
    out = HERE / 'data'
    base = out / 'new_isomer_test'
    build(base)
    shutil.copyfile(DATA / 'folds_new_isomer_test.json', base / 'split_manifest.json')
    check('split_new_isomer_test', sha(base / 'split_manifest.json'))
    freeze(base, 'new_isomer_test')

    other = out / 'new_solvent_test'
    shutil.rmtree(other, ignore_errors=True)
    shutil.copytree(base, other)
    shutil.copyfile(DATA / 'folds_new_solvent_test.json', other / 'split_manifest.json')
    check('split_new_solvent_test', sha(other / 'split_manifest.json'))
    freeze(other, 'new_solvent_test')
    print('written:', base, other)


if __name__ == '__main__':
    main()
