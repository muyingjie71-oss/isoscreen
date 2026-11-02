"""Build a Colab snapshot from the current two-workbook scientific bank.

Canonicalization and experimental screening belong to curation_merge. This
module only indexes the reviewed molecular structures, retains the existing
family roles, freezes the resulting files, and packages unchanged training code.
"""
from __future__ import annotations

from collections import Counter
import csv
import gzip
import importlib
import json
import math
from pathlib import Path
import shutil
import sys
from zipfile import ZipFile, ZipInfo, ZIP_DEFLATED

from .curation_contract import (FIXED_OUTER_SOURCE, family_role_signature,
    load_json, load_locked_curation_assets, save_json, sha256)

TRAINING_CODE = ('data.py', 'graphs.py', 'losses.py', 'metrics.py', 'model.py',
                 'run_5fold.py', 'train.py', 'trainer.py', 'test_core.py',
                 'prepare_training_data.py')

VARIANT_DIRECTORIES = ('A_2D', 'B1_GLOBAL', 'B2_SURFACE', 'B3_SPATIAL')


def _training_files(source):
    source = Path(source)
    if (source / 'shared').is_dir():
        return tuple(p.relative_to(source).as_posix()
                     for p in sorted((source / 'shared').glob('*.py')))
    return TRAINING_CODE


def _read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def _load_helpers(directory):
    directory = str(Path(directory).resolve())
    sys.path.insert(0, directory)
    for name in ('dataset_helpers_v2', 'canonicalize_v2', 'graphs'):
        prior = sys.modules.get(name)
        if prior and Path(prior.__file__).resolve().parent != Path(directory):
            del sys.modules[name]
    return importlib.import_module('dataset_helpers_v2')


def validate_reviewed_views(records, pairs, assignment):
    from rdkit import Chem
    ids = {record['canonical_record_id']: record for record in records}
    if len(ids) != len(records) or not ids:
        raise ValueError('Reviewed Task1 IDs must be nonempty and unique')
    missing = sorted({r['Family_ID'] for r in records} - set(assignment))
    if missing:
        raise ValueError('Model family absent from fixed assignment: ' + str(missing))
    identities = {}
    for record in records:
        if record.get('Model_Eligible') != '是':
            raise ValueError('Unqualified record in model Task1: ' + record['canonical_record_id'])
        gamma, temperature = float(record['gamma_inf']), float(record['T_K'])
        if not (math.isfinite(gamma) and gamma > 0 and math.isfinite(temperature)
                and temperature > 0):
            raise ValueError('Nonpositive/nonfinite experimental value in model view')
        if abs(math.log(gamma) - float(record['ln_gamma_inf'])) > 1e-12:
            raise ValueError('ln gamma differs from original experimental gamma')
        for field in ('Solute_SMILES', 'Solvent_SMILES', 'Solvent_Mol_ID',
                      'Effective_Conditions'):
            if not record.get(field):
                raise ValueError('Required confirmed model field missing: ' + field)
        if record.get('Strict_Identity_Eligible') != '是':
            raise ValueError('Current dual-InChI qualification missing from model view')
        for role in ('Solute', 'Solvent'):
            inchi = record.get(role + '_InChI', '')
            key = record.get(role + '_Identity', '')
            smiles = record[role + '_SMILES']
            if not inchi.startswith('InChI=1') or not record.get(role + '_Identity_Evidence'):
                raise ValueError('Complete confirmed InChI or identity evidence missing: ' + role)
            token = (inchi, key, smiles)
            if token not in identities:
                molecule = Chem.MolFromSmiles(smiles)
                if molecule is None or Chem.InchiToInchiKey(inchi) != key or Chem.MolToInchiKey(molecule) != key:
                    raise ValueError('Reviewed complete InChI / Key / graph disagree: ' + role + ' ' + key)
                identities[token] = True
    pair_ids = set()
    group_families = {}
    for pair in pairs:
        pid = pair['Pair_ID']
        if pid in pair_ids:
            raise ValueError('Duplicate matched Pair_ID')
        pair_ids.add(pid)
        a, b = (ids[pair['canonical_record_id_' + end]] for end in ('A', 'B'))
        for field in ('Family_ID', 'Study_ID', 'Solvent_Mol_ID'):
            if a[field] != b[field] or pair[field] != a[field]:
                raise ValueError('Pair endpoint identity mismatch: ' + pid + ' ' + field)
        if a['Member_ID'] == b['Member_ID']:
            raise ValueError('Matched pair uses identical constitutional member')
        if abs(float(a['T_K']) - float(b['T_K'])) > 1e-9:
            raise ValueError('Matched pair temperatures differ')
        expected = float(a['ln_gamma_inf']) - float(b['ln_gamma_inf'])
        if abs(expected - float(pair['D_AB'])) > 1e-12:
            raise ValueError('Matched target differs from two experimental endpoints')
        gid = pair['MatchedGroup_ID']
        if gid in group_families and group_families[gid] != a['Family_ID']:
            raise ValueError('Matched group crosses constitutional families')
        group_families[gid] = a['Family_ID']
    return {'task1': len(records), 'task2_pairs': len(pairs),
            'matched_groups': len(group_families)}


def _copy_training_shell(project, destination):
    source = Path(project) / '10_ML_Colab'
    if (source / 'shared').is_dir():
        # The data builder copies code and protocols, never old descriptor
        # coverage, common cohorts, checkpoints or fitted preprocessing.
        for name in ('shared', 'descriptor_calculation', *VARIANT_DIRECTORIES):
            if not (source / name).is_dir():
                raise ValueError('Incomplete four-input training shell: ' + name)
            shutil.copytree(source / name, destination / name,
                            ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        if (source / 'README.md').is_file():
            shutil.copy2(source / 'README.md', destination / 'README.md')
        return
    for name in TRAINING_CODE:
        shutil.copy2(source / name, destination / name)
    for source_file in source.iterdir():
        if source_file.is_file() and source_file.suffix in ('.txt', '.yaml', '.yml'):
            shutil.copy2(source_file, destination / source_file.name)
    for directory in ('configs',):
        if (source / directory).is_dir():
            shutil.copytree(source / directory, destination / directory)
    notebook = source / 'Surrogate_Modelling_Colab_v2.ipynb'
    if notebook.is_file():
        shutil.copy2(notebook, destination / notebook.name)


def _notebook_scope(destination, report):
    path = destination / 'Surrogate_Modelling_Colab_v2.ipynb'
    if not path.is_file():
        return
    notebook = load_json(path)
    for cell in notebook.get('cells', []):
        cell['source'] = [line.replace('Layer_A_v2_Strict_InChI_v1',
            'Layer_A_v3_Current_Source03') for line in cell.get('source', [])]
    for cell in notebook.get('cells', []):
        if cell.get('cell_type') == 'markdown':
            cell['source'] = [
                '# Colab 训练：七来源当前最终数据\n',
                f'当前训练输入：{report["task1"]:,} 个实测单点、'
                f'{report["task2_pairs"]:,} 对异构体实验。\n',
                '数据范围与第八文件夹的第二张表一致。固定的 84 个 family '
                '及各轮 train、validation、test 家族名单保持不变。\n',
                '使用 Layer_A_v3_Current_Source03 结果目录；当前数据快照有独立指纹。\n',
            ]
            break
    save_json(path, notebook)


def build_training_snapshot(project_root, science_dir, output_dir, asset_dir=None,
                            zip_output=False):
    project, science, output = map(lambda p: Path(p).resolve(),
                                 (project_root, science_dir, output_dir))
    if output == project or project in output.parents or science == output:
        raise ValueError('Build the Colab snapshot in a separate staging directory')
    if output.exists() and any(output.iterdir()):
        raise ValueError('Colab staging output must be empty')
    locked = load_locked_curation_assets(asset_dir)
    assets = locked['directory']
    scientific_report_path = science / 'curation/data_preparation_report.json'
    scientific_report_sha = sha256(scientific_report_path)
    report = load_json(scientific_report_path)
    ready = load_json(science / 'SCIENCE_READY.json')
    if report.get('status') != 'PASS' or ready.get('status') != 'PASS':
        raise ValueError('Scientific curation has not passed its current-scope audit')
    for name, expected in report['csv_sha256'].items():
        path = (science / name.replace('\\', '/')).resolve()
        if not path.is_relative_to(science) or sha256(path) != expected:
            raise ValueError('Scientific bank hash changed before training build: ' + name)
    records = _read_csv(science / 'data/task1.csv')
    pairs = _read_csv(science / 'data/task2.csv')
    raw_path = science / 'curation/source_observations.csv.gz'
    with gzip.open(raw_path, 'rt', encoding='utf-8-sig', newline='') as stream:
        raw_keys = [r['canonical_record_id'] for r in csv.DictReader(stream)]
    raw_ids = set(raw_keys)
    if len(raw_ids) != locked['lock']['expected_input_rows'] or len(raw_keys) != len(raw_ids):
        raise ValueError('Current source03 raw bank differs from the locked 163074-row scope')
    if any(r.get('selected_input_record_id') not in raw_ids for r in records):
        raise ValueError('Model representative is outside the current seven-source bank')
    counts = validate_reviewed_views(records, pairs, locked['outer']['assignment'])
    report.update(counts)
    report['input_scope'] = 'Seven current final 03_ISOMERS; exactly 163074 selected source records'
    report['outer_family_assignment_changed'] = False
    report['complete_family_split_changed'] = False
    output.mkdir(parents=True, exist_ok=True)
    (output / 'data').mkdir()
    (output / 'curation/tools').mkdir(parents=True)
    _copy_training_shell(project, output)
    # Only current scientific outputs are active evidence; historical banks are
    # saved in the pre-migration backup rather than reintroduced into this bank.
    for source_file in sorted((science / 'curation').iterdir()):
        if source_file.is_file() and source_file.suffix in ('.csv', '.json', '.gz'):
            shutil.copy2(source_file, output / 'curation' / source_file.name)
    shutil.copy2(scientific_report_path, output / 'curation/scientific_selection_report.json')
    shutil.copy2(science / 'data/task1.csv', output / 'curation/reviewed_model_task1.csv')
    shutil.copy2(science / 'data/task2.csv', output / 'curation/reviewed_model_task2.csv')
    for name in ('family_fold_manifest.json', 'fixed_model_split_v7.json'):
        shutil.copy2(assets / name, output / 'curation' / name)
    tools = output / 'curation/tools'
    for source_file in sorted((assets / 'common_modules').glob('*.py')):
        shutil.copy2(source_file, tools / source_file.name)
    shutil.copy2(Path(__file__).parent / 'curation_rebuild_training.py',
                 tools / 'rebuild_training_views.py')
    helpers = _load_helpers(tools)
    graph_stats, fold_counts = helpers.graphs_and_split(records, pairs, output,
        output / 'curation', output / 'curation/fixed_model_split_v7.json')
    # The helper reads the immutable old file but writes only current counts
    # and current metadata; compare every original family membership again.
    split = load_json(output / 'data/split_manifest.json')
    if family_role_signature(split) != locked['signature']:
        raise ValueError('Training snapshot changed fixed family memberships')
    split['outer_assignment_source'] = FIXED_OUTER_SOURCE
    split['input_scope'] = report['input_scope']
    split['family_active_flags'] = {
        family: any(r['Family_ID'] == family for r in records)
        for family in sorted(locked['outer']['assignment'])}
    save_json(output / 'data/split_manifest.json', split)
    helpers.write_csv(output / 'data/task1.csv', records)
    helpers.write_csv(output / 'data/task2.csv', pairs)
    report['graph_statistics'] = graph_stats
    report['outer_fold_counts'] = fold_counts
    report['families_task1'] = len({r['Family_ID'] for r in records})
    report['families_task2'] = len({r['Family_ID'] for r in pairs})
    report['reviewed_model_inputs_are_current_scope'] = True
    save_json(output / 'curation/data_preparation_report.json', report)
    _notebook_scope(output, report)
    (output / 'README.md').write_text(
        '# Colab 训练包：七来源当前最终数据\n\n'
        '打开 `Surrogate_Modelling_Colab_v2.ipynb`，按顺序运行并上传同名 ZIP。\n\n'
        f'当前训练输入：{counts["task1"]:,} 个实测单点、{counts["task2_pairs"]:,} '
        f'个 matched pair、{counts["matched_groups"]:,} 个 matched group。'
        '范围与第八文件夹第二张表中的模型视图一致，来源为七份当前最终表三。\n\n'
        '原 γ∞、温度、完整来源历史和排除原因保存在 `curation`。'
        '两个配对端点都来自同一研究、溶剂、温度、家族与相容实验条件。'
        '仅经过完整 InChI 结构确认及实验质量筛选的记录进入 `data`。\n\n'
        '原 84 个 family 的外层 fold 与各轮 train、validation、test 家族名单不变，'
        '只重算当前数据量和有数据家族标记。所有原训练代码保持原样；'
        '本次未运行正式训练。\n\n'
        '`prepare_training_data.py --verify` 校验当前冻结哈希及分子图。'
        '`curation/tools/rebuild_training_views.py --stage <新目录>` 可离线从'
        '本包的已审核模型实验视图重建图缓存、固定划分及训练 CSV；'
        '完整合并、去重和筛选由第八文件夹的同名生成脚本及中央审核资产重建。\n',
        encoding='utf8')
    if (project / '10_ML_Colab/shared').is_dir():
        current_readme = (project / '10_ML_Colab/README.md').read_text(encoding='utf8')
        (output / 'README.md').write_text(current_readme + '\n\n'
            f'本次重建完整实验视图：{counts["task1"]:,} 个单点、'
            f'{counts["task2_pairs"]:,} 对配对、{counts["matched_groups"]:,} 个配对组。'
            '第二层覆盖及共同样本须在此快照中重新校验，不复制旧训练结果。\n',
            encoding='utf8')
    # Freeze only files actually shipped. Tool hashes are checked by the
    # existing unchanged prepare_training_data verifier.
    external = {p.relative_to(output).as_posix(): sha256(p)
        for p in sorted((output / 'curation').iterdir()) if p.is_file()}
    freeze = {
        'version': '3.0-current-source03', 'created_date': '2026-10-09',
        'requires_matched_condition_signature': True,
        'requires_strict_dual_inchi': True,
        'files': {p.name: sha256(p) for p in sorted((output / 'data').iterdir())
                  if p.is_file() and p.name != 'freeze.json'},
        'external_files': external,
        'input_files': {
            'scientific_selection_report.json': scientific_report_sha,
            'source_observations.csv.gz': sha256(science / 'curation/source_observations.csv.gz'),
            'family_fold_manifest.json': sha256(assets / 'family_fold_manifest.json'),
            'fixed_model_split_v7.json': sha256(assets / 'fixed_model_split_v7.json'),
            'reviewed_model_task1.csv': sha256(science / 'data/task1.csv'),
            'reviewed_model_task2.csv': sha256(science / 'data/task2.csv')},
        'rebuild_tool_files': {p.relative_to(tools).as_posix(): sha256(p)
            for p in sorted(tools.rglob('*')) if p.is_file() and '__pycache__' not in p.parts},
        'report': report,
    }
    if sha256(scientific_report_path) != scientific_report_sha:
        raise ValueError('Scientific selection changed during training snapshot creation')
    for name, expected in report['csv_sha256'].items():
        if sha256(science / name.replace('\\', '/')) != expected:
            raise ValueError('Scientific bank changed during training snapshot creation: ' + name)
    save_json(output / 'data/freeze.json', freeze)
    result = {'status': 'BUILT', **counts, 'molecules': graph_stats,
        'fixed_family_roles_unchanged': True, 'training_code_unchanged': {
            name: sha256(output / name) == sha256(project / '10_ML_Colab' / name)
            for name in _training_files(project / '10_ML_Colab')}, 'output': str(output),
        'training_configuration_unchanged': {
            p.relative_to(output).as_posix(): sha256(p) == sha256(
                project / '10_ML_Colab' / p.relative_to(output))
            for p in sorted(output.rglob('*'))
            if p.is_file() and (p.suffix in ('.yaml', '.yml') or p.name == 'requirements.txt')},
        'freeze_sha256': sha256(output / 'data/freeze.json')}
    save_json(output / 'upload_package_verification.json', result)
    if (output / 'shared').is_dir():
        # Four-input notebooks use this final marker to recover a interrupted
        # ZIP extraction. It belongs to the rebuilt scientific snapshot,
        # rather than an older copied package verification report.
        save_json(output / 'verification/package_verification.json', {
            'status': 'BUILT_FROM_REVIEWED_EXPERIMENTS', **counts,
            'freeze_sha256': result['freeze_sha256'],
            'fixed_family_roles_unchanged': True,
            'training_code_unchanged': result['training_code_unchanged'],
            'formal_training_performed': False,
            'physical_calculation_performed_by_snapshot_builder': False,
        })
    if zip_output:
        result['zip'] = package_zip(output)
    elif (output / 'shared' / 'package_integrity.py').is_file():
        _package_integrity(output).seal_package(output)
    return result


def _package_integrity(directory):
    spec = importlib.util.spec_from_file_location('_portable_package_integrity',
        Path(directory) / 'shared' / 'package_integrity.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def package_zip(directory):
    """Package exactly the staged payload with deterministic ZIP metadata."""
    directory = Path(directory).resolve()
    name = ('Surrogate_Modelling_A_B1_B2_B3_Colab_v1.zip'
            if (directory / 'shared').is_dir() else 'Surrogate_Modelling_Colab_v2.zip')
    target = directory / name
    if (directory / 'shared' / 'package_integrity.py').is_file():
        integrity = _package_integrity(directory)
        integrity.seal_package(directory)
        payloads = integrity.payload_paths(directory, include_marker=True)
    else:
        payloads = [p for p in sorted(directory.rglob('*')) if p.is_file()
                    and '__pycache__' not in p.parts and p.suffix not in ('.pyc', '.zip')]
    # Colab uses this file to detect a complete extraction.  Keep it last so
    # an interrupted extraction cannot leave a success marker ahead of files.
    completion_marker = directory / 'verification' / 'package_verification.json'
    if completion_marker in payloads:
        payloads.remove(completion_marker)
        payloads.append(completion_marker)
    with ZipFile(target, 'w', ZIP_DEFLATED, compresslevel=4) as archive:
        for path in payloads:
            info = ZipInfo(path.relative_to(directory).as_posix(), (2026, 10, 9, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            with archive.open(info, 'w') as output, path.open('rb') as source:
                shutil.copyfileobj(source, output, 1024 * 1024)
    with ZipFile(target) as archive:
        if archive.testzip() is not None:
            raise ValueError('Training ZIP integrity failure')
        if archive.namelist() != [p.relative_to(directory).as_posix() for p in payloads]:
            raise ValueError('Training ZIP does not contain the exact staged payload')
    return {'path': str(target), 'sha256': sha256(target), 'files': len(payloads),
            'bytes': target.stat().st_size}
