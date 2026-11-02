"""Paths and fixed-family locks for the two-workbook curation layout.

This module never opens or changes a source workbook and never chooses folds.
The original assignment and complete train/validation/test family sets are
scientific inputs; migration changes where they live, not their contents.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

CURATION_FOLDER = '08_Dataset_Curation'
WORKBOOKS = ('01_MERGED_DEDUPLICATED', '02_EXPERIMENTS_PAIRS_ISOMERS')
FIXED_OUTER_SOURCE = '_pipeline/assets/curation/family_fold_manifest.json'
LOCK_NAME = 'curation_asset_lock.json'


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               allow_nan=False) + '\n', encoding='utf8')


def family_role_signature(manifest):
    """Compare memberships, independent of data counts and path metadata."""
    return {
        'outer_assignment': manifest['outer_assignment'],
        'rounds': {
            str(round_id): {
                role: sorted(values[role + '_families'])
                for role in ('train', 'validation', 'test')
            }
            for round_id, values in manifest['rounds'].items()
        },
    }


def validate_fixed_family_sets(outer, complete):
    assignment = outer['assignment']
    if complete['outer_assignment'] != assignment:
        raise ValueError('Complete fixed split differs from original family assignment')
    if len(assignment) != 84:
        raise ValueError('Expected the previously locked 84-family assignment')
    if set(assignment.values()) != {f'Fold_{i}' for i in range(1, 6)}:
        raise ValueError('Fixed family assignment must contain the five original folds')
    if set(complete['rounds']) != {str(i) for i in range(1, 6)}:
        raise ValueError('Expected all five complete fixed rounds')
    for round_id, values in complete['rounds'].items():
        groups = [set(values[role + '_families'])
                  for role in ('train', 'validation', 'test')]
        if any(groups[i] & groups[j] for i in range(3) for j in range(i + 1, 3)):
            raise ValueError('Fixed split contains family leakage')
        if set.union(*groups) != set(assignment):
            raise ValueError('Fixed split does not partition all original families')
        if groups[2] != {family for family, fold in assignment.items()
                         if fold == 'Fold_' + str(round_id)}:
            raise ValueError('Fixed test role differs from original outer fold')
    return family_role_signature(complete)


def load_locked_curation_assets(asset_dir=None):
    asset_dir = Path(asset_dir or Path(__file__).parent / 'assets' / 'curation')
    lock = load_json(asset_dir / LOCK_NAME)
    for name, expected in lock['files'].items():
        path = (asset_dir / name).resolve()
        if not path.is_relative_to(asset_dir.resolve()):
            raise ValueError('Asset lock escapes central curation directory')
        if not path.is_file() or sha256(path) != expected:
            raise ValueError('Locked curation input missing or changed: ' + name)
    outer = load_json(asset_dir / 'family_fold_manifest.json')
    complete = load_json(asset_dir / 'fixed_model_split_v7.json')
    signature = validate_fixed_family_sets(outer, complete)
    signature_sha = hashlib.sha256(json.dumps(signature, sort_keys=True,
        separators=(',', ':'), ensure_ascii=False).encode('utf8')).hexdigest()
    if signature_sha != lock['fixed_family_role_signature_sha256']:
        raise ValueError('Family assignment or complete round roles changed')
    return {'directory': asset_dir, 'lock': lock, 'outer': outer,
            'complete': complete, 'signature': signature}


def curation_output(project_root, workbook):
    if workbook not in WORKBOOKS:
        raise ValueError('Only the two authorized curation workbooks are outputs')
    return Path(project_root) / CURATION_FOLDER / (workbook + '.xlsx')


def migrate_split_metadata(manifest):
    """Only source-location metadata changes; every family role is immutable."""
    before = family_role_signature(manifest)
    result = dict(manifest)
    result['outer_assignment_source'] = FIXED_OUTER_SOURCE
    result['outer_assignment_changed'] = False
    result['complete_family_split_changed'] = False
    if family_role_signature(result) != before:
        raise AssertionError('Metadata migration changed family memberships')
    return result
