"""Rebuild graph-indexed training views from the package's reviewed experiments.

This is a staging-only operation. It does not redo experimental canonicalization
and never reassigns a family; the original experiment bank and decisions remain
the authority for the separately generated two curation workbooks.
"""
from pathlib import Path
import argparse
import json
import shutil
import dataset_helpers_v2 as helpers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=Path, required=True,
                        help='New output directory, separate from this package')
    args = parser.parse_args()
    package = Path(__file__).resolve().parents[2]
    stage = args.stage.resolve()
    if stage == package or package in stage.parents:
        raise ValueError('Rebuild into a separate new directory')
    if stage.exists() and any(stage.iterdir()):
        raise ValueError('Rebuild stage must be empty')
    freeze = json.loads((package / 'data/freeze.json').read_text(encoding='utf8'))
    for relative, expected in freeze['external_files'].items():
        if helpers.sha(package / relative) != expected:
            raise ValueError('Reviewed evidence changed: ' + relative)
    records = helpers.read_csv(package / 'curation/reviewed_model_task1.csv')
    pairs = helpers.read_csv(package / 'curation/reviewed_model_task2.csv')
    (stage / 'data').mkdir(parents=True)
    stats, counts = helpers.graphs_and_split(records, pairs, stage,
        package / 'curation', package / 'curation/fixed_model_split_v7.json')
    helpers.write_csv(stage / 'data/task1.csv', records)
    helpers.write_csv(stage / 'data/task2.csv', pairs)
    current = json.loads((package / 'data/split_manifest.json').read_text(encoding='utf8'))
    rebuilt = json.loads((stage / 'data/split_manifest.json').read_text(encoding='utf8'))
    # Add the original published location/scope metadata and active flags;
    # every role membership and newly computed count must match first.
    if rebuilt['rounds'] != current['rounds'] or rebuilt['outer_assignment'] != current['outer_assignment']:
        raise ValueError('Rebuilt fixed split differs from the published one')
    shutil.copy2(package / 'data/split_manifest.json', stage / 'data/split_manifest.json')
    checks = {name: helpers.sha(stage / 'data' / name) == digest
              for name, digest in freeze['files'].items()}
    if not all(checks.values()):
        raise ValueError('Rebuilt model views differ from frozen files: ' + str(checks))
    print(json.dumps({'status': 'PASS', 'files_exact': checks,
                      'graph_statistics': stats, 'fold_counts': counts},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
