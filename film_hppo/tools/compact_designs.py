"""Compact existing design JSON files, with an exact backup before replacement."""

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from film_rl.design import compact_design


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path, help='One design JSON or a designs directory')
    args = parser.parse_args()
    source = (args.input if args.input.is_absolute() else PROJECT / args.input).resolve()
    if not source.is_relative_to(PROJECT):
        parser.error('Input must be inside film_hppo')
    if not source.exists():
        parser.error(f'Input does not exist: {source}')
    files = sorted(source.rglob('*.json')) if source.is_dir() else [source]
    backup = PROJECT / 'outputs/design_format_backups' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    changes = []
    for path in files:
        path = path.resolve()
        if not path.is_relative_to(PROJECT):
            raise ValueError(f'File escapes project: {path}')
        original = path.read_bytes()
        document = json.loads(original.decode('utf-8-sig'))
        if not isinstance(document, dict) or not {'layers', 'score', 'target_errors'} <= document.keys():
            continue
        if document.get('design_format') == 'compact_v1':
            continue
        compact = compact_design(document)
        assert compact['layers'] == document['layers']
        assert compact['score'] == document['score']
        raw = (json.dumps(compact, ensure_ascii=False, indent=2) + '\n').encode('utf-8')
        saved = backup / path.relative_to(PROJECT)
        saved.parent.mkdir(parents=True, exist_ok=True)
        saved.write_bytes(original)
        # The live learner can replace best designs. Never overwrite a newer snapshot.
        if path.read_bytes() != original:
            print(f'Skipped changing file: {path}')
            continue
        temporary = path.with_name(path.name + '.compact.tmp')
        temporary.write_bytes(raw)
        if path.read_bytes() != original:
            temporary.unlink()
            print(f'Skipped changing file: {path}')
            continue
        temporary.replace(path)
        changes.append({'file': str(path), 'backup': str(saved), 'old_bytes': len(original),
                        'new_bytes': len(raw), 'score': document['score'],
                        'original_sha256': hashlib.sha256(original).hexdigest()})
    if changes:
        (backup / 'manifest.json').write_text(json.dumps(changes, indent=2), encoding='utf-8')
    print(json.dumps({'converted': len(changes), 'old_bytes': sum(x['old_bytes'] for x in changes),
                      'new_bytes': sum(x['new_bytes'] for x in changes),
                      'backup_directory': str(backup) if changes else None}, indent=2))


if __name__ == '__main__':
    main()
