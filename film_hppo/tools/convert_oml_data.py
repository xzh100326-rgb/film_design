"""Convert the archived OML-PPO data to refractiveindex.info YAML, offline.

Run from film_hppo: python tools/convert_oml_data.py
Original files and the pinned Git commit live in outputs/references/oml_ppo.
No training configurations are changed. Existing different YAML is never overwritten.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import sys
from urllib.parse import quote

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from film_rl.refractiveindex import read_refractiveindex

PIANO_MATERIALS = {
    'Al2O3 AE target 50nm real 72nm': 'Al2O3',
    'Fe2O3-Si-CdS-model': 'Fe2O3',
    'HfO2-online': 'HfO2',
    'Ni-Kyutae': 'Ni',
    'Si 1 DC-updoped silicon-aSi parameterized-300s-200w-sputtering': 'Si',
    'Si3N4-Plasmatherm-UV+Visible': 'Si3N4',
    'SiO2-AE-Target 500nm-5A_s-CGJi-20171021': 'SiO2',
    'Ti-AE-target 15nm': 'Ti',
    'TiN Lab18-1 10to2 noheating postAnneal': 'TiN',
    'ZnO-Lab18-01-1000s-34nm': 'ZnO',
    'ZnS-SJ20-Si sub-target 30nm real 19nm': 'ZnS',
    'ZnSe-30nm-SiOx_Si sub': 'ZnSe',
    'aSi-Plasmatherm-25w-extended': 'Si',
}


def read_table(path):
    ignored = []
    rows = []
    if path.suffix == '.csv':
        with path.open(encoding='utf-8-sig', newline='') as stream:
            reader = csv.DictReader(stream)
            if not {'wl', 'n', 'k'} <= set(reader.fieldnames or []):
                raise ValueError(f'{path}: expected wl,n,k header')
            for line, record in enumerate(reader, start=2):
                values = [record[key] for key in ('wl', 'n', 'k')]
                if all(value is not None and not value.strip() for value in values):
                    ignored.append(line)
                    continue
                rows.append([float(value) for value in values])
        unit = 'um'
    else:
        for line, text in enumerate(path.read_text(encoding='utf-8-sig').splitlines(), start=1):
            parts = text.split()
            if not parts or all(value == '--' for value in parts):
                ignored.append(line)
                continue
            if len(parts) not in (2, 3):
                raise ValueError(f'{path}:{line}: expected wavelength,n[,k]')
            rows.append([float(value) for value in parts])
        unit = 'nm'
    table = np.asarray(rows, dtype=np.float64)
    if table.ndim != 2 or table.shape[1] not in (2, 3) or not np.isfinite(table).all():
        raise ValueError(f'{path}: invalid table')
    if unit == 'nm':
        table[:, 0] /= 1000.0
    if (np.any(table[:, 0] <= 0) or np.any(table[:, 1] <= 0)
            or (table.shape[1] == 3 and np.any(table[:, 2] < 0))):
        raise ValueError(f'{path}: nonphysical wavelength/n/k')
    # No sorting, deduplication, smoothing or extrapolation is silently applied.
    if not np.all(np.diff(table[:, 0]) > 0):
        raise ValueError(f'{path}: wavelengths must be strictly increasing')
    return table, unit, ignored


def destination(relative):
    group = relative.parts[0] if len(relative.parts) > 1 else 'root'
    stem = relative.stem
    material = {'a-Si': 'Si', 'PECVD-Si': 'Si', 'Sputter-Si': 'Si'}.get(stem, stem)
    if group == 'piano_black':
        material = PIANO_MATERIALS[stem]
        slug = re.sub(r'[^A-Za-z0-9_-]+', '_', stem).strip('_')
        version = f'OML-PPO_piano_black_{slug}'
    elif group == '400-2000-acs':
        version = 'OML-PPO_400-2000-acs'
    elif group == 'root':
        version = 'OML-PPO' + (f'_{stem}' if material != stem else '')
    else:
        raise ValueError(f'Unrecognized data group: {group}')
    return material, group, Path('data/main') / material / 'nk' / (version + '.yml')


def write_new_or_identical(path, raw):
    if path.exists() and path.read_bytes() != raw:
        raise FileExistsError(f'Existing different file preserved: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(raw)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, default=PROJECT / 'outputs/references/oml_ppo')
    parser.add_argument('--install-database', type=Path,
                        help='Also add named versions to an existing database/ directory')
    args = parser.parse_args()
    package = args.package.resolve()
    manifest = json.loads((package / 'source_manifest.json').read_text(encoding='utf-8'))
    records_by_path = {entry['path']: entry for entry in manifest['files']}
    source = package / 'source/data'
    database = package / 'database'
    install = args.install_database.resolve() if args.install_database else None
    if install is not None and not (install / 'data').is_dir():
        raise FileNotFoundError(f'Expected existing database/data: {install}')
    records = []
    for path in sorted(source.rglob('*')):
        if path.suffix.lower() not in ('.csv', '.txt'):
            continue
        relative = path.relative_to(source)
        original = records_by_path['data/' + relative.as_posix()]
        if hashlib.sha256(path.read_bytes()).hexdigest() != original['sha256']:
            raise ValueError(f'Source hash mismatch: {path}')
        table, unit, skipped = read_table(path)
        material, group, target = destination(relative)
        kind = 'nk' if table.shape[1] == 3 else 'n'
        comment = (
            f'Converted from OML-PPO data/{relative.as_posix()}; commit {manifest["commit"]}. '
            f'Original wavelength unit: {unit}; YAML unit: micrometre. '
            'Original numeric samples retained, with no new resampling or extrapolation. '
            'Source CSV may contain constant extensions or sparse sampling already added upstream; '
            'its numeric coverage is not proof of measured physical validity. '
            'Repository training code uses quadratic interpolation; film_hppo currently uses linear interpolation. '
            'Not an official refractiveindex.info dataset; underlying experimental provenance is not inferred.'
        )
        if kind == 'n':
            comment += ' Source has no k column; k remains unspecified, not filled with zero.'
        references = (
            'H. Wang et al. Automated multi-layer optical design via deep reinforcement learning. '
            'Machine Learning: Science and Technology 2 (2021), 025013. '
            'https://doi.org/10.1088/2632-2153/abc327\nSource: ' + original['url']
        )
        lines = ['# Converted local dataset; wavelength in micrometres.',
                 'REFERENCES: ' + json.dumps(references),
                 'COMMENTS: ' + json.dumps(comment),
                 'DATA:', '  - type: tabulated ' + kind, '    data: |']
        lines.extend('      ' + ' '.join(format(float(value), '.17g') for value in row) for row in table)
        raw = ('\n'.join(lines) + '\n').encode('utf-8')
        output = database / target
        write_new_or_identical(output, raw)
        assume = kind == 'n'
        # Verify all original knots and all interval midpoints through the real project reader.
        wave_um = np.concatenate((table[:, 0], (table[:-1, 0] + table[1:, 0]) / 2))
        wave_nm = wave_um * 1000
        # Restrict edge roundoff to the exact physical coverage; do not alter source samples.
        wave_nm[0] = np.nextafter(wave_nm[0], np.inf)
        wave_nm[len(table)-1] = np.nextafter(wave_nm[len(table)-1], -np.inf)
        values, metadata = read_refractiveindex(output, wave_nm, assume_zero_k=assume)
        expected_n = np.interp(wave_nm / 1000, table[:, 0], table[:, 1])
        expected_k = np.interp(wave_nm / 1000, table[:, 0], table[:, 2]) if kind == 'nk' else np.zeros(len(wave_nm))
        np.testing.assert_allclose(values, expected_n + 1j*expected_k, rtol=1e-12, atol=1e-12)
        if kind == 'n':
            try:
                read_refractiveindex(output, wave_nm)
            except ValueError as error:
                if 'k is not provided' not in str(error):
                    raise
            else:
                raise AssertionError('Missing k must remain explicit')
        if install:
            write_new_or_identical(install / target, raw)
        records.append({
            'material': material, 'group': group, 'source_file': 'data/' + relative.as_posix(),
            'source_url': original['url'], 'source_sha256': original['sha256'],
            'yaml_file': target.as_posix(), 'yaml_sha256': metadata['sha256'],
            'data_type': 'tabulated ' + kind, 'points': len(table), 'source_wavelength_unit': unit,
            'wavelength_range_nm': [float(table[0, 0]*1000), float(table[-1, 0]*1000)],
            'skipped_placeholder_lines': skipped, 'missing_k': assume,
            'verified_samples': len(wave_nm), 'installed_file': str(install / target) if install else None,
        })
    report = {'repository': manifest['repository'], 'commit': manifest['commit'],
              'datasets': records, 'dataset_count': len(records), 'training_executed': False}
    (package / 'conversion_manifest.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    for group in ('root', '400-2000-acs', 'piano_black'):
        entries = []
        for item in records:
            if item['group'] != group:
                continue
            entry = {'name': item['material'], 'file': item['yaml_file']}
            entries.append(entry)
        snippet = {'_note': 'Reference list, not a complete training configuration. Select required materials only.',
                   'material_source': 'refractiveindex', 'refractiveindex_dir': str(database), 'materials': entries}
        if group == 'piano_black':
            snippet['_note'] += ' HfO2 TXT has no k: selection will fail unless you intentionally add assume_zero_k=true. No zero-k assumption is enabled here.'
        (package / f'config_materials_{group}.json').write_text(json.dumps(snippet, indent=2), encoding='utf-8')
    rows = ['# OML-PPO converted optical constants', '',
            f'Source: {manifest["repository"]}; pinned commit `{manifest["commit"]}`.', '',
            'Use `database` as refractiveindex_dir, or the named OML-PPO versions installed in your existing database.',
            'Existing experiment configurations and existing material versions were not changed.', '',
            '## Conversion and interpretation', '',
            '- CSV wavelengths are in um; piano_black TXT wavelengths are in nm and were divided by 1000.',
            '- No new resampling, smoothing, extrapolation or optical-constant fitting was applied.',
            '- Empty Al.csv rows and HfO2 TXT -- placeholders are recorded in conversion_manifest.json.',
            '- HfO2-online.txt provides only n. Its YAML intentionally omits k; assume_zero_k requires explicit consent in config.',
            '- Upstream CSV endpoints may be synthetic constant extensions. Numeric coverage does not establish measured validity.',
            '- Original OML-PPO utils.py uses scipy interp1d(kind="quadratic"); this project reads tables with linear interpolation.',
            '  Identical data therefore does not guarantee identical intermediate spectra between implementations.',
            '- PECVD-Si, Sputter-Si, and a-Si were mapped to Si folders with distinct dataset filenames.',
            '- The source notebook is archived, not executed or converted into optical data.',
            '- Original repository LICENSE is retained under source/LICENSE; upstream experimental data provenance remains as supplied.', '',
            '## Datasets', '',
            '| Group | Material | Samples | Range (nm) | Type | YAML |',
            '|---|---|---:|---:|---|---|']
    for item in records:
        low, high = item['wavelength_range_nm']
        link = 'database/' + quote(item['yaml_file'])
        rows.append(f'| {item["group"]} | {item["material"]} | {item["points"]} | {low:.9g}–{high:.9g} | '
                    f'{item["data_type"]} | [{Path(item["yaml_file"]).name}]({link}) |')
    (package / 'README.md').write_text('\n'.join(rows) + '\n', encoding='utf-8')
    print(json.dumps({'converted': len(records), 'verified_samples': sum(item['verified_samples'] for item in records),
                      'package': str(package), 'installed_database': str(install) if install else None}))


if __name__ == '__main__':
    main()
