"""Export benchmark inputs and the runnable HTML preprocessing module.

Run from any directory: python export_from_project.py --project-root /path/to/project
No credentials, participant responses, resources, MHTML, or model files are copied.
"""
import argparse
import csv
import hashlib
import json
import shutil
import subprocess
from collections import Counter
from pathlib import Path

BUNDLE = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def export(project):
    records, pages, differences, provenance = [], {}, [], []
    for source in ('en', 'de', 'es', 'zh'):
        pair = source + '2ko'
        csv_rel = f'data/survey/raw-survey-data/csv_files/webpage_chrome_manual_{source}.csv'
        json_rel = f'data/survey/preprocessed-survey-data/{pair}/webpage_chrome_manual.json'
        csv_path, json_path = project / csv_rel, project / json_rel
        with csv_path.open(encoding='utf-8-sig', newline='') as stream:
            annotations = list(csv.DictReader(stream))
        canonical = json.loads(json_path.read_text(encoding='utf-8'))
        assert len(annotations) == len(canonical)
        dest = BUNDLE / 'data/annotations' / csv_path.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(csv_path, dest)
        provenance.extend({'source': rel, 'sha256': digest(project / rel)}
                          for rel in (csv_rel, json_rel))
        for row, item in zip(annotations, canonical):
            sid = item['sample_id']
            changed = {}
            for col, val in [('원래단어', item['original']),
                             ('어색한 단어', item['translation_approach']['chrome']),
                             ('적합단어 1', item['translation_manual'])]:
                if row[col] != val:
                    changed[col] = {'annotation_csv': row[col], 'experiment_json': val}
            if changed:
                differences.append({'id': f'{pair}:{sid}', 'differences': changed})
            original_path = project / item['file_original']
            page_id = f'{source}/{original_path.parent.name}'
            html_path = f'data/html/{page_id}/original.html'
            if page_id not in pages:
                dest = BUNDLE / html_path
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(original_path, dest)
                pages[page_id] = {'page_id': page_id, 'html_path': html_path,
                                 'url': item['URL'], 'sha256': digest(dest)}
            refs = [item['translation_manual']] if item['translation_manual'].strip() else []
            # Do not silently replace a canonical missing reference with CSV text.
            if row['적합단어 1'] == item['translation_manual']:
                refs.extend(row[k] for k in ('적합단어 2', '적합단어 3') if row[k].strip())
            records.append({
                'id': f'{pair}:{sid}', 'sample_id': sid, 'language_pair': pair,
                'source_language': source, 'target_language': 'ko',
                'page_id': page_id, 'html_path': html_path, 'url': item['URL'],
                'original': item['original'], 'chrome_translation': item['translation_approach']['chrome'],
                'reference_translation': item['translation_manual'], 'references': refs,
                'exclude_from_user_survey': item['exclude_from_user_survey'],
                'failure_type': row['failure type'],
            })
    assert len(records) == 1018 and len(pages) == 174
    dest = BUNDLE / 'data/samples.jsonl'
    dest.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records), encoding='utf-8')
    dump(BUNDLE / 'data/pages.json', list(pages.values()))
    dump(BUNDLE / 'data/annotation_differences.json', differences)
    rel = 'common/process_html.py'
    dest = BUNDLE / 'process_html.py'
    shutil.copyfile(project / rel, dest)
    provenance.append({'source': rel, 'source_sha256': digest(project / rel),
                       'exported_sha256': digest(dest)})
    revision = subprocess.run(['git', '-C', str(project), 'rev-parse', 'HEAD'],
                              check=True, capture_output=True, text=True).stdout.strip()
    dump(BUNDLE / 'data/provenance.json', {
        'source_git_commit': revision,
        'note': 'Working-tree files were exported; hashes identify their actual contents. '
                'The working tree contained local changes; the commit alone is not a full provenance record.',
        'samples': len(records), 'language_specific_pages': len(pages),
        'distinct_exact_urls': len({r['url'] for r in records}),
        'samples_by_language_pair': dict(Counter(r['language_pair'] for r in records)),
        'canonical_metadata': 'preprocessed-survey-data JSON; not the raw annotation CSV',
        'excluded_from_user_survey': sum(r['exclude_from_user_survey'] for r in records),
        'missing_primary_reference': [r['id'] for r in records if not r['reference_translation'].strip()],
        'source_files': provenance,
    })
    print(f'Exported {len(records)} samples and {len(pages)} language-specific pages.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, required=True)
    export(parser.parse_args().project_root.resolve())
