"""Offline dataset integrity check. No model, credentials, or GPU is required."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from run import ROOT, fragment, read_jsonl


def validate(check_targets=False):
    samples = read_jsonl(ROOT / 'data/samples.jsonl')
    pages = json.loads((ROOT / 'data/pages.json').read_text(encoding='utf-8'))
    assert len(samples) == 1018, 'Expected 1,018 benchmark samples'
    assert len({s['id'] for s in samples}) == 1018, 'Duplicate sample IDs'
    counts = dict(Counter(s['language_pair'] for s in samples))
    assert counts == {'en2ko': 524, 'de2ko': 194, 'es2ko': 200, 'zh2ko': 100}
    assert len(pages) == len({s['page_id'] for s in samples}) == 174
    by_page = {p['page_id']: p for p in pages}
    assert len(by_page) == len(pages)
    for page in pages:
        path = (ROOT / page['html_path']).resolve()
        assert path.is_relative_to(ROOT), 'HTML path escapes bundle'
        assert path.is_file(), f'Missing HTML: {page["page_id"]}'
        assert not path.is_symlink(), f'External symlink: {page["page_id"]}'
        assert hashlib.sha256(path.read_bytes()).hexdigest() == page['sha256'], f'HTML checksum mismatch: {page["page_id"]}'
    for sample in samples:
        assert sample['page_id'] in by_page
        assert sample['html_path'] == by_page[sample['page_id']]['html_path']
        assert sample['url'] == by_page[sample['page_id']]['url']
    manifest = ROOT / 'SHA256SUMS'
    verified = 0
    if manifest.exists():
        for line in manifest.read_text(encoding='utf-8').splitlines():
            digest, rel = line.split('  ', 1)
            path = (ROOT / rel).resolve()
            assert path.is_relative_to(ROOT), 'Manifest path escapes bundle'
            assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, f'Checksum mismatch: {rel}'
            verified += 1
    result = {'samples': len(samples), 'samples_by_language_pair': counts,
              'language_specific_pages': len(pages),
              'distinct_exact_urls': len({s['url'] for s in samples}),
              'html_checksums_verified': len(pages), 'manifest_files_verified': verified,
              'excluded_from_user_survey': sum(s['exclude_from_user_survey'] for s in samples),
              'missing_reference_ids': [s['id'] for s in samples if not s['reference_translation'].strip()]}
    if check_targets:
        failures = []
        for n, sample in enumerate(samples, 1):
            try:
                text = fragment(sample)
                assert '<TRAN_0>' in text and '</TRAN_0>' in text
            except (ValueError, AssertionError) as exc:
                failures.append({'id': sample['id'], 'reason': str(exc)})
            if n % 100 == 0:
                print(f'Checked {n}/{len(samples)} targets', flush=True)
        result['target_matching'] = {'checked': len(samples), 'matched': len(samples) - len(failures),
                                     'failures': failures}
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-targets', action='store_true', help='Also parse all HTML with the original preprocessing')
    parser.add_argument('--report', type=Path, help='Save a new JSON report')
    args = parser.parse_args()
    result = validate(args.check_targets)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.report:
        with args.report.open('x', encoding='utf-8') as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
