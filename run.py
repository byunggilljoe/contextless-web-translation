"""Portable benchmark runner. All relative input/output paths use the current directory.

Dataset HTML paths are resolved against this file, independently of cwd.
Only translate/score without --dry-run contact the selected model backend.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import importlib.metadata
import json
import os
from pathlib import Path
import re
import sys
import time
from urllib import error, request

from bs4 import BeautifulSoup
from process_html import clean, extract_translate_tag, inject_translate_tag, read_html_file

ROOT = Path(__file__).resolve().parent
LANGUAGES = {'en': 'English', 'de': 'German', 'es': 'Spanish', 'zh': 'Chinese', 'ko': 'Korean'}
COMMENTS = {'English': 'comments', 'German': 'Kommentare', 'Spanish': 'comentarios',
            'Chinese': '评论', 'Korean': '댓글'}


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
            stream.flush()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


@lru_cache(maxsize=4)
def page_soup(html_path, do_clean=True):
    soup = read_html_file(ROOT / html_path)
    if soup is None:
        raise ValueError('unreadable_html')
    return clean(soup) if do_clean else soup


def fragment(sample, window=128, do_clean=True):
    """Use the original matching, first occurrence, prettify, and character window."""
    marked = inject_translate_tag(page_soup(sample['html_path'], do_clean), sample['original'], 'TRAN')
    if marked is None:
        raise ValueError('target_not_found')
    result = extract_translate_tag(marked, 'TRAN_0', window, window)
    if result is None:
        raise ValueError('target_not_found')
    return result


def qa_messages(sample, html):
    source = LANGUAGES[sample['source_language']]
    target = LANGUAGES[sample['target_language']]
    # System and user wording from translate_target_string/ray_version.py.
    return [
        {'role': 'system', 'content': f'You are an expert in translating web page into {target} considering surrounding context. It is okay to change part of speech into proper one regarding {target} websites. Translate in a natural {target} language rather than a translation tone. '},
        {'role': 'user', 'content': f'''- The following is a HTML of a website
####
{html}
####

#Instruction

- Find a text between <TRAN_0> and </TRAN_0> tags and translate it into {target}.
- Do not add preambles.
- Return the translated text wrapped in the same tags.
  For example, your response should be like this:
    <TRAN_0>{COMMENTS[source]}</TRAN_0> -> <TRAN_0>{COMMENTS[target]}</TRAN_0>
'''}]


def contextless_messages(sample):
    # A clearly identified new baseline; --mode qa --window 0 is the original
    # zero-window tagged-HTML ablation, which is different from bare text.
    return [{'role': 'system', 'content': 'You are an expert translator.'},
            {'role': 'user', 'content': f'Translate the following {LANGUAGES[sample["source_language"]]} text into '
             f'{LANGUAGES[sample["target_language"]]}. Return only the translation wrapped in '
             f'<TRAN_0> and </TRAN_0> tags.\nText: {sample["original"]}'}]


def select_samples(args):
    samples = read_jsonl(args.data)
    if args.pair != 'all':
        samples = [s for s in samples if s['language_pair'] == args.pair]
    if args.sample_ids:
        requested = set(args.sample_ids)
        samples = [s for s in samples if s['id'] in requested]
        missing = requested - {s['id'] for s in samples}
        if missing:
            raise ValueError(f'Unknown or filtered sample IDs: {sorted(missing)}')
    if args.survey_only:
        samples = [s for s in samples if not s['exclude_from_user_survey']]
    if args.limit is not None:
        samples = samples[:args.limit]
    if not samples:
        raise ValueError('No samples selected')
    return samples


def prepare(sample, args):
    row = {'id': sample['id'], 'language_pair': sample['language_pair'],
           'page_id': sample['page_id'], 'original': sample['original'],
           'mode': args.mode, 'window_chars_per_side': args.window,
           'clean_html': not args.no_clean, 'status': 'prepared'}
    try:
        row['messages'] = (qa_messages(sample, fragment(sample, args.window, not args.no_clean))
                           if args.mode == 'qa' else contextless_messages(sample))
    except (ValueError, AssertionError) as exc:
        row.update(status='preprocessing_error', error=str(exc))
    return row


class APIBackend:
    """Chat Completions compatible API, including a local `vllm serve` endpoint."""
    def __init__(self, args):
        self.args = args
        self.base = args.base_url or os.environ.get('OPENAI_BASE_URL')
        if not self.base:
            raise ValueError('Set OPENAI_BASE_URL or --base-url (including /v1 when applicable)')
        self.key = os.environ.get('OPENAI_API_KEY', '')

    def one(self, messages):
        payload = json.dumps({'model': self.args.model, 'messages': messages,
                              'temperature': self.args.temperature,
                              'max_tokens': self.args.max_tokens}).encode()
        headers = {'Content-Type': 'application/json'}
        if self.key:
            headers['Authorization'] = 'Bearer ' + self.key
        for attempt in range(3):
            try:
                req = request.Request(self.base.rstrip('/') + '/chat/completions', data=payload, headers=headers)
                with request.urlopen(req, timeout=self.args.timeout) as response:
                    body = json.load(response)
                choice = body['choices'][0]
                content = choice['message']['content']
                if not isinstance(content, str):
                    raise ValueError('empty_or_nontext_response')
                return {'raw_response': content, 'finish_reason': choice.get('finish_reason'),
                        'usage': body.get('usage')}
            except error.HTTPError as exc:
                if exc.code in (429, 500, 502, 503, 504) and attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                # Do not copy service error bodies, endpoints, or credentials into results.
                return {'backend_error': f'HTTP {exc.code}'}
            except (error.URLError, TimeoutError, OSError) as exc:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                return {'backend_error': type(exc).__name__}
            except (ValueError, KeyError, IndexError, TypeError) as exc:
                return {'backend_error': type(exc).__name__}

    def generate(self, messages):
        with ThreadPoolExecutor(max_workers=self.args.workers) as pool:
            return list(pool.map(self.one, messages))


class VLLMBackend:
    def __init__(self, args):
        from vllm import LLM, SamplingParams
        self.args = args
        self.engine = LLM(model=args.model, tensor_parallel_size=args.tensor_parallel_size,
                          max_model_len=args.max_model_len,
                          gpu_memory_utilization=args.gpu_memory_utilization,
                          enable_prefix_caching=not args.no_prefix_cache)
        self.tokenizer = self.engine.get_tokenizer()
        self.params = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens, seed=args.seed)

    def generate(self, messages):
        prompts = [self.tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                   for m in messages]
        results = self.engine.generate(prompts, self.params, use_tqdm=False)
        return [{'raw_response': r.outputs[0].text, 'finish_reason': r.outputs[0].finish_reason,
                 'usage': {'prompt_tokens': len(r.prompt_token_ids),
                           'completion_tokens': len(r.outputs[0].token_ids)}} for r in results]


def backend(args):
    if not args.model:
        raise ValueError('--model is required for live inference')
    return APIBackend(args) if args.backend == 'api' else VLLMBackend(args)


def parse_translation(response):
    matches = re.findall(r'<TRAN_0>(.*?)</TRAN_0>', response, re.DOTALL | re.IGNORECASE)
    if matches and matches[-1].strip():
        return matches[-1].strip()
    # The original short generation cap can omit a closing tag.
    tail = re.search(r'<TRAN_0>([^<]+)$', response, re.DOTALL | re.IGNORECASE)
    return tail[1].strip() if tail and tail[1].strip() else None


def execute(rows, args, parse):
    engine = None
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start:start + args.batch_size]
        jobs = [r for r in batch if r['status'] == 'prepared']
        if args.dry_run:
            for row in jobs:
                row['status'] = 'dry_run'
        elif jobs:
            if engine is None:
                engine = backend(args)
            responses = engine.generate([r['messages'] for r in jobs])
            if len(responses) != len(jobs):
                raise ValueError('Backend response count does not match job count')
            for row, response in zip(jobs, responses):
                row.update(response)
                if 'backend_error' in response:
                    row['status'] = 'backend_error'
                    continue
                value = parse(response['raw_response'])
                field = 'score' if args.command == 'score' else 'translation'
                row[field] = value
                row['status'] = ('truncated' if response.get('finish_reason') == 'length' else
                                 'ok' if value is not None else 'parse_error')
        yield from batch
        print(f'Processed {min(start + args.batch_size, len(rows))}/{len(rows)}', file=sys.stderr)


def judge_messages(sample, translation):
    source = LANGUAGES[sample['source_language']].lower()
    target = LANGUAGES[sample['target_language']].lower()
    # Rubric from survey_llm/json_answer_to_csv.py. The original script named
    # languages "english"/"korean" but compared them to "ko", so its optional
    # Korean UI example was never inserted; keep that actual prompt behavior.
    criteria = f'''1. Meaning is incorrect; the cause of the error cannot be determined.
2. Meaning is incorrect.
3. Meaning is conveyed correctly in context.
4. Meaning is correct, but the {target} expression is awkward. 
5. Meaning is correct and the {target} expression sounds natural in the web context.
'''
    return [{'role': 'user', 'content': f'''Your are a {source}-to-{target} translator.
Here is 
    - a fragment HTML of a webpage where the translation target string is marked with <TRAN_0> tag.
    - Translation of the target string in {target}.
    - Evaluation criteria for the translation.
You are required to score the {target} translation according to the criteria.

== Fragment of a webpage ==
{fragment(sample, 512)}

== {target} Translation of the target string ==
{translation}

== Evaluation Criteria ==
{criteria}

== Score ==
Your answer should be a number between 1 and 5.
Finish your answer with a newline only with score in the format of "score: <score>".
'''}]


def parse_score(response):
    matches = re.findall(r'^\s*score:\s*([1-5])\s*$', response, re.MULTILINE | re.IGNORECASE)
    return int(matches[-1]) if matches else None


def score_jobs(samples, args):
    predictions = read_jsonl(args.predictions) if args.predictions else []
    by_id = {r['id']: r for r in predictions}
    if len(by_id) != len(predictions):
        raise ValueError('Duplicate prediction IDs')
    selected_ids = {s['id'] for s in samples}
    if args.predictions and selected_ids.isdisjoint(by_id):
        raise ValueError('Predictions do not match the selected dataset')
    rows = []
    for sample in samples:
        row = {'id': sample['id'], 'language_pair': sample['language_pair'], 'status': 'prepared'}
        prediction = by_id.get(sample['id'], {})
        translation = (sample['chrome_translation'] if args.candidate == 'chrome' else
                       sample['reference_translation'] if args.candidate == 'reference' else
                       prediction.get('translation') if prediction.get('status') == 'ok' else None)
        row['candidate'] = args.candidate
        row['translation'] = translation
        if not translation or not translation.strip():
            row['status'] = 'missing_translation'
        else:
            try:
                row['messages'] = judge_messages(sample, translation)
            except (ValueError, AssertionError) as exc:
                row.update(status='preprocessing_error', error=str(exc))
        rows.append(row)
    return rows


def summary(rows):
    result = {'total': len(rows), 'status_counts': dict(Counter(r['status'] for r in rows))}
    if any('candidate' in r for r in rows):
        valid = [r['score'] for r in rows if r['status'] == 'ok' and r.get('score') is not None]
        result.update(scored=len(valid), coverage=len(valid) / len(rows) if rows else 0,
                      mean_valid=sum(valid) / len(valid) if valid else None,
                      mean_missing_as_1=(sum(valid) + len(rows) - len(valid)) / len(rows)
                      if rows and not any(r['status'] == 'dry_run' for r in rows) else None)
    return result


def add_selection(parser):
    parser.add_argument('--data', type=Path, default=ROOT / 'data/samples.jsonl')
    parser.add_argument('--pair', default='all', choices=['all', 'en2ko', 'de2ko', 'es2ko', 'zh2ko'])
    parser.add_argument('--sample-ids', nargs='+')
    parser.add_argument('--limit', type=positive_int)
    parser.add_argument('--survey-only', action='store_true', help='Exclude flagged items, not an exact recreation of survey assignments')
    parser.add_argument('--output', type=Path, required=True, help='New output file; existing files are never overwritten')


def positive_int(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError('must be positive')
    return result


def nonnegative_int(value):
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError('must be nonnegative')
    return result


def add_model(parser):
    parser.add_argument('--backend', choices=['api', 'vllm'], default='api')
    parser.add_argument('--model')
    parser.add_argument('--base-url', help='Defaults to OPENAI_BASE_URL')
    parser.add_argument('--workers', type=positive_int, default=4)
    parser.add_argument('--batch-size', type=positive_int, default=32)
    parser.add_argument('--temperature', type=float, default=0.01)
    parser.add_argument('--max-tokens', type=positive_int, default=32)
    parser.add_argument('--timeout', type=positive_int, default=180)
    parser.add_argument('--seed', type=nonnegative_int, default=0, help='vLLM only; API randomness is server-dependent')
    parser.add_argument('--tensor-parallel-size', type=positive_int, default=1)
    parser.add_argument('--max-model-len', type=positive_int, default=15872)
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.9)
    parser.add_argument('--no-prefix-cache', action='store_true', help='Direct vLLM only; API cache is server-controlled')
    parser.add_argument('--dry-run', action='store_true', help='Save actual prompts without loading or calling a model')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    cleaning = commands.add_parser('clean', help='Apply the original HTML cleansing implementation')
    cleaning.add_argument('--input', type=Path, required=True)
    cleaning.add_argument('--output', type=Path, required=True)
    for name in ('prepare', 'translate'):
        sub = commands.add_parser(name)
        add_selection(sub)
        sub.add_argument('--mode', choices=['qa', 'contextless'], default='qa')
        sub.add_argument('--window', type=nonnegative_int, default=128, help='Characters per side, not tokens')
        sub.add_argument('--no-clean', action='store_true')
        if name == 'translate':
            add_model(sub)
    scoring = commands.add_parser('score')
    add_selection(scoring)
    add_model(scoring)
    scoring.set_defaults(max_tokens=512)
    scoring.add_argument('--candidate', choices=['predictions', 'chrome', 'reference'], default='predictions')
    scoring.add_argument('--predictions', type=Path)
    args = parser.parse_args()
    if args.command == 'clean':
        soup = read_html_file(args.input)
        if soup is None:
            parser.error('Cannot read HTML')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x', encoding='utf-8') as stream:
            stream.write(str(clean(soup)))
        return 0
    if args.output.exists() or Path(str(args.output) + '.meta.json').exists():
        parser.error('Output or metadata already exists; choose a new path')
    if args.command == 'score' and args.candidate == 'predictions' and not args.predictions:
        parser.error('--predictions is required for --candidate predictions')
    if args.command in ('translate', 'score') and not args.dry_run and not args.model:
        parser.error('--model is required for live inference')
    samples = select_samples(args)
    rows = score_jobs(samples, args) if args.command == 'score' else [prepare(s, args) for s in samples]
    if args.command != 'prepare':
        write_jsonl(args.output, execute(rows, args, parse_score if args.command == 'score' else parse_translation))
    else:
        write_jsonl(args.output, rows)
    # Re-read so the metadata also describes the actual on-disk artifact.
    result = summary(read_jsonl(args.output))
    packages = {}
    for package in ('beautifulsoup4', 'soupsieve', 'vllm', 'transformers', 'torch'):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    # API URL is unnecessary for a shareable experimental record.
    config.pop('base_url', None)
    write_json(str(args.output) + '.meta.json', {'config': config, 'packages': packages,
               'python': sys.version.split()[0], 'summary': result})
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return int(any(r['status'] not in ('ok', 'prepared', 'dry_run') for r in read_jsonl(args.output)))


if __name__ == '__main__':
    raise SystemExit(main())
