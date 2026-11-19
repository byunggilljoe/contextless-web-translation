"""Small full-page Q&A implementation: shared block prompts, selected nodes,
parallel generation, and text-only reinsertion into the original DOM.

This is a portable adaptation, not the original timing experiment harness.
"""
import argparse
from copy import deepcopy
from pathlib import Path
import re
import time

from bs4 import Comment, NavigableString
from process_html import clean, read_html_file
from run import (add_model, backend, positive_int, nonnegative_int,
                 write_json, write_jsonl)

BASE_PROMPT = '''
- The following is a HTML of a website
####
{html}
####

- Find a text in a tag and translate it into Korean (한글). The text in the HTML can be identified by a <T_(NUMBER)> tags.
- If translation is not needed, just return the original text.
- Only translated text required. Do not provide any explanation or preambles.
- Response format
    <T_(NUMBER)> TEXT_BEFORE_TRANSLATION </T_(NUMBER)> --> TRANSLATED_TEXT_HERE
- Response Examples
    <T_123> Hello  --> 안녕하세요
    <T_124> Where are you from? --> 어디서 오셨어요?
    <T_125> 안녕하세요. -->  안녕하세요.
'''


def plan(original, block_size, side, selected=None):
    cleaned = deepcopy(original)
    before = list(original.descendants)
    after = list(cleaned.descendants)
    original_by_copy = {id(copy): source for source, copy in zip(before, after)
                        if isinstance(copy, NavigableString)}
    clean(cleaned, as_new_bs=False)
    nodes = [node for node in cleaned.find_all(string=True)
             if not isinstance(node, Comment) and node.strip()
             and node.parent.name not in ('script', 'style', 'svg', 'path')]
    originals, texts = [], []
    for i, node in enumerate(nodes):
        originals.append(original_by_copy[id(node)])
        texts.append(str(node))
        wrapper = cleaned.new_tag(f'T_{i}')
        node.wrap(wrapper)
    wanted = set(range(len(nodes))) if selected is None else set(selected)
    if wanted - set(range(len(nodes))):
        raise ValueError('Selected node ID is outside the page node range')
    marked = cleaned.prettify()
    jobs = []
    for start in range(0, len(nodes), block_size):
        end = min(start + block_size, len(nodes))
        left = marked.index(f'<T_{start}>')
        closing = f'</T_{end - 1}>'
        right = marked.index(closing) + len(closing)
        context = marked[max(0, left - side):right + side]
        base = BASE_PROMPT.format(html=context)
        for i in range(start, end):
            if i in wanted:
                jobs.append({'node_id': i, 'block_id': start // block_size,
                             'original': texts[i], 'status': 'prepared',
                             'messages': [{'role': 'user', 'content': base +
                              f'- You need to translate <T_{i}> into Korean (한글)\n- 당신의 답변:\n'}]})
    return originals, jobs


def parse(response, node_id):
    # Do not apply an incomplete generation to the page.
    if '-->' in response:
        result = response.rsplit('-->', 1)[-1].split('</', 1)[0].strip()
    else:
        match = re.search(fr'<T_{node_id}>(.*?)</T_{node_id}>', response, re.DOTALL)
        result = match[1].strip() if match else response.strip()
    return result or None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--block-size', type=positive_int, default=32, help='Text nodes sharing one context prefix')
    parser.add_argument('--side-chars', type=nonnegative_int, default=10)
    parser.add_argument('--ids', type=nonnegative_int, nargs='+', help='Translate only these zero-based node IDs')
    add_model(parser)
    args = parser.parse_args()
    paths = [args.output, Path(str(args.output) + '.nodes.jsonl'), Path(str(args.output) + '.meta.json')]
    if any(path.exists() for path in paths):
        parser.error('Output files already exist; choose a new output path')
    if not args.dry_run and not args.model:
        parser.error('--model is required for live inference')
    original = read_html_file(args.input)
    if original is None:
        parser.error('Cannot read HTML')
    # Parsing and reserializing may normalize markup; compare against parsed DOM.
    structure = [(t.name, dict(t.attrs)) for t in original.find_all()]
    nodes, jobs = plan(original, args.block_size, args.side_chars, args.ids)
    engine = None
    started = time.perf_counter()
    for start in range(0, len(jobs), args.batch_size):
        batch = jobs[start:start + args.batch_size]
        if args.dry_run:
            for row in batch:
                row['status'] = 'dry_run'
            continue
        if engine is None:
            engine = backend(args)
        responses = engine.generate([row['messages'] for row in batch])
        if len(responses) != len(batch):
            raise ValueError('Backend response count does not match job count')
        for row, response in zip(batch, responses):
            row.update(response)
            translation = parse(response.get('raw_response', ''), row['node_id'])
            row['status'] = ('backend_error' if 'backend_error' in response else
                             'truncated' if response.get('finish_reason') == 'length' else
                             'ok' if translation is not None else 'parse_error')
            row['translation'] = translation
            if row['status'] == 'ok':
                # replace_with(string) escapes any markup emitted by the model.
                nodes[row['node_id']].replace_with(translation)
    assert structure == [(t.name, dict(t.attrs)) for t in original.find_all()]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        stream.write(str(original))
    write_jsonl(paths[1], jobs)
    summary = {'total_visible_nodes': len(nodes), 'selected_nodes': len(jobs),
               'translated_nodes': sum(r['status'] == 'ok' for r in jobs),
               'failed_nodes': sum(r['status'] not in ('ok', 'dry_run') for r in jobs),
               'block_size': args.block_size, 'side_chars': args.side_chars,
               'backend': args.backend, 'model': args.model, 'dry_run': args.dry_run,
               'batch_size': args.batch_size, 'workers': args.workers,
               'temperature': args.temperature, 'max_tokens': args.max_tokens,
               'seed_vllm_only': args.seed, 'prefix_cache_vllm': not args.no_prefix_cache,
               'seconds_including_engine_startup': time.perf_counter() - started,
               'dom_tags_and_attributes_preserved': True}
    write_json(paths[2], summary)
    print(summary)
    return int(summary['failed_nodes'] > 0)


if __name__ == '__main__':
    raise SystemExit(main())
