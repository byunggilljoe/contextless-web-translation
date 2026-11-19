# Code overview

[한국어](PROJECT_OVERVIEW.md)

The method uses surrounding HTML to interpret a translation target and generates the translation of that target. During page translation, multiple nodes share a common context prefix and their questions are processed in parallel.

## Processing flow

1. Read translation targets and HTML paths from `data/samples.jsonl`.
2. Clean HTML and insert a `TRAN_0` tag around the target using `process_html.py`.
3. Build a Q&A prompt with a character window around the target tag.
4. Execute translation through direct vLLM inference or a Chat Completions API using `run.py`.
5. Save translations, model responses, and statuses to JSONL, and settings and summary statistics to `.meta.json`.
6. Use `run.py score` to send cleansed HTML context and a translation to the judge model for a score from 1 to 5.

## Files

| Path | Role |
|---|---|
| `data/samples.jsonl` | Translation targets, language pairs, Chrome translations, and manual reference translations |
| `data/html/` | HTML snapshots by source language |
| `data/pages.json` | Page sources, paths, and hashes |
| `data/annotations/` | Annotation CSVs |
| `process_html.py` | HTML cleansing, text normalization, target tagging, and context extraction |
| `run.py` | Translation/evaluation CLI and API/vLLM backends |
| `translate_page.py` | Text-node selection, block context, parallel translation, and DOM reinsertion |
| `validate.py` | Dataset count, path, hash, and target-matching checks |
| `tests/` | Integration tests using a mock API |
| `validation/` | 7B/32B local GPU inference results, settings, and reviews |
| `reference/` | Reference implementations for quality, caching, and timing experiments |
| `maintainer/` | Dataset export and ZIP packaging tools |

## Page translation

`translate_page.py` assigns IDs such as `T_0` and `T_1` to text nodes in cleansed HTML. Groups of `--block-size` nodes share HTML context, and `--ids` selects the nodes to translate. Each translation replaces the corresponding text node in the original DOM. Tags and attributes are preserved at the parsed DOM level.

`--window` in target-string experiments counts characters per side. `--block-size` in page translation counts text nodes.
