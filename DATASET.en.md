# Dataset

[한국어](DATASET.md)

The dataset contains 1,018 markup translation items targeting Korean. The 174 HTML snapshots are counted per source language; they correspond to 74 distinct exact URL strings.

| Source → Korean | Items | HTML snapshots |
|---|---:|---:|
| English | 524 | 73 |
| German | 194 | 39 |
| Spanish | 200 | 36 |
| Chinese | 100 | 26 |
| Total | **1,018** | **174** |

## Directory structure

```text
data/
├── samples.jsonl
├── pages.json
├── html/
│   ├── en/{page_number}/original.html
│   ├── de/{page_number}/original.html
│   ├── es/{page_number}/original.html
│   └── zh/{page_number}/original.html
├── annotations/
│   └── webpage_chrome_manual_{en,de,es,zh}.csv
├── annotation_differences.json
└── provenance.json
```

Each line of `samples.jsonl` is one translation target. Multiple targets can refer to the same HTML snapshot. Join records to `pages.json` using `page_id`; open the snapshot using `html_path`, resolved from the bundle root. Page numbering can have gaps.

## Files

Paths below are relative to `data/`.

| File | Contents |
|---|---|
| `samples.jsonl` | One translation item per line |
| `pages.json` | Page IDs, source URLs, HTML paths, and SHA-256 checksums |
| `html/{en,de,es,zh}/{page_id}/original.html` | Source HTML snapshots |
| `annotations/*.csv` | Annotation CSVs, alternate Korean references, and other-language Chrome translations |
| `annotation_differences.json` | Differences between the annotation CSVs and experimental metadata |
| `provenance.json` | Source revision, file hashes, and export transformations |

## Record schema

| Field | Meaning |
|---|---|
| `id` | Unique `{language_pair}:{sample_id}`, e.g. `en2ko:1` |
| `sample_id` | 1-based item ID within a language pair |
| `language_pair` | `en2ko`, `de2ko`, `es2ko`, or `zh2ko` |
| `source_language`, `target_language` | Language codes |
| `page_id` | Language-specific snapshot ID, e.g. `en/0`; numbering can have gaps |
| `html_path` | HTML path relative to the bundle root |
| `url` | Source URL |
| `original` | Translation target, including stored whitespace |
| `chrome_translation` | Chrome translation collected with the dataset |
| `reference_translation` | Primary manual Korean translation; empty for one item |
| `references` | Primary and alternate manual references; CSV alternatives are included when the primary agrees with the experimental metadata |
| `exclude_from_user_survey` | Human-survey exclusion flag; set on 61 items |
| `failure_type` | Annotated translation failure category |

All 1,018 items are included by default. `--survey-only` excludes the 61 flagged items; it does not select participant-specific survey assignments.

No train/dev/test split is defined. For a new split, group items by underlying URL/site to keep strings from the same page and its language variants together.

## Example and loading

Selected fields from the first record (`en2ko:1`):

```json
{
  "id": "en2ko:1",
  "sample_id": 1,
  "language_pair": "en2ko",
  "page_id": "en/0",
  "html_path": "data/html/en/0/original.html",
  "original": "closure type",
  "chrome_translation": "패쇄형",
  "reference_translation": "잠금 장치",
  "references": ["잠금 장치"]
}
```

Run this Python example from the bundle root:

```python
import json
from pathlib import Path

root = Path(".")
with (root / "data/samples.jsonl").open(encoding="utf-8") as stream:
    samples = [json.loads(line) for line in stream if line.strip()]
pages = json.loads((root / "data/pages.json").read_text(encoding="utf-8"))
pages_by_id = {page["page_id"]: page for page in pages}

sample = samples[0]
page = pages_by_id[sample["page_id"]]
assert sample["html_path"] == page["html_path"]
html_bytes = (root / sample["html_path"]).read_bytes()
print(sample["id"], sample["original"], sample["reference_translation"])
print("HTML bytes:", len(html_bytes))
```

The runner resolves HTML paths from its installation directory. The example above uses the current directory; run it from the project root containing the code and `data/`.

## Annotation differences

`samples.jsonl` uses the experimental metadata as its primary source. The CSV annotations differ on four items:

| ID | Difference |
|---|---|
| `en2ko:8` | The experimental Chrome translation has a trailing space |
| `en2ko:67` | The experimental metadata has an empty manual reference and a different Chrome translation |
| `de2ko:5` | The experimental target has a trailing space (`Mehr `) |
| `de2ko:83` | The experimental target has a trailing space (`Sie sparen `) |

Both versions are recorded in `annotation_differences.json`. Reference scoring marks `en2ko:67` as `missing_translation`.

## Target matching

`process_html.py` cleans HTML, inserts target tags, and extracts a character window around the first marked occurrence (`TRAN_0`). Matching is case-insensitive and normalizes HTML text, including colon removal. The stored target is compared without trimming. No unique XPath is specified for an item.

With Beautiful Soup 4.12.3, 1,010 targets match after cleansing. The eight unmatched targets are:

| ID | Target |
|---|---|
| `en2ko:220` | `Subscribe` |
| `en2ko:337` | `Open positions` |
| `de2ko:5` | `Mehr ` |
| `de2ko:83` | `Sie sparen ` |
| `de2ko:146` | `Offene Stellen` |
| `de2ko:147` | `Büros` |
| `es2ko:169` | `Puestos disponibles` |
| `es2ko:170` | `Oficinas` |

The trailing spaces explain the matching failures for `de2ko:5` and `de2ko:83`. Failed items remain in the result files with status `preprocessing_error`. See [VALIDATION.json](VALIDATION.json) for the audit, or run `python validate.py --check-targets`.

## HTML snapshots

HTML bytes are preserved for comparisons with and without cleansing. Snapshots contain page scripts and embedded resources. The experiment runners parse HTML without executing JavaScript or fetching pages. External CSS, images, fonts, screenshots, and MHTML are not included, so browser rendering can differ from the source website.
