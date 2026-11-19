# Validation

[한국어](VALIDATION.md)

Environment: Python 3.12, Beautiful Soup 4.12.3, and soupsieve 2.6.

## Dataset checks

| Check | Result |
|---|---|
| Translation item count and unique IDs | 1,018 verified |
| HTML paths and SHA-256 hashes | 174 verified |
| Target matching after cleansing | 1,010 matched, 8 unmatched |
| Empty primary manual references | 1 |

See [VALIDATION.json](VALIDATION.json) for audit results and [DATASET.en.md](DATASET.en.md) for known dataset issues.

```bash
python validate.py
python validate.py --check-targets
```

The first command checks dataset structure and `SHA256SUMS`. The second also checks HTML cleansing and target matching for all items.

## Execution tests

```bash
python -m unittest discover -s tests -v
```

All six integration tests passed. A local HTTP mock server checks translation and score persistence, authentication failures, malformed responses, token truncation, missing references, selected-node reinsertion, and overwrite protection. The bare-text condition is checked to exclude HTML and reference translations from its input.

Prompt generation, cleansing settings, context-window changes, and selected-node ID generation were also checked with actual HTML snapshots.

## Live local inference

Qwen2.5-Coder-7B-Instruct and 32B-Instruct were each run on one NVIDIA A100-SXM4-80GB. Both models used the same 10 items: three English, three German, two Spanish, and two Chinese. All 10 items for each model completed with `status=ok` and `finish_reason=stop`. There were no target-matching failures, response-parsing failures, or generation truncations.

Environment: Python 3.12.2, vLLM 0.15.1, PyTorch 2.9.1, Transformers 4.57.6, and huggingface-hub 0.36.2. The tested versions are pinned in `requirements-vllm.txt`.

See the [7B/32B comparison](validation/LOCAL_MODEL_32B.en.md) for sources, translations, manual references, commands, and settings. Some contextual errors were observed; translation quality requires separate evaluation.

The 32B model was also run with `--mode contextless` for the five README items. All five completed with `status=ok` and `finish_reason=stop`. The contextful outputs come from `--mode qa --window 128`. Model revision, temperature, generation cap, and seed match across conditions; each mode uses its own prompt format. [Five-example comparison](examples/comparison_5.json)

Each of the five README examples has source, contextless, and contextful screenshots, for 15 comparison images in total. Original page overviews for the three language-specific pages are also included. Target text was checked against recorded inference outputs, and PNG files and source HTML hashes were verified.

## Reproduction scope

Commercial API calls and the paper's complete quality/timing tables have not been verified. The integration tests in `tests/` use a mock server. The files in `validation/` contain actual GPU inference results for the same 10 items with each model.

When comparing experiments, record model revisions, GPU/server environments, library versions, and generation settings. Run settings and installed package versions are available in each result's `.meta.json`.
