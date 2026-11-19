# Reference implementations

Source implementations for inspecting the translation method and experiment settings. Run experiments with `run.py` and `translate_page.py` in the bundle root; the files here retain their research project imports and paths.

| Source | Role |
|---|---|
| `common/process_html.py` | HTML cleansing, normalization, target tagging, and context extraction |
| `preprocess_data.py` | CSV processing and HTML preparation |
| `translate_target_string/ray_version.py` | Target-string Q&A prompts and model execution |
| `translate_target_string/ray_version_no_prompt_context.py` | No-context experiment |
| `translate_all/main.py` | Full-page Q&A, DOM mapping, block selection, and cache reuse |
| `translate_all/speedup_tests/huggingface_use_cache_test.py` | Transformers cache and batch generation |
| `translate_all/main_vllm.py` | Full-page vLLM translation and timing experiments |
| `translate_all/timing.py`, `translate_all/animation.py` | Timing and animation helpers |
| `survey_llm/json_answer_to_csv.py` | Evaluation prompt and 1–5 rubric |
| `configs/gen_params.py`, `script/*.sh` | Generation settings and experiment launch commands |

## Relation to the runnable code

`process_html.py` in the bundle root matches `common/process_html.py`. `run.py` uses the target Q&A wording and matching/window functions with API or direct vLLM execution. Its `contextless` condition provides bare target text, while `qa --window 0` retains the tagged target and instructions. The runner records unmatched targets, malformed responses, and truncations explicitly. Scores require the `score: N` format; parsing failures remain unscored.

`translate_page.py` uses block-context Q&A with text-node mapping and a single user message per request. The reference full-page implementations include assistant-prefill and cache-management details that differ from this runner. Timing results from these execution paths are not directly interchangeable.

File versions, hashes, and export transformations are recorded in `data/provenance.json`. The snapshot includes extensions beyond the publication-time experiments.
