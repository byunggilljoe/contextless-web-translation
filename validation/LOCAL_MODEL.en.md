# Local model inference: 7B

[한국어](LOCAL_MODEL.md) · [32B comparison on the same items](LOCAL_MODEL_32B.en.md)

Qwen2.5-Coder-7B-Instruct was run on three English, three German, two Spanish, and two Chinese items selected from the beginning of each language pair. **All 10 items completed with successful responses and parsing, with no generation truncation.**

All responses ended with `finish_reason=stop`. These results verify execution; several translations contain contextual errors.

## Environment

| Setting | Value |
|---|---|
| Run date | 2026-11-19 |
| Model | `Qwen/Qwen2.5-Coder-7B-Instruct` |
| Model revision | `c03e6d358207e414f1eca0bb1891e29f1db0e242` |
| GPU | NVIDIA A100-SXM4-80GB × 1 |
| Python | 3.12.2 |
| vLLM / PyTorch | 0.15.1 / 2.9.1 |
| Transformers / huggingface-hub | 4.57.6 / 0.36.2 |
| CUDA runtime / driver | 12.8 / 565.57.01 |
| HTML cleansing / context | Enabled / 128 characters per side |
| Generation | Temperature 0.01, maximum 32 tokens, seed 0 |
| Model settings | bfloat16, prefix caching enabled, tensor parallel 1 |

## Command

Run from the bundle root. Select the GPU and model cache location for your environment.

```bash
python -m pip install -r requirements-vllm.txt
python run.py translate --backend vllm --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --pair all --sample-ids en2ko:1 en2ko:2 en2ko:3 de2ko:1 de2ko:2 de2ko:3 \
  es2ko:1 es2ko:2 zh2ko:1 zh2ko:2 --window 128 --temperature 0.01 \
  --max-tokens 32 --seed 0 --gpu-memory-utilization 0.45 \
  --output outputs/local_model_10.jsonl
```

The cached model at the revision above was loaded offline. Check the model revision reported at startup and installed package versions when comparing runs.

## Item results

The Korean outputs and references below are preserved as recorded.

| ID | Source | Model translation | Manual reference |
|---|---|---|---|
| en2ko:1 | closure type | 드로스트링 | 잠금 장치 |
| en2ko:2 | dismiss | 닫기 | 닫기 |
| en2ko:3 | Helpful | 유용함 | 도움이돼요 |
| de2ko:1 | Stoffart | 재료 | 원단 |
| de2ko:2 | Herkunft | 원산지 | 원산지 |
| de2ko:3 | Nützlich | 유용함 | 도움이돼요 |
| es2ko:1 | Enviar a | 보내기 | 배송 |
| es2ko:2 | Devoluciones | 반품 | 반품 |
| zh2ko:1 | 访问 Amazon Essentials 品牌旗舰店 | Amazon Essentials 브랜드 오픈 스토어 방문 | Amazon Essentials 브랜드 공식스토어 방문 |
| zh2ko:2 | 面料类型 | 재질 | 소재 |

`dismiss → 닫기`, `Herkunft → 원산지`, and `Devoluciones → 반품` match their manual references. Other outputs use different expressions, such as `Helpful → 유용함` and `面料类型 → 재질`.

For `closure type`, the model generated the nearby value `드로스트링` (“drawstring”) instead of translating the field label. `Enviar a` was translated as `보내기` (“send”) in a shipping context. In the Chinese brand-store phrase, `品牌旗舰店` became `오픈 스토어` (“open store”) rather than the reference's `공식스토어` (“official store”). No separate LLM judging or human evaluation scores were assigned.

Full prompts, raw responses, and token usage are in the [result JSONL](local_model_10.jsonl). Settings, environment, and summary statistics are in the [run metadata](local_model_10.meta.json).
