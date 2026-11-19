# Local 32B inference and comparison with 7B

[한국어](LOCAL_MODEL_32B.md)

`Qwen2.5-Coder-32B-Instruct` was run on three English, three German, two Spanish, and two Chinese items. **All 10 items completed with `status=ok` and `finish_reason=stop`, with no response-parsing failures or generation truncation.**

The 32B model used the same 10 items as the 7B model. Input message lists were identical for every item. Translation changes include both improvements and regressions; no aggregate quality score was assigned.

## Comparison settings

| Setting | Value |
|---|---|
| Run date | 2026-11-19 |
| Model | `Qwen/Qwen2.5-Coder-32B-Instruct` |
| Model revision | `381fc969f78efac66bc87ff7ddeadb7e73c218a7` |
| GPU | NVIDIA A100-SXM4-80GB × 1 |
| Python | 3.12.2 |
| vLLM / PyTorch | 0.15.1 / 2.9.1 |
| Transformers / huggingface-hub | 4.57.6 / 0.36.2 |
| CUDA runtime / driver | 12.8 / 565.57.01 |
| Model precision | bfloat16, unquantized |
| Context / HTML cleansing | 128 characters per side / enabled |
| Generation | Temperature 0.01, maximum 32 tokens, seed 0 |
| Batch size / prefix caching | 32 / enabled |
| Maximum model length / tensor parallel | 15,872 / 1 |

All 10 input message lists match the 7B run exactly. The execution options differ only in model ID and GPU memory utilization. Memory utilization was set to 0.45 for 7B and 0.9 for 32B to accommodate the model sizes. Loading the 32B weights used approximately 61 GiB of GPU memory.

## Command

Run from the bundle root. Select the GPU and model cache location for your environment.

```bash
python -m pip install -r requirements-vllm.txt
python run.py translate --backend vllm --model Qwen/Qwen2.5-Coder-32B-Instruct \
  --pair all --sample-ids en2ko:1 en2ko:2 en2ko:3 de2ko:1 de2ko:2 de2ko:3 \
  es2ko:1 es2ko:2 zh2ko:1 zh2ko:2 --window 128 --temperature 0.01 \
  --max-tokens 32 --seed 0 --gpu-memory-utilization 0.9 \
  --output outputs/local_model_32b_10.jsonl
```

The cached model at the revision above was loaded offline.

## Item comparison

The Korean outputs and references below are preserved as recorded.

| ID | Source | 7B translation | 32B translation | Manual reference |
|---|---|---|---|---|
| en2ko:1 | closure type | 드로스트링 | 단추 유형 | 잠금 장치 |
| en2ko:2 | dismiss | 닫기 | 닫기 | 닫기 |
| en2ko:3 | Helpful | 유용함 | 유용함 | 도움이돼요 |
| de2ko:1 | Stoffart | 재료 | 소재 | 원단 |
| de2ko:2 | Herkunft | 원산지 | 원산지 | 원산지 |
| de2ko:3 | Nützlich | 유용함 | 유용함 | 도움이돼요 |
| es2ko:1 | Enviar a | 보내기 | 보내기 | 배송 |
| es2ko:2 | Devoluciones | 반품 | 반품 | 반품 |
| zh2ko:1 | 访问 Amazon Essentials 品牌旗舰店 | Amazon Essentials 브랜드 오픈 스토어 방문 | 아마존 에센셜스 브랜드 공식 스토어 방문 | Amazon Essentials 브랜드 공식스토어 방문 |
| zh2ko:2 | 面料类型 | 재질 | 면단 종류 | 소재 |

## Review

- `Stoffart` changed from the 7B output `재료` (“material”) to `소재`, a more suitable term in the clothing context.
- The Chinese brand-store phrase changed from `오픈 스토어` (“open store”) to `공식 스토어` (“official store”).
- For `closure type`, 32B generated `단추 유형` (“button type”) instead of the nearby value `드로스트링` (“drawstring”), but it still differs from the reference `잠금 장치` (“closure/fastening”).
- `Enviar a → 보내기` and `Helpful / Nützlich → 유용함` were unchanged. These outputs do not fully reflect the shipping and helpfulness-rating UI contexts.
- `面料类型` changed from `재질` to the less natural expression `면단 종류`.

Six items have identical translation strings across models, and four differ. These are string comparisons, not quality scores. No separate LLM judging or human evaluation scores were assigned to this small run.

[32B raw results](local_model_32b_10.jsonl) · [32B run metadata](local_model_32b_10.meta.json) · [7B results](LOCAL_MODEL.en.md)
