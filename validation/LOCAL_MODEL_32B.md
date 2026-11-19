# 32B 로컬 모델 실행 및 7B 비교

[English](LOCAL_MODEL_32B.en.md)

`Qwen2.5-Coder-32B-Instruct`로 영어 3건, 독일어 3건, 스페인어 2건, 중국어 2건을 실행했습니다. **10건 모두 `status=ok`, `finish_reason=stop`으로 완료됐으며, 응답 파싱 실패와 토큰 잘림은 없었습니다.**

The 32B model completed live inference for the same 10 items as the 7B model. All responses parsed successfully. Input message lists were identical for every item. Translation changes include both improvements and regressions; no aggregate quality score was assigned.

## 비교 조건

| 항목 | 설정 |
|---|---|
| 실행일 | 2026-11-19 |
| 모델 | `Qwen/Qwen2.5-Coder-32B-Instruct` |
| 모델 리비전 | `381fc969f78efac66bc87ff7ddeadb7e73c218a7` |
| GPU | NVIDIA A100-SXM4-80GB × 1 |
| Python | 3.12.2 |
| vLLM / PyTorch | 0.15.1 / 2.9.1 |
| Transformers / huggingface-hub | 4.57.6 / 0.36.2 |
| CUDA 런타임 / 드라이버 | 12.8 / 565.57.01 |
| 모델 정밀도 | bfloat16, 비양자화 |
| 문맥 / HTML 정제 | 좌우 각각 128문자 / 사용 |
| 생성 설정 | temperature 0.01, 최대 32토큰, seed 0 |
| 배치 / prefix caching | 32 / 사용 |
| 최대 모델 길이 / tensor parallel | 15,872 / 1 |

7B와 입력 메시지가 10건 모두 정확히 일치합니다. 실행 옵션은 모델명과 GPU 메모리 사용 비율만 다릅니다. GPU 메모리 비율은 모델 크기에 맞춰 7B에서 0.45, 32B에서 0.9로 설정했습니다. 32B 가중치 로딩에 사용된 GPU 메모리는 약 61GiB입니다.

## 실행 명령

패키지 최상위 폴더에서 실행합니다. GPU 번호와 모델 캐시 위치는 실행 환경에 맞게 설정합니다.

```bash
python -m pip install -r requirements-vllm.txt
python run.py translate --backend vllm --model Qwen/Qwen2.5-Coder-32B-Instruct \
  --pair all --sample-ids en2ko:1 en2ko:2 en2ko:3 de2ko:1 de2ko:2 de2ko:3 \
  es2ko:1 es2ko:2 zh2ko:1 zh2ko:2 --window 128 --temperature 0.01 \
  --max-tokens 32 --seed 0 --gpu-memory-utilization 0.9 \
  --output outputs/local_model_32b_10.jsonl
```

위 리비전의 캐시된 모델을 오프라인으로 로딩했습니다.

## 항목별 비교

| ID | 원문 | 7B 번역 | 32B 번역 | 수작업 참조 |
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

## 결과 검토

- `Stoffart`는 7B의 `재료`에서 32B의 `소재`로 바뀌어 의류 문맥에 더 적절한 표현이 됐습니다.
- 중국어 브랜드 스토어 문구는 `오픈 스토어`에서 `공식 스토어`로 바뀌었습니다.
- `closure type`은 주변 값인 `드로스트링` 대신 `단추 유형`을 생성했지만, 참조의 `잠금 장치`와는 차이가 있습니다.
- `Enviar a → 보내기`, `Helpful / Nützlich → 유용함`은 두 모델에서 같았습니다. 배송·도움 평가 UI의 문맥을 충분히 반영하지 못한 결과입니다.
- `面料类型`은 7B의 `재질`에서 32B의 `면단 종류`로 바뀌어 표현이 더 어색해졌습니다.

10건 중 번역 문자열이 같은 항목은 6개, 다른 항목은 4개입니다. 이는 문자열 비교이며 품질 점수가 아닙니다. 이 소규모 실행에는 별도의 LLM 채점이나 사람 평가 점수를 부여하지 않았습니다.

[32B 원시 결과](local_model_32b_10.jsonl) · [32B 실행 메타데이터](local_model_32b_10.meta.json) · [7B 실행 결과](LOCAL_MODEL.md)
