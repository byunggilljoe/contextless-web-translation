# 로컬 모델 실행 결과 / Local model inference

[English](LOCAL_MODEL.en.md)

[동일 항목의 32B 비교 결과](LOCAL_MODEL_32B.md)

Qwen2.5-Coder-7B-Instruct로 영어 3건, 독일어 3건, 스페인어 2건, 중국어 2건을 실행했습니다. 각 언어쌍의 앞쪽 항목을 선택했습니다. **10건 모두 정상 응답·파싱으로 완료되었으며, 생성 토큰 잘림은 없었습니다.**

Live GPU inference completed for 10 items across all four source languages. All responses parsed successfully with `finish_reason=stop`. These results verify execution; several translations contain contextual errors.

## 실행 환경

| 항목 | 값 |
|---|---|
| 실행일 | 2026-11-19 |
| 모델 | `Qwen/Qwen2.5-Coder-7B-Instruct` |
| 모델 리비전 | `c03e6d358207e414f1eca0bb1891e29f1db0e242` |
| GPU | NVIDIA A100-SXM4-80GB × 1 |
| Python | 3.12.2 |
| vLLM / PyTorch | 0.15.1 / 2.9.1 |
| Transformers / huggingface-hub | 4.57.6 / 0.36.2 |
| CUDA 런타임 / 드라이버 | 12.8 / 565.57.01 |
| HTML 정제 / 문맥 | 정제 사용 / 좌우 각 128문자 |
| 생성 설정 | temperature 0.01, 최대 32토큰, seed 0 |
| 모델 설정 | bfloat16, prefix caching 사용, tensor parallel 1 |

## 실행 명령

패키지 최상위 폴더에서 실행합니다. GPU 번호와 모델 캐시 위치는 실행 환경에 맞게 설정합니다.

```bash
python -m pip install -r requirements-vllm.txt
python run.py translate --backend vllm --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --pair all --sample-ids en2ko:1 en2ko:2 en2ko:3 de2ko:1 de2ko:2 de2ko:3 \
  es2ko:1 es2ko:2 zh2ko:1 zh2ko:2 --window 128 --temperature 0.01 \
  --max-tokens 32 --seed 0 --gpu-memory-utilization 0.45 \
  --output outputs/local_model_10.jsonl
```

위 리비전의 캐시된 모델을 오프라인으로 로딩하여 실행했습니다. 실행 시 출력되는 모델 리비전과 설치 버전을 확인하여 비교합니다.

## 항목별 결과

| ID | 원문 | 모델 번역 | 수작업 참조 |
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

`dismiss → 닫기`, `Herkunft → 원산지`, `Devoluciones → 반품`은 참조 번역과 일치합니다. `Helpful → 유용함`, `面料类型 → 재질`처럼 참조와 표현이 다른 결과도 있습니다.

`closure type`은 항목명 대신 주변 값인 `드로스트링`을 생성했고, `Enviar a`는 배송 문맥에서 `보내기`로 번역했습니다. 중국어 `品牌旗舰店`에 해당하는 표현도 `공식스토어` 대신 `오픈 스토어`로 생성했습니다. 이 결과에는 별도의 LLM 채점이나 사람 평가 점수를 부여하지 않았습니다.

전체 프롬프트·원시 응답·토큰 사용량은 [결과 JSONL](local_model_10.jsonl), 설정·환경·요약 통계는 [실행 메타데이터](local_model_10.meta.json)에 있습니다.
