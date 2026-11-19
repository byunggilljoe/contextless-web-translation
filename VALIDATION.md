# 검증

[English](VALIDATION.en.md)

환경: Python 3.12, Beautiful Soup 4.12.3, soupsieve 2.6.

## 데이터 검사

| 항목 | 결과 |
|---|---|
| 번역 항목 수·고유 ID | 1,018개 확인 |
| HTML 경로·SHA-256 | 174개 확인 |
| 정제 후 대상 매칭 | 1,010개 성공, 8개 실패 |
| 빈 수작업 참조 번역 | 1개 |

항목별 결과는 [VALIDATION.json](VALIDATION.json), 알려진 데이터 문제는 [DATASET.md](DATASET.md)에 있습니다.

```bash
python validate.py
python validate.py --check-targets
```

첫 명령은 데이터 구성과 `SHA256SUMS`를 검사합니다. 두 번째 명령은 전체 항목의 HTML 정제·대상 매칭도 검사합니다.

## 실행 테스트

```bash
python -m unittest discover -s tests -v
```

통합 테스트 6개가 통과했습니다. 로컬 HTTP 모의 서버로 번역·채점 결과 저장, 인증 실패·형식 오류·토큰 잘림 처리, 빈 참조 처리, 선택 노드 반영, 파일 덮어쓰기 방지를 검사합니다. 무문맥 조건의 입력에 HTML이나 참조 번역이 포함되지 않는지도 검사합니다.

실제 HTML로 프롬프트 생성, 정제 유무와 문맥 크기 변경, 선택 노드 ID 생성도 확인했습니다.

## 실제 로컬 추론

Qwen2.5-Coder-7B-Instruct와 32B-Instruct를 각각 NVIDIA A100-SXM4-80GB 1개에서 실행했습니다. 영어 3건, 독일어 3건, 스페인어 2건, 중국어 2건으로 구성된 동일한 10개 항목을 사용했습니다. 두 모델 모두 10건씩 `status=ok`, `finish_reason=stop`으로 완료했습니다. 대상 매칭 실패, 응답 파싱 실패, 생성 토큰 잘림은 없었습니다.

사용 환경: Python 3.12.2, vLLM 0.15.1, PyTorch 2.9.1, Transformers 4.57.6, huggingface-hub 0.36.2. `requirements-vllm.txt`에 확인한 실행 버전을 고정했습니다.

항목별 원문·번역·수작업 참조, 실행 명령과 설정은 [7B·32B 비교 결과](validation/LOCAL_MODEL_32B.md)에 있습니다. 일부 문맥 선택 오류가 관찰되었으므로 번역 품질은 별도 평가가 필요합니다.

32B 모델로 README의 5개 항목을 `--mode contextless`에서도 실제 실행했습니다. 5건 모두 `status=ok`, `finish_reason=stop`으로 완료했습니다. contextful은 기존 `--mode qa --window 128` 결과이며, 두 조건의 모델·리비전·temperature·최대 생성 토큰·seed는 같습니다. 두 조건은 각 모드의 프롬프트 형식을 사용합니다. [5개 예시 비교](examples/comparison_5.json)

## 재현 범위

외부 상용 API 호출과 논문의 품질·속도 표 전체는 검증하지 않았습니다. `tests/`의 통합 테스트는 모의 서버를 사용하며, `validation/`은 동일한 10개 항목에 대한 모델별 실제 GPU 추론 결과를 담고 있습니다.

실험 결과를 비교할 때는 모델·리비전, GPU·서버 환경, 라이브러리 버전, 생성 설정을 함께 기록합니다. 실행 설정과 설치된 주요 패키지 버전은 결과의 `.meta.json`에서 확인할 수 있습니다.
