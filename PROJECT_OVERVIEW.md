# 코드 구조

[English](PROJECT_OVERVIEW.en.md)

HTML의 주변 문맥으로 번역 대상의 의미를 판별하고, 대상 텍스트의 번역만 생성합니다. 페이지 번역에서는 여러 노드가 공통 문맥을 공유하며 질문을 병렬 처리합니다.

## 처리 흐름

1. `data/samples.jsonl`에서 번역 대상과 HTML 경로를 읽습니다.
2. `process_html.py`가 HTML을 정제하고 대상 문자열에 `TRAN_0` 태그를 삽입합니다.
3. 대상 태그 좌우의 문자 문맥으로 Q&A 프롬프트를 구성합니다.
4. `run.py`가 직접 vLLM 또는 Chat Completions API로 번역 요청을 실행합니다.
5. 번역·모델 응답·상태를 JSONL에, 설정과 요약 통계를 `.meta.json`에 저장합니다.
6. `run.py score`가 정제 HTML 문맥과 번역문을 평가 모델에 전달하여 1–5점으로 채점합니다.

## 파일 구성

| 경로 | 역할 |
|---|---|
| `data/samples.jsonl` | 번역 대상, 언어쌍, Chrome 번역, 수작업 참조 번역 |
| `data/html/` | 언어별 HTML 스냅샷 |
| `data/pages.json` | 페이지 출처·경로·해시 |
| `data/annotations/` | 주석 CSV |
| `process_html.py` | HTML 정제, 텍스트 정규화, 태그 삽입, 문맥 추출 |
| `run.py` | 번역·평가 CLI와 API/vLLM 백엔드 |
| `translate_page.py` | 텍스트 노드 선택, 블록 문맥 구성, 병렬 번역, DOM 반영 |
| `validate.py` | 데이터 수·경로·해시·대상 매칭 검사 |
| `tests/` | 모의 API를 사용하는 통합 테스트 |
| `validation/` | 7B·32B 로컬 GPU 추론 결과·설정·검토 내역 |
| `reference/` | 품질·캐시·속도 실험의 참고 구현 |
| `maintainer/` | 데이터 추출과 ZIP 생성 도구 |

## 페이지 번역

`translate_page.py`는 정제 HTML의 텍스트 노드에 `T_0`, `T_1` 등의 ID를 붙입니다. `--block-size`개 노드가 HTML 문맥을 공유하고, `--ids`로 선택한 노드의 번역을 요청합니다. 각 번역문은 원본 DOM의 대응 텍스트 노드에 반영합니다. 태그와 속성은 파싱된 DOM 기준으로 유지됩니다.

대상 문자열 실험의 `--window`는 문자 수, 페이지 번역의 `--block-size`는 텍스트 노드 수입니다.
