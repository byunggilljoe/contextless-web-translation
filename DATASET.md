# 데이터셋 구조

문맥이 누락된 웹 마크업 텍스트의 한국어 번역을 평가하기 위한 데이터셋입니다. **번역 항목 1,018개와 HTML 스냅샷 174개**로 구성됩니다. [English](DATASET.en.md)

| 원문 → 한국어 | 번역 항목 | HTML 스냅샷 |
|---|---:|---:|
| 영어 | 524 | 73 |
| 독일어 | 194 | 39 |
| 스페인어 | 200 | 36 |
| 중국어 | 100 | 26 |
| 합계 | **1,018** | **174** |

174개는 언어별로 저장된 HTML 수입니다. 동일 사이트의 언어별 페이지가 포함되어 있으며, 정확한 URL 문자열 기준으로 중복을 제거하면 74개입니다.

## 폴더 구조

```text
data/
├── samples.jsonl
├── pages.json
├── html/
│   ├── en/{페이지번호}/original.html
│   ├── de/{페이지번호}/original.html
│   ├── es/{페이지번호}/original.html
│   └── zh/{페이지번호}/original.html
├── annotations/
│   └── webpage_chrome_manual_{en,de,es,zh}.csv
├── annotation_differences.json
└── provenance.json
```

`samples.jsonl`의 한 줄이 하나의 번역 대상입니다. **하나의 HTML에 여러 번역 대상이 연결될 수 있습니다.** `page_id`로 `pages.json`의 페이지 정보를 찾고, `html_path`로 HTML을 읽습니다. 페이지 번호는 연속적이지 않을 수 있습니다.

## 파일별 역할

아래 경로는 `data/` 기준입니다.

| 파일 | 내용 |
|---|---|
| `samples.jsonl` | 번역 대상, 언어쌍, 페이지 경로, Chrome 번역, 수작업 참조 번역 |
| `pages.json` | 페이지 ID, 출처 URL, HTML 경로, SHA-256 해시 |
| `html/{en,de,es,zh}/{페이지번호}/original.html` | 정제 전 HTML 스냅샷 |
| `annotations/*.csv` | 주석 CSV 4개. 대체 한국어 참조 번역과 다른 언어의 Chrome 번역도 포함 |
| `annotation_differences.json` | 주석 CSV와 실험용 메타데이터 사이의 차이 |
| `provenance.json` | 데이터·코드 출처, 기준 리비전, 파일 해시 |

## 번역 항목의 필드

`samples.jsonl`은 UTF-8 JSONL 형식입니다.

| 필드 | 자료형 | 의미 |
|---|---|---|
| `id` | 문자열 | 전체 항목의 고유 ID. 예: `en2ko:1` |
| `sample_id` | 정수 | 언어쌍 내에서 1부터 시작하는 항목 ID |
| `language_pair` | 문자열 | `en2ko`, `de2ko`, `es2ko`, `zh2ko` |
| `source_language` | 문자열 | 원문 언어 코드: `en`, `de`, `es`, `zh` |
| `target_language` | 문자열 | 번역 언어 코드: `ko` |
| `page_id` | 문자열 | 언어별 HTML ID. 예: `en/0` |
| `html_path` | 문자열 | 패키지 최상위 폴더 기준 HTML 상대 경로 |
| `url` | 문자열 | 수집한 페이지의 출처 URL |
| `original` | 문자열 | 번역 대상 텍스트. 저장된 공백을 포함 |
| `chrome_translation` | 문자열 | 데이터 수집 당시 Chrome 번역 |
| `reference_translation` | 문자열 | 대표 수작업 한국어 참조 번역. 1개 항목은 빈 문자열 |
| `references` | 문자열 배열 | 대표·대체 수작업 참조 번역. 대표 번역이 실험 메타데이터와 일치할 때 CSV의 대체 번역도 포함 |
| `exclude_from_user_survey` | 불리언 | 사람 평가 제외 여부. 61개 항목에 `true` 지정 |
| `failure_type` | 문자열 | 주석에 기록된 번역 오류 유형 |

`id`는 전체에서 고유하지만 `sample_id`는 언어쌍별로 중복될 수 있습니다. 기본 실험에는 1,018개 항목이 모두 포함됩니다. `--survey-only`는 사람 평가 제외 표시가 있는 61개를 제외합니다.

학습·검증·시험 분할은 지정되어 있지 않습니다. 새로 분할할 때는 같은 페이지와 그 언어별 항목이 서로 다른 분할에 섞이지 않도록 URL·사이트 단위로 묶습니다.

## 항목 예시와 읽는 방법

첫 번째 항목(`en2ko:1`)의 주요 필드입니다.

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

다음 예제는 패키지 최상위 폴더에서 실행합니다.

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

실험 실행기는 HTML 경로를 코드 파일이 있는 폴더 기준으로 해석합니다. 위 예제는 현재 디렉터리 기준이므로 코드와 `data/`가 있는 프로젝트 최상위 폴더에서 실행합니다.

## 주석 차이와 누락값

`samples.jsonl`은 실험용 메타데이터를 기준으로 합니다. 주석 CSV와 차이가 있는 항목은 다음 네 개입니다.

| ID | 차이 |
|---|---|
| `en2ko:8` | 실험용 Chrome 번역 끝에 공백이 있음 |
| `en2ko:67` | 실험 메타데이터의 수작업 참조 번역이 비어 있고 Chrome 번역도 CSV와 다름 |
| `de2ko:5` | 실험용 원문 끝에 공백이 있음: `Mehr ` |
| `de2ko:83` | 실험용 원문 끝에 공백이 있음: `Sie sparen ` |

양쪽 값은 `annotation_differences.json`에 기록되어 있습니다. `en2ko:67`의 수작업 참조 번역을 평가하면 `missing_translation`으로 기록됩니다.

## HTML에서 번역 대상 찾기

`process_html.py`는 HTML을 정제하고 대상 문자열에 태그를 삽입한 뒤 첫 번째 일치 위치(`TRAN_0`) 주변의 문자 문맥을 추출합니다. 대소문자를 구분하지 않으며 HTML 텍스트의 콜론 제거 등 정규화를 적용합니다. 저장된 대상 문자열의 양끝 공백은 제거하지 않습니다. 항목별 고유 XPath는 제공하지 않습니다.

Beautiful Soup 4.12.3에서 정제 후 1,010개 항목이 매칭됩니다. 아래 8개는 매칭되지 않습니다.

| ID | 번역 대상 |
|---|---|
| `en2ko:220` | `Subscribe` |
| `en2ko:337` | `Open positions` |
| `de2ko:5` | `Mehr ` |
| `de2ko:83` | `Sie sparen ` |
| `de2ko:146` | `Offene Stellen` |
| `de2ko:147` | `Büros` |
| `es2ko:169` | `Puestos disponibles` |
| `es2ko:170` | `Oficinas` |

`de2ko:5`와 `de2ko:83`은 원문 끝의 공백 때문에 매칭되지 않습니다. 매칭 실패 항목은 실행 결과에 `preprocessing_error`로 남습니다. [검사 결과](VALIDATION.json)를 보거나 `python validate.py --check-targets`로 확인할 수 있습니다.

## HTML 리소스

정제 유무 비교를 위해 수집 HTML을 제공합니다. 공개본에서는 31개 HTML에 남아 있던 웹사이트 토큰·식별자 형태의 값을 `REDACTED`로 치환했습니다. 번역 대상·참조 번역과 정제 후 좌우 512문자 이내의 벤치마크 문맥은 유지됩니다. 변경 전후 해시는 [변경 기록](data/public_redactions.json)에 있습니다. 페이지 스크립트와 내장 리소스가 포함되어 있으며, 실행기는 JavaScript 실행이나 페이지 접속 없이 HTML을 파싱합니다. HTML 렌더링용 외부 CSS·이미지·폰트·MHTML은 포함하지 않아 브라우저에서 보이는 모양은 수집 당시와 다를 수 있습니다. README의 번역 예시 스크린샷은 `examples/`에 있습니다.
