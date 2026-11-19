# 배포 관리

[English](README.en.md)

## 데이터 추출

연구 저장소에서 데이터와 참고 구현을 추출합니다.

```bash
python maintainer/export_from_project.py --project-root /path/to/research-project
```

기준 메타데이터는 실험용 JSON이며, 주석 CSV와의 차이는 `data/annotation_differences.json`에 기록합니다. `data/provenance.json`에는 기준 Git 리비전, 실제 파일 해시, 사설 API 호스트 치환 내역을 기록합니다.

## ZIP 생성

문서·코드·데이터 변경 후 검증하고 새 ZIP을 만듭니다. 기존 ZIP과 다른 출력명을 지정합니다.

```bash
python validate.py
python maintainer/package.py --output ../reproduction_bundle_new.zip
```

생성기는 `SHA256SUMS`를 갱신하고 HTML을 포함합니다. `outputs/`, 가상환경, `.env`, 캐시, Git 이력은 제외합니다.

## GitHub 및 이용 조건

`.gitignore`는 `data/html/`을 제외합니다. 코드와 주석을 Git으로 관리하고 HTML은 데이터 ZIP으로 배포할 수 있습니다. HTML을 Git에 포함하려면 해당 규칙을 변경합니다.

배포 라이선스는 미지정입니다. 공개 시 코드와 수집 웹페이지 데이터의 이용 조건을 정해 안내합니다. HTML에는 스크립트와 내장 리소스가 포함되며, 출처는 `data/pages.json`에서 확인할 수 있습니다.
