# 작업 로그

날짜순 기록. 상세 분석/근거가 긴 항목은 별도 문서로 빼고 여기서는 링크 + 한 줄 요약만 남김.
새 세션은 [PROJECT_STATE.md](PROJECT_STATE.md)를 먼저 읽고, 여기서는 최근 항목 몇 개만 훑으면 됨.

---

## 2026-08-16 — 데이터 전처리 파이프라인 v1→v3

- AIHub 판례 데이터 구조 파악, `판례내용`/`판결요지` 등 필드 분석, chunking 전 토큰 수
  통계(평균 3,491 / 중앙값 2,465 / 90th 6,446 토큰) 확인
- `scripts/build_train_val_dataset.py`를 v1(page_content 가공) → v2(원본 그대로 추출) →
  v3(DB_data 물리 복사 + 사건종류명별 85:15 stratified split)로 변경
- 결과: `data/DB_data/*.json` 44,700건, `data/Train/train_query.json`,
  `data/Val/val_query.json` 생성
- 상세: [data_preprocessing_log.md](data_preprocessing_log.md)

## 2026-08-17 — 진행상태 추적 체계 도입 (PROJECT_STATE.md / log.md)

- **배경**: 서버를 그때그때 대여해서 작업하다 보니 에이전트 세션이 매번 새로 시작되고,
  그때마다 프로젝트 설명 + 진행상황 설명을 반복해야 하는 문제 발생
- **결정**: [PROJECT_STATE.md](PROJECT_STATE.md)(현재 상태/로드맵/다음 계획 스냅샷)와
  이 log.md(완료된 작업의 날짜별 기록)를 분리해서 운영. 계획 하나가 끝날 때마다 두 파일을
  같이 갱신하는 걸 규칙으로 정함 (PROJECT_STATE.md 6절 참고)
- **로드맵 검토**: 사용자가 작성한 5단계 로드맵(데이터→retriever→generation→backend→배포)을
  LLM Application Engineer 포트폴리오 관점에서 검토. 큰 축은 빠진 것 없음. Hybrid search,
  reranker, 평가 프레임워크(RAGAS 등) 선정, streaming 응답, 평가 하니스 재사용화, 가벼운
  데모 UI(Flutter보다 우선) 등을 보완 제안으로 추가 — 근거는 PROJECT_STATE.md 3~4절
- **v1 코드 정리**: `server/main.py`, `README.md`, `lib/main.dart`(Flutter 앱) 등은 이번에
  새로 시작하기 전의 이전 시도(v1)였음을 확인. 폐기하고 처음부터 재시작 중이며, v1은
  참고 자산으로만 PROJECT_STATE.md 2절에 남겨둠 (v1을 왜 버렸는지는 아직 미기록 — TODO)
- **확인된 현재 상태**: `server/model_test.py`(Qwen3-Embedding-0.6B / BGE-M3 / KURE-v1
  dense retrieval 비교) 작성은 되어 있으나 `cache_embeddings/` 결과 없음 → 아직 미실행
  상태로 진행 중

## 2026-08-17 — GitHub 저장소 Chatbot-v2로 이전, v1 산출물 삭제

- **저장소 이전**: 기존 `minjunmj/Rag_Chatbot`(Public)은 그대로 두고, 새 저장소
  `minjunmj/Chatbot-v2`(Public)를 만들어 기존 커밋 히스토리(A~"data", 총 14개)를 그대로
  push. 로컬 `origin`도 새 저장소로 전환. V1 종료 지점에는 `v1-final` 태그를 남겨서
  "여기까지가 v1"을 명확히 표시함 (커밋 `1aeb4d3`)
- **git 계정 정보 정리**: 커밋 작성자가 macOS 계정 정보로 자동 채워져 있던 것을 확인하고
  `user.name=박민준`, `user.email=dhfkzlfj@naver.com`으로 전역 설정. 이미 push된 첫 V2
  커밋도 amend + force-push로 작성자 정보를 맞춤
- **v1 산출물 삭제**: 처음부터 새로 시작하기로 하면서 v1의 완성된 결과물을 저장소에서
  제거 — Flutter 모바일 앱 전체(android/, lib/, web/, windows/, test/, pubspec*,
  analysis_options.yaml, .metadata), Docker 배포 설정(Dockerfile, docker-compose.yml,
  .dockerignore), FastAPI 서버 코드(main.py, acc_test.py), v1 의존성 목록
  (server/requirements.txt), v1 소개 README.md
  - 코드 자체는 사라지지 않고 `v1-final` 태그에 남아있음 (필요하면
    `git checkout v1-final -- <path>`로 복원 가능)
  - 유지한 것: `data/`, `docs/`, `scripts/`(데이터 파이프라인), `.tools/`(AIHub 다운로드
    도구), `server/model_test.py`(진행 중인 임베딩 비교) — v2에서도 계속 쓰는 것들
- 상세: PROJECT_STATE.md 2절 갱신
