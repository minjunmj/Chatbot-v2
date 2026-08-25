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

## 2026-08-18 — 임베딩 모델 비교 실험 결과 (model_test.py), KURE-v1 선정

- **후보 선정 기준** (왜 이 3개인가):
  - **Qwen3-Embedding-0.6B** — 최신 범용 multilingual dense 임베딩 모델이 이 도메인(한국어
    법률 판례)에서 실제로 얼마나 통하는지 확인하는 용도. "최신 모델이니까 당연히 잘하겠지"라는
    가정을 그냥 믿지 않고 직접 검증해보는 후보.
  - **BGE-M3** — 이미 여러 벤치마크에서 검증된 강력한 범용 baseline. 게다가 dense뿐 아니라
    sparse/multi-vector도 한 모델에서 지원하는 구조라, 나중에 hybrid retrieval(BM25 등과
    결합, Phase 2 로드맵 ➕ 항목)로 확장할 때도 그대로 재사용할 수 있다는 점까지 고려해서 넣음.
  - **KURE-v1** — BGE-M3를 한국어로 추가 튜닝한 모델. "한국어에 특화해서 학습한 retrieval
    모델이 실제 한국 판례 검색에서도 범용 모델보다 더 잘하는가"를 직접 대조해보기 위한 핵심
    후보 — 이 실험에서 사실상 BGE-M3의 대조군 역할.
- **조건**: chunking 없이 판례내용 전문을 통으로 encode (max_seq_length 초과분은 모델
  기본 truncation), dense retrieval만 사용(BM25/rerank 없음), val_query.json(쿼리당 정답
  case_id 1개) 기준 Recall@1/5/10/20·MRR@10

| model | recall@1 | recall@5 | recall@10 | recall@20 | mrr@10 | encode_time(s) |
|---|---|---|---|---|---|---|
| qwen3-embedding-0.6b | 0.3409 | 0.5416 | 0.6238 | 0.6984 | 0.4276 | 0.2 (캐시 로드, 아래 참고) |
| bge-m3 | 0.3114 | 0.4982 | 0.5736 | 0.6424 | 0.3914 | 2804.4 |
| kure-v1 | **0.4012** | **0.6192** | **0.6941** | **0.7607** | **0.4941** | 2825.3 |

- **결과**: KURE-v1(BGE-M3 한국어 튜닝)이 전 지표에서 1위. BGE-M3 대비 recall@1 +9.0%p,
  mrr@10 +0.10 — 한국어 법률 도메인에서 언어별 파인튜닝이 뚜렷하게 유효함을 확인.
- **선정**: 임베딩 모델은 **KURE-v1**로 결정 (기준선 확보).
- **참고(주의)**: qwen3-embedding-0.6b의 `corpus_encode_time_sec: 0.2`는 실제 인코딩 시간이
  아님 — `cache_embeddings/qwen3-embedding-0.6b_corpus.npy`가 전날(08-17) 실행에서 이미
  생성돼 있어 이번 실행은 캐시를 `np.load`만 한 것(0.2초=로드 시간). bge-m3/kure-v1은 이번에
  새로 인코딩해서 각각 ~47분 소요. recall/mrr 지표 자체는 캐시 여부와 무관하게 유효하지만
  "인코딩 속도" 비교 근거로는 이 표를 쓰지 말 것.
- **다음**: chunking 전략이 아직 미결정 상태라, 최종 확정 전에 chunk 기반 retrieval 효과를
  추가로 확인하기 위해 `server/jhgan.py`(jhgan/ko-sbert-sts, chunk_size=300자) 실험 착수.

## 2026-08-18 — chunk 기반 retrieval 실험 결과 (jhgan.py)

- **조건**: jhgan/ko-sbert-sts, chunk_size=200자(글자 단위, 겹침 없음, 문서 823,763개 chunk로
  분할), 문서 단위 dedupe는 chunk 랭킹에서 score-level max(같은 case_id는 최고 점수 chunk만
  유지) — chunk_size=300으로 시작했다가, 이 모델 토크나이저 기준 300자가 128토큰 한도를 넘어
  청크 내부가 다시 잘리는 문제가 확인돼 200자(96토큰)로 낮춰서 재실행함

| model | recall@1 | recall@5 | recall@10 | recall@20 | mrr@10 |
|---|---|---|---|---|---|
| jhgan-ko-sbert-sts_chunk200 | **0.4082** | **0.6457** | **0.7227** | **0.7919** | **0.5084** |

- **표면적 결과**: no-chunk 1위였던 KURE-v1(recall@1 0.4012, mrr@10 0.4941)보다도 전 지표에서
  높게 나옴.
- **주의 — 이 비교는 confound(교란)가 있음**: 이 실험은 "모델"과 "chunking 여부"를 동시에
  바꿨음(jhgan+chunk vs KURE-v1+no-chunk). jhgan은 STS 전용으로 튜닝된 상대적으로 작은
  klue-bert-base 계열 모델이라, retrieval 전용으로 학습된 훨씬 큰 KURE-v1보다 모델 자체 성능은
  낮을 것으로 예상됨. 그런데도 결과가 더 높게 나온 건 "jhgan이 더 좋은 모델"이라서가 아니라
  "chunking(긴 문서를 통으로 인코딩할 때 생기는 정보 희석을 chunk 단위로 좁혀서 완화)"이
  원인일 가능성이 높음. 즉 이 표만으로는 "jhgan이 이겼다"가 아니라 "chunking 자체가 유효할
  수 있다"까지만 결론 내릴 수 있음.
- **다음**: 모델을 KURE-v1로 고정한 채 chunking 여부만 바꿔서 isolate하는 실험이 필요
  (KURE-v1 + chunk_size=200 + 동일 max-pooling dedupe) — 아직 착수 전.

## 2026-08-18 — BGE-M3 hybrid(dense+sparse) 실험 결과 (bge_hybrid.py)

- **조건**: chunking 없음(model_test.py와 동일 조건). BGE-M3의 dense 벡터와 sparse(lexical
  weight, 모델 자체 학습 — BM25 아님) 벡터를 RRF(Reciprocal Rank Fusion, k=60)로 결합.
  dense_only는 model_test.py의 bge-m3 결과 재현 여부를 보는 sanity check도 겸함.

| model | recall@1 | recall@5 | recall@10 | recall@20 | mrr@10 |
|---|---|---|---|---|---|
| dense_only | 0.3115 | 0.4984 | 0.5736 | 0.6426 | 0.3915 |
| sparse_only | 0.3026 | 0.5195 | 0.6125 | 0.7016 | 0.3965 |
| hybrid_rrf | **0.3707** | **0.5957** | **0.6758** | **0.7449** | **0.4666** |

- **sanity check 통과**: dense_only 수치가 model_test.py의 bge-m3 결과(recall@1 0.3114,
  mrr@10 0.3914)와 거의 동일 — FlagEmbedding 기반 dense 출력이 sentence-transformers 결과와
  일치함을 확인.
- **sparse가 예상보다 강함**: sparse_only가 recall@1만 빼면 dense_only를 전반적으로 앞섬
  (recall@20: 0.7016 vs 0.6426). 법률 판례는 법조문 번호·사건번호·고유명사처럼 키워드 매칭이
  중요한 도메인이라는 가설과 부합.
- **hybrid 효과 확인**: hybrid_rrf가 dense_only/sparse_only 각각보다 전 지표에서 뚜렷하게
  개선 (recall@1 +5.9~6.8%p, mrr@10 +0.070~0.075) — dense+sparse 결합이 이 도메인에서
  확실히 유효함.
- **그러나 KURE-v1(no-chunk)엔 아직 못 미침**: hybrid_rrf(recall@1 0.3707, mrr@10 0.4666)가
  KURE-v1 no-chunk 단독(recall@1 0.4012, mrr@10 0.4941)보다 낮음 → 이 데이터셋에서는
  "hybrid로 범용 모델을 보강하는 것"보다 "한국어 도메인 특화 튜닝(KURE-v1)"이 더 크게
  기여한다는 뜻. (KURE-v1은 sparse 출력을 지원하지 않아 동일한 dense+sparse hybrid를 그대로
  적용할 수 없음 — 필요하면 BM25와의 hybrid를 별도로 검토해야 함)

## 2026-08-18 — KURE-v1 chunk 크기별 isolate 실험 결과 (kure_chunk.py), confound 해소

- **목적**: jhgan.py 결과(chunk_size=200, recall@1 0.4082)가 KURE-v1 no-chunk baseline(recall@1
  0.4012)을 앞섰지만 "모델"과 "chunking 여부"를 동시에 바꿔서 원인을 특정할 수 없었음(confound).
  모델을 KURE-v1로 고정하고 chunk 크기(200/500/1000자, 겹침 없음)만 바꿔 isolate.
- **조건**: 문서 단위 dedupe는 jhgan.py와 동일한 score-level max-pooling. KURE-v1
  max_seq_length=8192라 세 크기 모두 truncation 걱정 없음.

| model | recall@1 | recall@5 | recall@10 | recall@20 | mrr@10 | num_chunks | corpus_encode_time_sec |
|---|---|---|---|---|---|---|---|
| kure-v1_chunk200 | **0.5541** | **0.8048** | **0.8734** | **0.9192** | **0.661** | 823,763 | 7367.7 |
| kure-v1_chunk500 | 0.4995 | 0.7424 | 0.8194 | 0.878 | 0.6044 | 343,033 | 3537.2 |
| kure-v1_chunk1000 | 0.4402 | 0.6659 | 0.7522 | 0.8209 | 0.5387 | 182,533 | 2732.9 |

- **confound 해소됨**: 모델을 KURE-v1로 고정한 상태에서도 chunk_size=200이 no-chunk
  baseline(recall@1 0.4012, mrr@10 0.4941)을 큰 폭으로 앞섬(+0.153/+0.167p) — jhgan 실험에서
  본 성능 향상이 "jhgan이라는 모델의 우연한 궁합"이 아니라 **chunking 자체가 지배적 요인**임이
  확인됨. 게다가 kure-v1_chunk200(0.5541)이 같은 chunk 크기의 jhgan_chunk200(0.4082)보다도
  크게 앞서서, "도메인 특화 모델(KURE-v1) + chunking"이 둘 다 독립적으로 기여함을 보여줌.
- **chunk 크기와 성능은 단조관계**: 200 > 500 > 1000 순으로 전 지표에서 일관되게 감소 — 짧게
  자를수록 검색 정확도가 좋아짐. 셋 다 no-chunk보다는 위.
- **비용 트레이드오프**: chunk200은 성능이 가장 좋지만 문서당 chunk 수가 많아져(823,763개,
  no-chunk 대비 약 18배) corpus encode 시간이 7367.7초(≈2시간)로 chunk1000(2732.9초,
  ≈46분) 대비 2.7배. Vector DB 크기·인덱싱/검색 latency에도 비례해서 영향 줄 것으로 예상.
- **다음**: chunk_size 최종 결정 — 정확도만 보면 200이 압도적이지만, 실서비스 latency/DB
  용량까지 고려해서 최종 크기를 정할지, 아니면 성능 우선으로 200을 그대로 채택할지 결정 필요.
  결정 후 Vector DB 구축(FAISS vs pgvector)으로 진행.

## 2026-08-19 — Vector DB를 pgvector로 결정, DB 구축 파이프라인 준비

- **FAISS vs pgvector 검토 후 pgvector 선택**: 법률 판례 검색은 "법원명/사건종류명/선고일자로
  필터 + 의미 검색"을 같이 쓸 가능성이 높은데, FAISS는 메타데이터 필터를 자체 지원하지 않아
  직접 구현해야 함. pgvector는 SQL `WHERE` + `ORDER BY embedding <=> ...`로 한 쿼리에서 결합
  가능. 또한 PROJECT_STATE.md 3절에 이미 있던 "Phase 4에서 PostgreSQL 쓸 계획"과도 인프라가
  통합됨. chunk200 기준 82만 벡터 규모는 FAISS만의 성능 우위가 필요한 스케일이 아니라고 판단.
- **디스크 정리**: 선정 안 된 모델(qwen3-embedding-0.6b, bge-m3, jhgan-ko-sbert-sts)의 임베딩
  캐시(.npy/.npz)와 HF 모델 가중치를 삭제해 디스크 여유 8G → 20G로 확보. KURE-v1 모델과
  chunk200/500/1000 임베딩 캐시는 chunk 크기 결정에 아직 필요해서 보존.
- **train 데이터 미사용 확인**: model_test.py/jhgan.py/bge_hybrid.py/kure_chunk.py 4개 실험
  스크립트 전부 `val_query.json`만 참조, `train_query.json`은 지금까지 한 번도 사용 안 됨
  (grep으로 확인).
- **Postgres 16 + pgvector 0.6.0 설치**: 이 인스턴스(Vast 컨테이너, systemd 없음)에 apt로
  설치 후 supervisor 서비스(`postgres`)로 등록해 foreground 프로세스로 관리(`postgres` OS
  유저로 실행, `/opt/supervisor-scripts/postgres.sh`). DB `lexchatbot` 생성, `vector` extension
  활성화. 접속정보는 `server/.env`의 `DATABASE_URL`.
- **적재 파이프라인 작성** (`server/db/`):
  - `schema.sql` — `chunks_200`/`chunks_500`/`chunks_1000` 테이블. 필드는 사건번호/사건명/
    법원명/선고일자/사건종류명(metadata) + chunk_text + embedding(VECTOR(1024))만 포함하기로
    결정 — 판시사항/판결요지/판결유형/선고 등은 제외(판례내용 chunk만 검색 대상으로 확정,
    data_preprocessing_log.md 3절의 "보류 중"이었던 context/metadata 구분이 이 범위로 일부 정리됨)
  - `build_vector_db.py` — kure_chunk.py가 저장한 임베딩 캐시(.npy)는 벡터만 저장하고 chunk
    텍스트/사건번호는 저장 안 해서, 원본 JSON을 kure_chunk.py와 **동일한 순서/로직**으로
    재처리해 chunk 텍스트+metadata를 재생성한 뒤 캐시된 벡터와 인덱스를 맞춰 매칭. 재임베딩
    없이(GPU 불필요) COPY로 벌크 적재 후 HNSW 인덱스 생성.
  - `test_val_pgvector.py` — val_query.json으로 (1) recall@k/mrr@10이 kure_chunk.py의 numpy
    exact-search 결과와 근접한지 sanity check, (2) 쿼리당 검색 latency(mean/p50/p95) 측정.
    chunk 크기 최종 결정에 쓸 latency 실측 자료.
- **chunk1000으로 파이프라인 검증**: 182,533행 적재(kure_chunk.py 실험 때의 chunk 개수와
  정확히 일치 — 매핑 정상 확인). 적재 126.8초, HNSW 인덱스 빌드 754.3초, 테이블+인덱스 용량
  2,662MB (사전 어림 계산 ~2.2GB와 비슷한 수준).
- **다음**: chunk200/500도 같은 스크립트로 적재(사용자가 직접 실행) → 세 크기 모두
  test_val_pgvector.py로 latency 측정 → 정확도/용량/latency 종합해서 chunk 크기 최종 결정.

## 2026-08-19 — pgvector 실측 검증 결과 (chunk200 vs chunk1000), chunk200 쪽으로 기움

- **chunk200 적재 중 디스크 위기**: 인덱스 빌드 도중 여유 공간이 1.2GB까지 떨어짐(예상보다
  chunk200 실제 용량이 훨씬 큼). 이미 DB에 적재가 끝나 더는 필요 없어진
  `kure-v1_chunk200_corpus.npy`(1.6G) 캐시를 삭제해 즉시 여유 확보, 빌드는 문제없이 완료됨.
- **test_val_pgvector.py 초기 버그 발견+수정**: `DISTINCT ON (case_no)` 사용 시 Postgres 문법상
  `ORDER BY`가 case_no로 시작해야 하는데, 그러면 LIMIT이 벡터 거리 순이 아니라 사건번호(문자열)
  순으로 잘려서 사실상 무작위에 가까운 후보군을 가져오는 버그였음(첫 실행 때 chunks_200
  recall@1이 0.0034로 나와서 발견). CTE로 "① HNSW로 진짜 최근접 500개를 먼저 뽑고 ② 그 안에서만
  case_no dedupe"하도록 수정 후 재검증.
- **pgvector(HNSW) 실측 결과** — kure_chunk.py의 numpy exact-search 대비 전 지표에서 1~2%p
  낮게 나옴(ANN 근사 탐색이므로 정상적인 손실):

| chunk_size | recall@1 | recall@5 | recall@10 | recall@20 | mrr@10 | latency mean | latency p99 | DB 용량(테이블+인덱스) |
|---|---|---|---|---|---|---|---|---|
| 200 (pgvector) | 0.5433 | 0.7905 | 0.8577 | 0.9021 | 0.6487 | 7.73ms | 12.63ms | 11 GB |
| 1000 (pgvector) | 0.4293 | 0.6503 | 0.7341 | 0.8019 | 0.5259 | 6.78ms | 10.66ms | 2.66 GB |
| (참고) 200 exact | 0.5541 | 0.8048 | 0.8734 | 0.9192 | 0.661 | - | - | - |
| (참고) 1000 exact | 0.4402 | 0.6659 | 0.7522 | 0.8209 | 0.5387 | - | - | - |

- **트레이드오프 판단**:
  - **latency는 사실상 무승부**: chunk200이 데이터 4.5배 많은데도 mean 기준 0.95ms, p99 기준
    2ms 차이 — HNSW가 데이터 크기에 sub-linear하게 스케일함을 실측으로 확인. 사람이 체감할
    수준이 아님.
  - **정확도는 chunk200이 뚜렷하게 우세**: recall@1 +11.4%p(상대 +26.5%), mrr@10 +12.3%p.
  - **향후 rerank를 붙여도 이 격차는 유지됨**: rerank는 retrieval이 이미 뽑아온 top-k 안에서만
    재정렬 가능 — retrieval의 recall@k가 rerank 이후 최종 정확도의 상한선이 됨. recall@20
    기준 chunk1000은 쿼리의 19.8%가 애초에 top-20에 정답이 없어서(chunk200은 9.8%) rerank로도
    구제 불가능. 즉 rerank 도입 여부와 무관하게 chunk 크기 선택은 여전히 중요.
  - **용량은 chunk200이 4.1배 비싸지만(11GB vs 2.66GB)**, 절대량 자체는 실서비스 서버 기준
    감당 못할 수준은 아니라고 판단.
  - → 위 종합으로 **chunk200 쪽으로 결론이 기움** (chunk500은 테스트하지 않고 스킵, 이하 참고)
- **디스크 정리**: chunk1000 관련 자원(Postgres `chunks_1000` 테이블, `kure-v1_chunk1000_corpus.npy`
  캐시) 삭제 — 결과 수치는 위 표로 기록해뒀으므로 원본 데이터/DB는 더 안 남겨둠.
- **다음**: chunk200을 최종 chunk 크기로 확정할지 마지막 결정만 남음 (PROJECT_STATE.md 갱신 예정).

## 2026-08-19 — overlap 실험 (chunk200_overlap50 / chunk300 / chunk300_overlap100), chunk200 결론 재검토

- **동기**: chunk200(overlap 없음)으로 기울었던 결정이 dense 단독 기준이었고, chunk_size 다음으로
  안 본 하이퍼파라미터(overlap)가 있어서 추가 검토. `server/kure_chunk_overlap.py` 작성 —
  kure_chunk.py와 동일한 exact-search 방식(pgvector 빌드 없이 numpy로 빠르게 스크리닝),
  chunk_document에 overlap 지원 추가(`step = chunk_size - overlap`, chunk 길이 자체는 고정).
  디스크 절약을 위해 설정마다 임베딩 캐시를 지우려던 초기 설계는, DB 빌드(HNSW 인덱스)와 달리
  npy 자체는 가볍다는 걸 확인하고 **끝까지 보존하는 방식으로 변경**(재인코딩 낭비 방지).
- **결과** (val_query.json 7,280개, exact search):

| config | recall@1 | recall@5 | recall@10 | recall@20 | mrr@10 | num_chunks | encode_time |
|---|---|---|---|---|---|---|---|
| chunk200 (overlap 없음, 기존) | 0.5541 | 0.8048 | 0.8734 | 0.9192 | 0.661 | 823,763 | 7367.7s |
| **chunk200_overlap50** | **0.5848** | 0.8321 | 0.8912 | 0.9372 | **0.6899** | 1,090,921 | 10282.1s |
| chunk300_overlap0 | 0.5422 | 0.7896 | 0.8595 | 0.9087 | 0.647 | 556,578 | 5179.6s |
| **chunk300_overlap100** | **0.5826** | 0.8236 | 0.8861 | 0.9304 | **0.6849** | **823,763** | 7833.1s |

- **분석**:
  - overlap 추가가 뚜렷하게 유효함(chunk200_overlap50이 chunk200 대비 recall@1 +3.07%p —
    pgvector 근사 오차(~1.1%p)보다 3배 커서 노이즈로 보기 어려움).
  - overlap 없이 chunk_size만 300으로 키우면 오히려 나빠짐(chunk300_overlap0 recall@1 0.5422 <
    chunk200 0.5541) — 기존 "짧을수록 좋다" 경향과 일치, chunk_size 확대 자체는 손해.
  - **chunk300_overlap100이 chunk200과 chunk 개수가 정확히 동일(823,763)** — 즉 같은 DB
    용량/latency 비용으로 recall@1 +2.85%p를 추가로 얻는 결과. "공짜 개선"에 가까움.
  - chunk200_overlap50(최고 정확도, chunk 32% 더 많음) vs chunk300_overlap100(정확도 거의
    동률 -0.22%p, chunk 24.5% 더 적음) — 이 둘의 차이는 pgvector 근사 오차 범위 안이라 exact
    search만으로는 우열 확정 불가.
  - 캐시 보존됨: `kure-v1_chunk200_overlap50_corpus.npy`(2.08GB), `kure-v1_chunk300_overlap100_corpus.npy`
    (1.57GB) — 재인코딩 없이 pgvector 빌드에 재사용 가능. `kure-v1_chunk300_overlap0_corpus.npy`
    (1.06GB)는 결론에서 밀려서 참고용으로만 유지.
- **결정**: chunk200(overlap 없음) 단독 확정은 보류. chunk200_overlap50 vs chunk300_overlap100
  **둘 다 실제 pgvector에 적재해서 latency/용량까지 포함한 최종 비교** 진행하기로 함. 기존
  `chunks_200`(overlap 없는 버전) 테이블은 이 비교를 위해 삭제.
- **다음**: `build_vector_db.py`에 overlap 지원 추가(파일명 규칙 + chunk_document) →
  `chunks_200_overlap50`, `chunks_300_overlap100` 두 테이블 빌드 → `test_val_pgvector.py`로
  각각 recall/latency 실측 → 최종 chunk 전략 확정.

## 2026-08-19 — build_vector_db.py/test_val_pgvector.py overlap 지원, chunk200_overlap50 pgvector 실측

- **스크립트 정비**: `server/db/schema.sql`을 고정 3테이블 생성 대신 `make_chunk_table(suffix)`
  함수만 정의하도록 일반화(호출 시점에 원하는 조합만 생성). `build_vector_db.py`의
  `chunk_document()`에 overlap 파라미터 추가(kure_chunk_overlap.py와 동일 로직), npy
  파일명/테이블명 규칙을 `overlap=0`이면 기존과 동일(`chunks_{size}`), `overlap>0`이면
  `chunks_{size}_overlap{overlap}`로 분기. `test_val_pgvector.py`는 `--chunk-size` 대신
  `--table`로 임의 테이블명을 받도록 변경. 빌드 전 chunk_document 재생성 결과가 캐시된 npy의
  row 수와 정확히 일치하는지 사전 검증 완료(chunk200_overlap50: 1,090,921 일치, chunk300_overlap100:
  823,763 일치).
- **디스크 정리**: 기존 `chunks_200`, `chunks_500`(빈 테이블) 삭제 — overlap 두 후보를
  비교하기 위한 자리 확보. 결론에서 밀린 `kure-v1_chunk300_overlap0_corpus.npy`(1.06GB), 더는
  안 쓰는 `kure-v1_corpus.npy`(no-chunk baseline, 88M)도 삭제.
- **chunk200_overlap50 pgvector 빌드+검증**: 1,090,921행 적재(716.3초), HNSW 인덱스 빌드
  5395.6초(≈90분), 테이블+인덱스 총 용량 **15GB** (사전 추정 14.6GB와 근접). val_query.json
  7,280개로 검증:

| | recall@1 | recall@5 | recall@10 | recall@20 | mrr@10 | latency mean | latency p99 |
|---|---|---|---|---|---|---|---|
| chunk200 (overlap 없음, pgvector) | 0.5433 | 0.7905 | 0.8577 | 0.9021 | 0.6487 | 7.73ms | 12.63ms |
| **chunk200_overlap50 (pgvector)** | **0.5758** | 0.8202 | 0.8784 | 0.9228 | **0.6798** | 7.25ms | 11.50ms |
| (참고) chunk200_overlap50 exact | 0.5848 | 0.8321 | 0.8912 | 0.9372 | 0.6899 | - | - |

  - exact search 대비 pgvector 손실폭 약 0.9~1.4%p — 지금까지 패턴과 일치, 정상.
  - **기존 chunk200(overlap 없음) 대비 실제 DB 환경에서도 확실히 우세**: recall@1 +3.25%p,
    mrr@10 +3.11%p. **latency는 오히려 근소하게 더 빠름**(오차 범위 내로 보이나 최소한 손해는
    없음) — overlap을 추가해도 검색 속도에 불리하지 않음을 실측으로 재확인.
  - 트레이드오프는 순수하게 **DB 용량(+36%, 11GB→15GB)** 뿐.
- **디스크 위기**: 테이블 빌드 후 여유 공간이 1.7GB(95% 사용)까지 떨어짐 — chunk300_overlap100
  (예상 11GB)을 이어서 만들 공간이 없음. `chunks_200_overlap50` 테이블은 위 결과를 문서에
  기록한 뒤 삭제 예정(수치는 보존되므로 원본 없이도 최종 판단 가능, 필요하면 npy 캐시
  `kure-v1_chunk200_overlap50_corpus.npy`로 재인코딩 없이 재구축 가능).
- **다음**: `chunks_200_overlap50` 삭제 → 여유 공간 확보 → `chunks_300_overlap100` 빌드+검증 →
  두 후보(둘 다 chunk200 overlap-없음보다 우세, exact search 기준 사실상 동률) 중 최종 선택.

## 2026-08-20 — chunk300_overlap100 최종 확정 (chunking 전략 확정)

- **chunks_300_overlap100 pgvector 빌드+검증**: 823,763행 적재(547.4초), HNSW 인덱스 빌드
  3934.7초(≈66분), 테이블+인덱스 용량 **11GB**(사전 추정과 일치, chunk200과 row 수가 같아서
  용량도 거의 동일).
- **1차 latency 측정에서 이상치 발견**: 빌드 직후 바로 테스트했더니 mean 12.34ms/p99 31.07ms로
  chunk200_overlap50(7.25ms/11.50ms)보다 눈에 띄게 느리게 나옴 — row 수가 더 적은데(82만 vs
  109만) 더 느린 게 이상해서 재검증함.
- **재검증 결과 — 첫 측정은 cold cache 노이즈였음**: 같은 테이블로 바로 다시 측정하니
  mean 6.87ms/p99 ~10ms로 나옴(recall/mrr은 결정적 계산이라 두 번 다 동일). 빌드 직후
  캐시가 안 데워진 상태에서 첫 측정을 한 게 원인으로 추정 — latency 측정은 웜업 후 값을
  신뢰해야 한다는 교훈.
- **최종 비교** (chunk200_overlap50 vs chunk300_overlap100, 둘 다 pgvector 실측):

| | chunk200_overlap50 | chunk300_overlap100 |
|---|---|---|
| recall@1 | 0.5758 | 0.5714 (-0.44%p) |
| recall@5 | 0.8202 | 0.8085 (-1.17%p) |
| recall@10 | 0.8784 | 0.8703 (-0.81%p) |
| recall@20 | 0.9228 | 0.9133 (-0.95%p) |
| mrr@10 | 0.6798 | 0.6720 (-0.78%p) |
| latency mean | 7.25ms | 6.87ms (오히려 근소 우위) |
| latency p99 | 11.50ms | ~10ms (오히려 근소 우위) |
| DB 용량 | 15 GB | **11 GB (-27%)** |

  - latency는 사실상 무승부(chunk300_overlap100이 오히려 근소하게 빠름), 정확도는
    chunk200_overlap50이 소폭 우세(1%p 안팎, recall@1은 0.44%p로 거의 오차 수준), 용량은
    chunk300_overlap100이 27% 더 작음.
- **최종 결정: chunk300_overlap100 채택.** 근거 — 정확도 격차가 작아서(특히 recall@1은
  거의 무의미한 차이) 용량 27% 절감의 실익이 더 크다고 판단. Chunking 전략 확정:
  **chunk_size=300, overlap=100(글자 단위), KURE-v1 임베딩.**
- **정리**: `chunks_300_overlap100`을 정식 서빙용 테이블로 유지. `chunks_200_overlap50`은
  이미 삭제됨(수치는 위 표에 보존). 남은 npy 캐시 중 `kure-v1_chunk200_overlap50_corpus.npy`
  (2.08GB, 탈락 후보)는 정리 대상, `kure-v1_chunk300_overlap100_corpus.npy`(1.57GB, 채택된
  후보)는 재구축 대비 보존.
- **다음**: Phase 1(Chunking/Metadata 설계) 공식 종료. Phase 2로 — 사용자가 임베딩 모델
  도메인 파인튜닝(train_query.json 기반 contrastive fine-tuning) 착수 예정.

## 2026-08-20 — 임베딩 파인튜닝 설계 논의 (in-batch negative 버전 착수)

Phase 2 첫 작업으로 KURE-v1 도메인 파인튜닝 설계를 논의하고 1단계(in-batch negative만) 코드
작성. 논의된 결정사항:

- **학습 단위: chunk 단위** (문서 단위 아님) — 실제 retrieval이 chunk 단위(`chunks_300_overlap100`)
  로 이뤄지므로 정합성을 위해 chunk 단위로 학습하기로 함.
- **positive chunk 선택 방법**: `train_query.json`은 문서(case_id) 단위 정답만 있고 chunk
  단위 정답은 원본 데이터 자체에 없음 — 어차피 근사치(proxy label)를 쓸 수밖에 없는 상황.
  세 가지 대안(① 현재 KURE-v1로 정답 문서 내에서 최고 유사도 chunk 선택, ② 정답 문서의 모든
  chunk를 positive로 사용, ③ 학습 중 매 스텝 동적 재선택(MIL))을 검토. ①은 "모델 자신의
  판단으로 그 모델을 학습"시키는 순환 논리 우려가 제기됐으나, "82만 개 전체에서 정답 찾기"
  (어려운 문제, 지금 recall@1 0.57)와 "이미 정답이라고 알고 있는 문서 1개의 chunk 10~30개
  중에서 고르기"(훨씬 쉬운 하위 문제)는 난이도가 다르다는 근거로 순환 위험이 실질적으로는
  낮다고 판단. 판결문 특성상 당사자 표시·주문 등 무관한 chunk가 많아 ②(전부 positive)는
  노이즈가 클 것으로 예상돼 채택 안 함. **① 채택**, 다만 완벽한 정답이 아닌 근사치임을 명시.
- **loss / 학습 방식**: `MultipleNegativesRankingLoss`(InfoNCE 계열, sentence-transformers) —
  BGE-M3/KURE-v1 원래 학습 방식과 같은 계열이라 이어서 학습하기 자연스러움. 1단계는
  in-batch negative만 사용, hard negative는 2단계(추후 결정)에서 추가해 성능 개선분을 ablation
  형태로 비교할 계획.
- **batch size 중요성**: in-batch negative 방식은 배치 크기가 곧 negative 개수(batch_size-1)로
  직결돼서, 대조학습 일반적으로 batch가 클수록 학습 신호가 좋아짐 — 이 인스턴스(RTX 3090
  24GB) VRAM이 허용하는 한 크게 잡는 게 유리. gradient accumulation은 유효 batch는 키워도
  negative pool 크기(물리적 batch당 계산되는 것)는 안 늘려준다는 점 확인.
- **false negative 위험 확인**: train_query.json이 문서 단위로 순회하며 만들어진 파일이라
  **셔플 안 하면 인접한 두 쿼리가 같은 case_id를 가리키는 비율이 8.15%**(3,368/41,318)로
  실측됨 — 배치를 순서대로 자르면 같은 배치에 같은 문서 관련 쿼리가 뭉쳐 들어가 false
  negative가 심각해질 수 있음. **매 epoch 셔플**(표준 관행, HuggingFace Trainer 기본 동작)
  로 해결 — 셔플 후 잔여 충돌 확률은 대략 0.2~0.3% 수준으로 계산돼 허용 가능한 노이즈로 판단.
- **구현**: `server/finetune/` 신설.
  - `prepare_data.py` — train_query.json 순회하며 각 쿼리의 정답 문서 chunk들을
    `chunks_300_overlap100`에서 조회, 현재 KURE-v1로 유사도 계산해 최고 chunk를 positive로
    선택 → `finetune_pairs.jsonl` 생성
  - `train.py` — sentence-transformers `SentenceTransformerTrainer` + `MultipleNegativesRankingLoss`
    로 1단계(in-batch negative) 파인튜닝
  - `eval_model.py` — 임의 모델 경로(base KURE-v1 또는 파인튜닝 결과)를 받아 val_query.json
    기준 recall@k/mrr@10 계산 (kure_chunk_overlap.py와 동일한 exact-search 방법론) — 파인튜닝
    전후 비교용
- **다음**: 사용자가 `prepare_data.py` → `train.py` 실행 → `eval_model.py`로 base vs
  파인튜닝 비교. 이후 hard negative 채굴 방식 결정해서 2단계 진행.

## 2026-08-20 — Phase A(in-batch negative) 파인튜닝 결과, Phase B(hard negative) 착수

- **Phase A 결과 — 확실한 개선 확인** (base KURE-v1 exact search vs 파인튜닝 후, val_query.json
  7,280개, chunks_300_overlap100 코퍼스 기준):

| | base (exact) | Phase A (in-batch) | 차이 |
|---|---|---|---|
| recall@1 | 0.5826 | **0.6574** | **+7.48%p** (상대 +12.8%) |
| recall@5 | 0.8236 | 0.8776 | +5.40%p |
| recall@10 | 0.8861 | 0.9262 | +4.01%p |
| recall@20 | 0.9304 | 0.9622 | +3.18%p |
| mrr@10 | 0.6849 | **0.7519** | +6.70%p |

  - in-batch negative만으로도 전 지표에서 뚜렷한 개선 — 도메인 파인튜닝이 유효함을 확인.
  - 지난번 우려했던 "positive chunk 근사 라벨링(정답 문서 내 최고 유사도 chunk 선택)이
    심각하게 잘못됐으면 어쩌나" 하는 리스크도, 이렇게 일관된 개선이 나온 것 자체가 그 방법론이
    실제로 크게 잘못되진 않았다는 간접 증거로 봄 (라벨이 심하게 틀렸다면 이 정도 개선은
    나오기 어려움).
- **eval_model.py 디스크 이슈 발견+수정**: 파인튜닝 모델 평가 시 코퍼스 캐시를 float32로
  저장해서(3.37GB, base 캐시는 fp16이라 1.69GB였음) 디스크가 다시 위험 수준(1.9GB)까지
  떨어짐. 기존 캐시는 fp16으로 재저장해서 절반으로 줄이고, 스크립트도 앞으로 fp16으로
  저장하도록 수정.
- **eval_model.py base 모델 캐시 재사용 버그도 수정**: 원래 매번 재인코딩하도록 짜여있었는데
  (~2시간 10분 소요, 이전에 "65분"이라고 잘못 안내했던 것도 정정 — index 빌드 시간과 혼동),
  base KURE-v1은 `kure_chunk_overlap.py`가 만든 기존 캐시(`kure-v1_chunk300_overlap100_corpus.npy`)
  가 정확히 같은 걸 재사용하면 되므로 캐시 재사용 로직 추가. 파인튜닝 체크포인트 등 다른
  모델도 최초 1회만 인코딩 후 캐시.
- **Phase B(hard negative) 설계+구현**:
  - **채굴 기준 모델은 Phase A 결과물** — base 모델로 채굴하면 Phase A가 이미 고친 부분까지
    또 negative로 잡아 학습 신호가 낭비되므로, "지금 모델이 헷갈려하는 것"을 반영하기 위해
    Phase A 모델로 채굴.
  - `mine_hard_negatives.py` 신설 — 전체 코퍼스(82만 chunk)에서 학습 쿼리와 유사도 상위
    100개(`--search-pool`) 중 정답 문서가 아닌 것 중 상위 1개(`--num-negatives`)를 hard
    negative로 채택. eval_model.py가 Phase A 평가 시 만들어둔 코퍼스 캐시를 재사용해서
    재인코딩 없음. 5개 샘플로 검증 완료 — 전부 정답과 무관한 다른 문서에서 정상적으로
    negative를 찾음.
  - `train.py`를 확장해서 `--pairs-file`에 hard_negative_chunk 필드가 있으면 자동으로
    (anchor, positive, negative) 3열 데이터셋으로 학습하고, **시작점도 base가 아니라 Phase A
    결과 모델**로 자동 전환(처음부터 다시 학습하는 게 아니라 Phase A 위에 이어서 학습) —
    output도 `kure-v1-finetuned-hard`로 분리해 Phase A 결과와 별도 보존.
- **다음**: 사용자가 `mine_hard_negatives.py` → `train.py --pairs-file finetune_pairs_hard.jsonl`
  → `eval_model.py`로 Phase A vs Phase B 비교.

## 2026-08-20 — Phase B 1차 시도 실패(OOM) → 수정 → Phase B 결과가 Phase A보다 나빠짐 → 원인 수정

- **OOM 발생+수정**: `train.py`를 batch_size=32(Phase A와 동일 기본값)로 처음 실행했을 때
  CUDA OOM(24GB 꽉 참). hard negative가 있으면 배치당 텍스트 3개(anchor/positive/negative)를
  인코딩해야 해서 Phase A(2개)보다 메모리를 더 씀 — `train.py`에 phase별 기본값 분리 추가
  (hard negative 있으면 batch_size 기본 32→16) + `gradient_checkpointing=True` 추가.
- **Phase B 1차 결과 — Phase A보다 전 지표에서 나쁘게 나옴**:

| | Phase A (in-batch) | Phase B 1차 (hard negative) | 차이 |
|---|---|---|---|
| recall@1 | 0.6574 | 0.6231 | **-3.43%p** |
| recall@5 | 0.8776 | 0.8489 | -2.87%p |
| recall@10 | 0.9262 | 0.9033 | -2.29%p |
| recall@20 | 0.9622 | 0.9401 | -2.21%p |
| mrr@10 | 0.7519 | 0.7205 | **-3.14%p** |

- **원인 분석**: train_query.json의 case_ids가 전부 1개씩이라(41,319건 확인) multi-label
  누락 버그는 아님. 유력한 원인은 **`mine_hard_negatives.py`가 유사도 최상위 1개를 그대로
  hard negative로 썼다는 것** — hard negative 개념을 처음 논의할 때 "너무 상위(1~2위)는
  false negative 위험이 있어 5~30위권에서 고르는 경우도 많다"고 짚었었는데, 정작 구현에서는
  이 안전장치를 빠뜨렸었음. 법률 판례는 같은 법리를 인용하는 표준 문구·유사 사건 유형이
  많아서, "정답 문서는 아니지만 유사도 1위"인 chunk가 실제로도 상당히 관련 있는 내용일 위험이
  이 도메인에서 특히 큼. 부차적으로 계속 학습(Phase A 위에 같은 학습률로 3 epoch 추가)로 인한
  과적합/드리프트 가능성도 배제 못함.
- **수정**:
  - `mine_hard_negatives.py`에 `--skip-top`(기본 5) 추가 — 유사도 순위 1~5위는 건너뛰고
    6위부터 negative로 채택해 false negative 위험 완화. 3개 샘플로 전/후 비교 검증 완료.
  - `train.py`: Phase B 기본 학습률 2e-5→**5e-6**(계속 학습이라 더 보수적으로), 기본
    epoch 3→**1**(과적합 위험 완화)로 낮춤. 전부 phase별 기본값 자동 분기(직접 override 가능).
- **다음**: `mine_hard_negatives.py` 재실행(새 skip_top 적용) → `train.py --pairs-file
  finetune_pairs_hard.jsonl` 재학습 → `eval_model.py`로 재평가, Phase A/Phase B 1차와 비교.

## 2026-08-20 — 재사용 가능한 평가 하니스 신설 (`server/eval/`)

- **배경**: model_test.py/jhgan.py/bge_hybrid.py/kure_chunk.py/kure_chunk_overlap.py/
  test_val_pgvector.py/finetune/eval_model.py 7개 스크립트가 전부 "val_query.json 로딩 →
  검색 → recall@k/mrr@10 계산 → 출력"을 매번 새로 구현하고 있었음 — sparse/hybrid/rerank를
  계속 실험하려면 이 시점에 정리하는 게 이후 시간을 아껴줌(PROJECT_STATE.md Phase 1 로드맵에
  이미 예정돼있던 항목). query rewriting은 이번 스코프에서 제외하기로 함.
- **구조**:
  - `harness.py` — `evaluate(retriever, val_data, ...)` 하나로 recall@k/mrr@10(+선택적
    latency) 계산. `retriever.retrieve_batch(queries, max_k) -> list[list[str]]` 인터페이스만
    맞으면 어떤 검색 방법이든 그대로 평가 가능 — 새 검색 기법(sparse, hybrid, rerank) 추가할
    때 이 파일은 안 건드리고 retrievers.py에 클래스만 추가하면 됨
  - `retrievers.py` — 지금 당장 필요한 두 가지 구현체:
    - `DenseExactRetriever` — numpy exact search (kure_chunk.py류가 하던 방식, pgvector 없이
      빠른 스크리닝용)
    - `PgvectorRetriever` — 실제 Postgres+pgvector HNSW 검색 (test_val_pgvector.py가 하던
      CTE 쿼리 패턴 그대로, 실제 서빙 latency까지 측정 가능)
  - `run_eval.py` — CLI. `--mode exact --model-path ... --chunk-size ... --overlap ...` 또는
    `--mode pgvector --model-path ... --table ...`. exact 모드는 eval_model.py가 갖고 있던
    "모델+chunk 설정별 코퍼스 임베딩 캐시 재사용"(fp16 저장, 재인코딩 방지) 기능도 포함시켜서
    완전히 대체 가능하게 만듦(단, 캐시 파일명 규칙이 기존 스크립트들과 달라서 예전 캐시를
    자동으로 못 읽음 — 새로 인코딩됨).
- **검증**: 가짜 소규모 corpus(문서 3개)로 DenseExactRetriever+harness 로직 검증(recall@1=1.0
  정상), 캐시 저장/재사용 라운드트립 검증, `chunks_300_overlap100` 실제 테이블로 pgvector 모드
  200개 쿼리 샘플 실행 — 기존 전체 7,280개 결과(recall@1 0.5714)와 비슷한 범위(0.54) 확인.
- **정리 대상(사용자 판단 필요)**: 아래 스크립트들은 이제 `run_eval.py`로 대체 가능 —
  결과 수치는 이미 log.md/PROJECT_STATE.md에 다 기록돼 있어서 삭제해도 근거는 안 사라짐.
  단, bge_hybrid.py는 sparse 검색 로직 자체(하니스의 Retriever로 아직 포팅 안 함)가 남아있어서
  나중에 SparseRetriever/HybridRetriever 만들 때 참고용으로 남겨둘 가치는 있음:
  - `server/model_test.py`, `server/jhgan.py`, `server/bge_hybrid.py`
  - `server/kure_chunk.py`, `server/kure_chunk_overlap.py`
  - `server/db/test_val_pgvector.py`
  - `server/finetune/eval_model.py`
- **다음**: 사용자가 위 목록 중 지울 파일 결정. Phase B(hard negative) 재실행 결과 나오면
  이제부터는 `run_eval.py`로 평가.
- **정리 실행**: `model_test.py`, `jhgan.py`, `kure_chunk.py`, `kure_chunk_overlap.py` 삭제.
  `bge_hybrid.py`(sparse 로직 미포팅이라 참고용 보존), `test_val_pgvector.py`,
  `finetune/eval_model.py`는 남겨둠(추후 판단).
- **git**: 위 정리 + eval 하니스 신설 내용을 `.gitignore`에 `server/finetune/output/`,
  `server/finetune/*.jsonl` 추가(파인튜닝 모델 2.2GB가 GitHub 파일 크기 제한(100MB) 초과라
  누락 시 push 실패했을 것)한 뒤 커밋(`882f8f5`) + push 완료.

## 2026-08-21 — Phase B(hard negative, skip_top 수정판) 재실행 결과 — 최종 임베딩 모델 확정

- **결과 — Phase A를 확실히 상회, 가설 검증됨**:

| | base(exact) | Phase A(in-batch) | Phase B 1차(버그) | **Phase B 2차(skip_top=5)** |
|---|---|---|---|---|
| recall@1 | 0.5826 | 0.6574 | 0.6231 | **0.6826** |
| recall@5 | 0.8236 | 0.8776 | 0.8489 | **0.9040** |
| recall@10 | 0.8861 | 0.9262 | 0.9033 | **0.9446** |
| recall@20 | 0.9304 | 0.9622 | 0.9401 | **0.9698** |
| mrr@10 | 0.6849 | 0.7519 | 0.7205 | **0.7766** |

  - Phase A 대비 recall@1 +2.52%p, mrr@10 +2.47%p — "유사도 최상위 negative가 false
    negative일 가능성이 크다"는 어제 가설이 실측으로 확인됨.
  - **base 대비 전체 파이프라인 개선폭(최종 성과): recall@1 +10.0%p(상대 +17.2%),
    mrr@10 +9.17%p** — chunking 전략 확정(chunk300_overlap100) + Phase A(in-batch) +
    Phase B(hard negative, skip_top=5) 전체를 합친 순수 개선.
- **결정: 이 모델(`server/finetune/output/kure-v1-finetuned-hard/`)을 최종 임베딩 모델로 확정.**
  1차 시도 실패 → 원인 분석(false negative 가설) → 수정(`--skip-top`) → 검증까지의 전체
  과정이 기록으로 남아있어 근거가 명확함.
- **다음**: 임베딩 모델 파인튜닝(Phase 2 핵심 항목) 완료. 이후 후보: sparse+dense hybrid
  score 결합, reranker(기성 또는 파인튜닝), MMR 다양성 재정렬, 메타데이터 필터 결합,
  query rewriting은 이번 스코프 제외(2026-08-20 논의) — 우선순위는 PROJECT_STATE.md 참고.
- **디스크 정리**: Phase A 모델(2.2GB)+그 코퍼스 캐시(1.6GB)+중간 학습 데이터(finetune_pairs*.jsonl,
  111MB) 삭제 — Phase B가 최종이라 재현 불필요. 929MB(98%)까지 위험했던 여유를 4.8GB로 확보.
- **중요 발견**: `chunks_300_overlap100` 운영 DB는 아직 **base KURE-v1 임베딩**으로 구축돼있음
  (파인튜닝 전에 만들어짐) — Phase B 검증(recall@1 0.6826)은 별도 exact-search 방식으로 한
  것이지 DB 자체를 바꾼 게 아님. DB를 파인튜닝 임베딩으로 재구축해야 실제 검색에 반영됨.

## 2026-08-21 — sparse(BM25) 검색 설계: Postgres 내장 tsvector + Kiwi 채택, DB 재구축 준비

- **sparse 방식 재검토**: BGE-M3 learned sparse를 먼저 검토했으나, 이 인스턴스 pgvector가
  0.6.0(apt 최신 버전)이라 sparse 벡터 타입(`sparsevec`)이 필요한 0.7.0 미달 — 같은 테이블에
  못 넣고 별도 저장소가 필요해짐. 게다가 BGE-M3 모델을 KURE-v1(파인튜닝)과 별도로 GPU에
  상시 로드해야 하는 부담도 있음. **Postgres 내장 전문검색(tsvector/GIN) + Kiwi 형태소
  분석기**로 결정 — 같은 테이블에 컬럼만 추가하면 되고 별도 모델 상시 로드도 불필요. 다만
  Postgres 기본 `ts_rank`는 정식 BM25 공식과 동일하지 않음(개념적으로 유사한 역색인+빈도
  기반 랭킹, 파라미터화된 BM25는 아님) — 필요해지면 `pg_search`(ParadeDB) 등 확장 검토 가능.
- **Kiwi 처리 속도 실측**: `num_workers=-1`(전체 코어) 배치 처리 기준 823,763개 chunk 전체
  약 7분(1,930개/초) — GPU 불필요, 오래 안 걸림.
- **형태소 필터링**: 조사(J*)/어미(E*)/접미사(XS*)/문장부호 제외, 명사(NN*)/동사(VV,VA,VX)/
  부사(MAG,MAJ)/숫자·한자·외국어(SN,SH,SL)/어근(XR)만 남겨서 검색 노이즈 감소.
- **⚠️ 중요 버그 발견+확인**: Kiwi가 복합명사를 쪼갬(예: "손해배상"→"손해"+"배상" 별도 토큰).
  그래서 **검색어를 Kiwi 없이 원문 그대로 `to_tsquery`에 넣으면 매칭 실패**함(저장은
  "손해"+"배상" 두 토큰인데 검색은 "손해배상" 한 토큰이라 문자열이 다름). 테스트로 재현
  확인 후, **검색 시에도 사용자 질문을 반드시 Kiwi로 먼저 처리한 뒤 `plainto_tsquery`에
  넣어야 함**을 확인 — 나중에 SparseRetriever 구현 시 필수 반영 사항.
- **`build_vector_db.py` 확장**:
  - `chunk_text_kiwi TEXT`(Kiwi 토큰, 공백 조인) + `content_tsv tsvector`(GIN 인덱스) 컬럼을
    `schema.sql`의 `make_chunk_table`에 추가
  - `--npy-path` 옵션 추가 — 임의의 임베딩 캐시를 지정 가능해짐 (기존엔 `kure-v1_chunk...`
    고정 파일명만 지원해서 파인튜닝 모델 임베딩을 못 썼음)
  - COPY 적재 후 `UPDATE ... SET content_tsv = to_tsvector('simple', chunk_text_kiwi)` 한
    번에 처리, 이어서 GIN 인덱스 생성
  - 가짜 테이블로 COPY+UPDATE+GIN 검색까지 전체 파이프라인 검증 완료(위 버그도 이 과정에서
    발견)
- **다음**: `chunks_300_overlap100`을 파인튜닝 모델(`kure-v1-finetuned-hard`) 임베딩+
  Kiwi/tsvector 컬럼까지 포함해서 재구축 — `--npy-path`로 기존 eval 캐시
  (`eval_._output_kure-v1-finetuned-hard_chunk300_overlap100_corpus.npy`, 재인코딩 불필요)
  재사용.

## 2026-08-24 — DB 재구축 중 심각한 성능 버그 발견+수정: 기존 인덱스가 COPY를 극도로 느리게 만듦

- **증상**: 재구축 실행 중 "적재" 단계가 50,000행(COPY_BATCH 1개분) 이후 급격히 느려짐 —
  두 번째 50,000행 COPY에 11분 넘게 걸림(이 속도면 823,763행 전체에 3시간 이상 예상).
- **원인**: `chunks_300_overlap100`은 **이전 빌드(base 모델용)에서 이미 HNSW 인덱스가 붙어있는
  기존 테이블**이었음. `TRUNCATE`는 데이터만 비우고 인덱스는 그대로 두는데, 인덱스가 붙어있는
  상태로 COPY하면 Postgres가 **row 하나 넣을 때마다 HNSW 그래프를 실시간으로 갱신**해야 해서
  극도로 느려짐. 원래 빠른 이유(대량 데이터 먼저 넣고 인덱스는 나중에 한 번에 빌드)가
  깨진 것 — 완전히 새로운 테이블(`chunks_200` 등)을 만들 때는 안 나타나고, **기존 테이블을
  재사용/재구축할 때만** 나타나는 버그라 이번에 처음 발견됨.
- **수정**: `build_vector_db.py`의 `TRUNCATE` 직전에 `DROP INDEX IF EXISTS`로 기존 인덱스
  (HNSW, case_no, case_type, content_tsv GIN)를 전부 먼저 제거하도록 추가. 이후 코드는 원래
  대로 데이터 적재 완료 후 인덱스를 한 번에 재생성.
- **교훈**: 같은 테이블 이름으로 재구축(rebuild-in-place)하는 스크립트는 "완전히 빈 테이블"을
  가정하면 안 되고, 기존 인덱스/제약조건이 있을 수 있다는 걸 감안해서 짜야 함.
- **다음**: 수정된 스크립트로 재실행.
- **재실행 완료 — DB 재구축 성공**: 823,763행 적재(541.8초), content_tsv 채우기(67.6초),
  인덱스 빌드(HNSW+case_no+case_type+GIN, 4250.7초≈70.8분), 테이블+인덱스 총 용량 13GB.
  인덱스 목록/`content_tsv` 전체 행 채움/Kiwi 토큰 샘플까지 확인 완료.
  **`chunks_300_overlap100`이 이제 (1) 파인튜닝 모델(kure-v1-finetuned-hard) dense 임베딩,
  (2) Kiwi+tsvector 기반 sparse(BM25) 검색 컬럼을 모두 갖춘 정식 운영 DB로 확정됨.**
- **다음**: `server/eval/retrievers.py`에 `SparseRetriever`(Kiwi 토큰화 + `plainto_tsquery`
  검색) 구현 → dense(PgvectorRetriever)와 RRF로 결합하는 `HybridRetriever` → `run_eval.py`로
  성능 실측·비교.

## 2026-08-24 — SparseRetriever/HybridRetriever 구현, ⚠️ AND매칭 버그 발견+수정

- **`SparseRetriever` 구현**: `server/eval/retrievers.py`에 추가 — 검색어를 Kiwi로 토큰화
  (`build_vector_db.py`의 `KIWI_KEEP_PREFIXES`와 동일 기준 유지 필요, 주석으로 명시)한 뒤
  `content_tsv` 컬럼(GIN 인덱스)으로 검색.
- **⚠️ 버그 발견+수정 — `plainto_tsquery`는 전부 AND로 묶음**: 첫 구현은 `plainto_tsquery`를
  썼는데, 실제 val 쿼리 5개로 테스트하니 **전부 매칭 0건**이 나옴. 원인: val 질문이 길어서
  Kiwi 토큰이 40~50개나 되는데, `plainto_tsquery`는 이 전부를 `&`(AND)로 묶어서 "이 모든
  단어가 한 chunk(300자)에 다 들어있어야 매칭"이 됨 — 애초에 불가능한 조건. `to_tsquery`로
  직접 `|`(OR)로 묶어 "많이 겹칠수록 `ts_rank_cd` 점수가 높다"는 정상적인 랭킹 방식으로 수정.
  수정 후 실제 val 10개로 재검증: recall@10 6/10 (스크리닝 샘플, 정식 지표 아님).
- **`HybridRetriever` 구현**: dense(`PgvectorRetriever`)+sparse(`SparseRetriever`)를
  RRF(k=60, bge_hybrid.py 2026-08-18과 동일 상수)로 결합. 같은 10개 쿼리로 검증:
  recall@10 9/10 — dense 단독/sparse 단독보다 나은 방향성 확인(정식 7,280개 평가는 아직 안 함).
- **`run_eval.py`에 `--mode sparse`/`--mode hybrid` 추가**: sparse 모드는 dense 모델을 아예
  안 불러오도록 분기(불필요한 GPU 로딩 방지).
- **다음**: `run_eval.py --mode sparse`와 `--mode hybrid`로 val_query.json 7,280개 전체
  정식 평가 → dense 단독(recall@1 0.6826) 대비 실제 개선폭 확인.
- **⚠️ 성능 문제 발견 — sparse 쿼리가 쿼리당 ~5초로 매우 느림**: 실제 정식 평가 시작하자마자
  발견. 원인: 법률 질문 특성상 OR로 묶는 키워드가 20~50개나 돼서, 82만 행 중 40만 행 이상이
  매칭됨 → Postgres가 GIN 인덱스 대신 순차 스캔(seq scan)을 선택(`EXPLAIN ANALYZE`로 확인,
  인덱스 강제 사용해봐도 동일하게 느림 — `ts_rank_cd`를 매칭된 모든 후보에 대해 계산해야
  정렬 가능한 구조 자체가 병목).
  - **시도 1**: 문서빈도(ndoc) 상위 3%(307개) "너무 흔한 단어"(하/있/것/사건/수/등/원고/피고
    등 법률 문서 어디에나 있는 단어) 필터링 → 5.3초 → 0.78초로 개선(recall@10 12/20, 샘플).
  - **시도 2~4**: 필터링 임계값을 더 세게(1%, 0.5%), 단어 개수 제한(길이 기준/실제 희귀도
    기준 상위 N개만 사용) — 전부 추가 개선 없거나 오히려 느려지거나 정확도만 떨어짐. 원인
    불명확, 추가 조사는 보류.
  - **결정**: 사용자 확인 후 현재 상태(3% 필터링, 쿼리당 ~780ms)로 전체 평가 진행하기로 함
    — 평가는 1회성 비용이라 1.5~1.7시간 소요는 감수. **단, 실제 서빙 시 사용자 응답
    latency로 쓰기엔 여전히 느려서(dense 7~13ms 대비 100배 이상) 나중에 반드시 재검토
    필요** (Postgres 내장 검색엔 WAND/MaxScore 같은 조기 종료 최적화가 없어서 구조적 한계일
    수 있음 — `pg_search`(ParadeDB) 등 전용 확장 고려 대상).
  - **다음**: 정식 7,280개 sparse/hybrid 평가 실행(사용자).

## 2026-08-24 — 진짜 BM25로 SparseRetriever 재구현, hybrid RRF 격차 대폭 축소(그러나 아직 dense 못 넘음)

- **원인 재확인**: 위 세션에서 균등 가중(1.0/1.0) RRF가 dense 단독보다 크게 나빴던
  이유(recall@1 0.6826→0.4727)를 재검토하면서, `ts_rank_cd`가 IDF/tf 포화(k1)/문서길이
  보정(b)이 없는 단순 랭킹 함수라 sparse의 절대 품질 자체가 너무 낮았다는 결론(정식 BM25가
  아님, 학습도 불가능). 사용자가 "학습되는 sparse 모델(BGE-M3 등)"보다 **전통 BM25를 제대로
  구현**하는 쪽을 선택.
- **구현**: `server/eval/retrievers.py`의 `SparseRetriever`를 2단계 구조로 재작성.
  1단계(SQL, 기존과 동일)는 `content_tsv`/GIN으로 후보 pool_k(500)개를 `ts_rank_cd` 순으로
  빠르게 추리기만 하고, 2단계(Python, 신규)에서 후보들의 `chunk_text_kiwi`로 실제 단어
  빈도(tf)를 세서 표준 Okapi BM25 공식(`IDF(t) = log((N-ndoc+0.5)/(ndoc+0.5)+1)`,
  `k1=1.5`, `b=0.75`)으로 재점수·재정렬. IDF/평균 문서 길이(avgdl)는 `ts_stat()`으로
  `__init__` 시점에 코퍼스 전체를 한 번 스캔해 미리 캐싱(쿼리마다 재계산 안 함). 흔한 단어
  필터링(상위 3%)은 1단계 후보 검색에서만 적용하고, 실제 BM25 점수 계산에는 전체 코퍼스
  기준 IDF를 그대로 사용(흔한 단어는 IDF가 낮아 어차피 점수 기여 미미 — 걸러도 랭킹 품질
  손실 없음). 스키마/DB 변경 없음, 기존 컬럼 그대로 재사용.
- **정식 평가 결과 (val_query.json 7,280개)**:

  | | sparse 단독(구, ts_rank_cd) | sparse 단독(신, 진짜 BM25) | hybrid(신 BM25, 1.0/1.0) | dense 단독(최종 모델) |
  |---|---|---|---|---|
  | recall@1 | 0.203 | **0.5511** | 0.6309 | 0.6826 |
  | recall@5 | (미측정) | 0.7864 | 0.8515 | 0.9040 |
  | recall@10 | (미측정) | 0.8490 | 0.9070 | 0.9446 |
  | recall@20 | (미측정) | 0.8891 | 0.9515 | 0.9698 |
  | mrr@10 | (미측정) | 0.6523 | 0.7247 | 0.7766 |

  - **sparse 단독 성능이 극적으로 개선**됨(recall@1 0.203→0.5511) — IDF/tf포화/문서길이
    보정을 실제로 계산한 게 유효했음을 확인. dense 단독에는 아직 못 미치지만 격차가 크게
    좁혀짐(0.68 vs 0.20 → 0.68 vs 0.55).
  - **그러나 균등 가중(1.0/1.0) hybrid는 여전히 모든 지표에서 dense 단독보다 낮음**
    (recall@1 -5.17%p, mrr@10 -5.19%p). 다만 이전(약한 sparse 기준, -21%p 수준)보다는 훨씬
    양호 — sparse가 강해질수록 RRF의 "약한 retriever가 강한 retriever의 정답을 밀어내는"
    문제가 줄어드는 방향성은 확인됨.
- **`harness.py`에 tqdm 진행률 표시 추가**: `evaluate()`의 배치 루프에 진행 바가 없어서
  sparse/hybrid처럼 쿼리당 DB 왕복이 필요해 오래 걸리는(hybrid 7,280개 기준 약 1.5시간)
  케이스에서 진행 상황을 전혀 알 수 없었음(사용자가 "얼마나 더 걸릴지" 질문 → 코드에 진행률
  자체가 없다는 걸 확인). `for start in tqdm(range(0, n, batch_size), total=n_batches, ...)`로
  수정 — 다음 실행부터 배치 수 기준 ETA 표시됨.
- **다음**: sparse가 훨씬 강해졌으니 `sparse_weight`를 낮춰가는 가중 RRF 스윕을 소규모
  샘플(예: 50개)로 다시 실행해서, 이전(약한 sparse 기준)엔 없었던 "dense 단독을 넘어서는
  weight"가 새로 생겼는지 확인 필요 — 아직 미실행.

## 2026-08-25 — sparse_weight 정식 스윕 완료: 진짜 BM25로도 hybrid가 dense 단독을 못 넘음, dense 단독으로 확정

- **정식(7,280개 전체) 가중 RRF 스윕**: dense_weight=1.0 고정, sparse_weight만
  {1.0, 0.7, 0.5, 0.3, 0.15}로 바꿔가며 밤새 순차 실행(`run_eval.py --mode hybrid`).

  | sparse_weight | recall@1 | recall@5 | recall@10 | recall@20 | mrr@10 |
  |---|---|---|---|---|---|
  | 1.0 | 0.6309 | 0.8515 | 0.9070 | 0.9515 | 0.7247 |
  | 0.7 | 0.6332 | 0.8563 | 0.9074 | 0.9474 | 0.7279 |
  | 0.5 | 0.6420 | 0.8641 | 0.9155 | 0.9521 | 0.7367 |
  | 0.3 | 0.6508 | 0.8716 | 0.9272 | 0.9571 | 0.7464 |
  | 0.15 | 0.6600 | 0.8839 | 0.9327 | 0.9576 | 0.7553 |
  | **dense 단독** | **0.6826** | **0.9040** | **0.9446** | **0.9698** | **0.7766** |

- **결론**: weight를 낮출수록 dense 단독에 계속 가까워지지만(recall@1 기준 1.0→0.15 구간
  +2.91%p), 0.15까지 내려도 **전 지표에서 여전히 dense 단독보다 낮음**. 구간별 개선폭도
  둔화 중(0.7→0.5: +0.88%p, 0.5→0.3: +0.88%p, 0.3→0.15: +0.92%p로 정체) — 더 낮춰도(0.05
  등) dense를 역전할 가능성은 낮다고 판단. 이전(2026-08-24, 약한 ts_rank_cd 기반 sparse로
  했던 50개 샘플 스윕)에서도 동일하게 "weight→0으로 갈수록 수렴만 하고 못 넘음" 패턴이
  나왔는데, sparse를 진짜 BM25로 훨씬 강하게 만든 뒤에도(recall@1 단독 0.2035→0.5511) 같은
  결론이 재현됨 — sparse 품질 문제가 아니라, **파인튜닝된 dense 모델이 이 도메인/이
  val셋에서 이미 sparse가 보탤 여지가 거의 없을 만큼 강하다**는 것으로 결론.
- **결정: retriever는 dense 단독(`kure-v1-finetuned-hard` + pgvector HNSW)으로 최종 확정.**
  sparse+dense hybrid 튜닝은 여기서 중단. `SparseRetriever`/`HybridRetriever` 코드
  (`server/eval/retrievers.py`)는 **삭제하지 않고 보존** — 포트폴리오상 "hybrid를 시도했고
  왜 채택하지 않았는지 정량적으로 검증했다"는 과정 자체가 근거 자료이고, 나중에 다른
  dense 모델/코퍼스로 재검토할 수도 있어서 남겨둠.
- **다음**: 로드맵의 다음 우선순위인 **reranker(CrossEncoder, `bge-reranker-v2-m3` 우선
  검토)**로 이동.

## 2026-08-25 — ef_search 스윕, 디스크 정리, ColBERT 배제 결정 → reranker(CrossEncoder) 착수

- **`run_eval.py`에 `--k-list`/`--ef-search` 옵션 추가**: reranker에 넘길 후보 pool 크기를
  정하려면 recall@30/40/50까지 봐야 하는데 k_list가 하드코딩(1,5,10,20)돼있었음 —
  CLI에서 바꿀 수 있게 수정. ef_search도 같은 이유로 옵션화.
- **정식(7,280개) recall@k 확장 측정 (pgvector, dense 단독)**:

  | k | recall@k (ef=100) | recall@k (ef=200) | recall@k (ef=300) |
  |---|---|---|---|
  | 1 | 0.6710 | 0.6768 | 0.6780 |
  | 5 | 0.8902 | 0.8975 | 0.8990 |
  | 10 | 0.9316 | 0.9386 | 0.9401 |
  | 20 | 0.9565 | 0.9635 | 0.9651 |
  | 30 | 0.9657 | 0.9725 | 0.9742 |
  | 40 | 0.9710 | 0.9779 | 0.9794 |
  | 50 | 0.9745 | 0.9813 | 0.9834 |
  | mrr@10 | 0.7645 | 0.7707 | 0.7720 |
  | latency mean | 11.66ms | 19.47ms | 24.93ms |

  - ef_search 100→200 구간은 정확도 개선이 뚜렷(recall@1 +0.58%p)한데 200→300은 수확체감
    (recall@1 +0.12%p)이면서 latency만 계속 늘어남(19.5→24.9ms) → **ef_search=200을
    가성비 지점으로 채택**.
  - recall@20→30 구간 +0.9%p, 30→40 +0.5%p, 40→50 +0.35%p로 이득이 급격히 줄어듦 →
    **reranker 후보 pool은 top-20~30이면 충분**하다고 판단(그 이상은 비용 대비 이득 작음).
- **디스크 점검 (여유 공간 2.3GB로 위험 수준)**: 프로젝트 내부엔 예전에 이미 정리해둔 덕에
  낭비되는 파일이 없음을 재확인(`cache_embeddings`엔 최종 모델 캐시 1개+결과 json만,
  `finetune/output`엔 최종 모델 1개만, Postgres엔 `chunks_300_overlap100` 테이블 하나만
  존재). 32GB 중 실사용은 Postgres DB(15GB)+venv(5.6GB)+VSCode 서버(3.4GB, 프로젝트
  무관)+파인튜닝 모델(2.2GB)+코퍼스 캐시(1.6GB)로 전부 필요한 것들이었음.
  - **정리**: `eval_model.py`용 코퍼스 임베딩 캐시(.npy, 1.6GB) 삭제 — dense 단독이
    pgvector 기준으로 이미 확정돼서 exact 재검증 캐시가 더 필요 없다고 판단. 여유 공간
    2.3GB→**3.9GB**로 회복.
- **⚠️ ColBERT 배제 결정**: reranker 후보로 cross-encoder와 ColBERT 두 계열을 검토하던 중,
  ColBERT는 chunk당 벡터 1개가 아니라 **토큰 수만큼** 벡터를 저장해야 한다는 구조적 특성
  확인. 어림 계산: 823,763 chunk × 토큰 ~150개 × 128차원 × 4바이트 ≈ **순수 벡터만 최소
  55~60GB** (압축해도 5~10GB대까지가 한계). 지금 여유 공간(3.9GB)로는 어떤 압축을 걸어도
  불가능하고, 이 인스턴스 디스크(32GB)는 vast.ai 생성 시 고정이라 컨테이너 안에서 늘릴
  방법도 없음 — **ColBERT는 이 환경에서 스토리지 예산 밖으로 처음부터 배제.**
- **cross-encoder로 확정**: cross-encoder는 문서를 미리 인코딩해서 저장하는 구조가 아니라
  (쿼리+후보를 매번 그 자리에서 같이 인코딩) 코퍼스 크기와 무관하게 **추가 디스크가
  전혀 필요 없음** — 모델 가중치(`bge-reranker-v2-m3` 기준 1~2GB, 1회성 다운로드)만 있으면
  됨. 대신 후보 개수에 비례해 쿼리당 느려지므로, 위에서 정한 top-20~30 후보 pool과 궁합이
  맞음.
- **다음**: `bge-reranker-v2-m3`로 cross-encoder reranker 구현 → `server/eval/retrievers.py`에
  Reranker 클래스 추가(top-20~30 pgvector(ef_search=200) 후보를 CrossEncoder로 재정렬) →
  소규모 샘플 검증 후 정식 평가.

## 2026-08-25 — CrossEncoderReranker 구현+평가, off-the-shelf reranker가 dense보다 나쁨 → 파인튜닝 착수

- **`CrossEncoderReranker` 구현** (`server/eval/retrievers.py`): 1단계(pgvector/HNSW)로
  top_n(기본 30)개 후보를 뽑고 2단계(cross-encoder)로 (query, chunk_text) 쌍을 재점수·재정렬.
  `run_eval.py --mode rerank` 추가, `harness.py`의 배치 단위 tqdm이 rerank처럼 느린 모드엔
  갱신 간격이 너무 뜸해서 `CrossEncoderReranker.retrieve_batch` 안에 쿼리 단위 tqdm(leave=False)
  한 겹 더 추가.
- **정식(7,280개) 평가 — 예상 밖으로 dense 단독보다 나쁨**: `bge-reranker-v2-m3`(범용,
  한국어 법률 도메인 미세조정 안 됨)로 돌린 결과, 같은 ef_search=100 기준:

  | | dense 단독(ef=100) | rerank(top_n=30, ef=100) |
  |---|---|---|
  | recall@1 | 0.6710 | 0.6308 (-4.02%p) |
  | recall@5 | 0.8902 | 0.8663 (-2.39%p) |
  | recall@10 | 0.9316 | 0.9188 (-1.28%p) |
  | recall@20 | 0.9565 | 0.9526 (-0.39%p) |
  | recall@30 | 0.9657 | 0.9657 (동일 — 구조상 당연, pool 안 순서만 바뀜) |
  | mrr@10 | 0.7645 | 0.7305 (-3.40%p) |
  | latency | 11.66ms | 421.04ms (36배) |

  recall@30이 정확히 일치하는 건 reranker가 pool 안에서 순서만 바꾸고 pool 자체(recall
  상한)는 못 바꾼다는 구조적 사실과 일치 — 구현이 맞다는 sanity check. 문제는 recall@1/5/10/20이
  전부 dense보다 나빠졌다는 것 — 범용 cross-encoder가 이미 도메인 파인튜닝된 dense의 1등
  픽을 오히려 밀어내는 경우가 더 많았다는 뜻. sparse+hybrid 때와 같은 패턴("도메인 파인튜닝
  안 된 범용 방법이 이미 강한 도메인 dense를 못 이김")으로 추정.
- **결정: cross-encoder도 도메인 파인튜닝 시도** — KURE-v1 dense 모델을 base 위에 이어서
  학습해 +10%p 얻었던 것과 같은 전략을 reranker에도 적용. 새 스크립트 2개 작성(기존
  `prepare_data.py`/`train.py`/`mine_hard_negatives.py`는 안 건드리고 별도 파일로 분리):
  - **`server/finetune/prepare_reranker_data.py`(신규)**: `prepare_data.py`+
    `mine_hard_negatives.py`를 합친 것과 비슷하지만, 코퍼스 전체를 로컬 .npy 캐시로 GPU에
    올리는 대신 **pgvector(HNSW)를 그대로 사용** — 코퍼스를 로컬로 다시 끌어올 필요가
    없어서 디스크 추가 소요 0(지금 디스크 여유로는 예전 방식의 로컬 corpus 캐시를 감당
    못 함). 또한 hard negative를 Phase A가 아니라 **지금 서빙 중인 최종 모델
    (kure-v1-finetuned-hard)** 기준으로 채굴 — 실제 서빙에서 reranker가 보게 될 후보
    분포를 반영. (anchor=query, positive=정답 문서 내 최유사 chunk, negative_chunks=
    hard negative 여러 개) triplet을 `reranker_pairs.jsonl`에 저장.
  - **`server/finetune/train_reranker.py`(신규)**: `bge-reranker-v2-m3`를 base로
    `CrossEncoderTrainer`+`MultipleNegativesRankingLoss`(dense Phase A/B와 같은 InfoNCE
    계열)로 이어서 학습. **⚠️ 디스크 안전장치**: base 모델 로드 직후(가중치가 이미
    메모리에 다 올라간 시점) HF 캐시 폴더를 바로 삭제 — 안 그러면 학습 완료 후 파인튜닝
    모델(~2.2GB) 저장 시점에 base 캐시(~2.2GB)와 동시에 존재해야 해서 도합 4.4GB 필요한데
    지금 여유(<2GB)로는 불가능함. `huggingface_hub.scan_cache_dir()`로 정확한 캐시 경로를
    찾아서 삭제.
  - 15개 샘플로 두 스크립트 다 end-to-end 검증 완료(데이터 생성 → 학습 3스텝 → 저장,
    디스크 안전장치가 실제로 2.29GB 회수하는 것도 확인) — 테스트 산출물은 정리함.
- **다음**: `prepare_reranker_data.py`(train_query.json 41,319개 전체) → `train_reranker.py`
  전체 실행(사용자) → 파인튜닝된 reranker로 `run_eval.py --mode rerank --rerank-model-path
  ./output/bge-reranker-v2-m3-finetuned` 재평가해서 off-the-shelf 대비, dense 단독 대비
  개선 여부 확인.
