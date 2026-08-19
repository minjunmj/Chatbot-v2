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
