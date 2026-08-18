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
