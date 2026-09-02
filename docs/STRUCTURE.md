# 프로젝트 폴더/파일 구조

새 세션에서 "이 파일이 뭐하는 거였지?"가 헷갈릴 때 참고하는 용도. 각 결정의 근거/실험
결과는 여기 말고 [log.md](log.md)에 있음 — 이 문서는 "무엇이 어디 있는지"만 다룸.

---

## 최상위

| 경로 | 용도 |
|---|---|
| `README.md` | 프로젝트 최상위 소개 (무엇을 하는지, 아키텍처, 실행 방법) |
| `data/` | 원본 코퍼스 + train/val 쿼리셋 (git에는 안 올라감, `.gitignore`의 `/data/`) |
| `docs/` | 프로젝트 문서 (이 파일 포함) |
| `scripts/` | 원본 데이터 다운로드/전처리 스크립트 (AI Hub 판례 데이터 → DB_data/train/val 생성) |
| `server/` | 모든 서버 사이드 코드 (검색/생성 파이프라인, 파인튜닝, 배포) |

---

## `data/`

| 경로 | 용도 |
|---|---|
| `DB_data/*.json` | 검색 대상 원본 판례 코퍼스, 44,700건. 파일명이 곧 사건번호(`{case_no}.json`) |
| `Train/train_query.json` | 임베딩/reranker 파인튜닝용 (query, case_ids) 쌍, 41,319개 |
| `Val/val_query.json` | 검색 성능 평가용 (query, case_ids) 쌍, 7,280개 — 모든 recall@k 실험의 기준 |
| `Val/val_query_by_type.json` | val_query.json에서 사건유형(22개)별로 최대 10개씩 뽑은 표본, 171개. `build_typed_test_set.py` 산출물 — 사건유형별 답변 품질 비교(full vs chunks judge 비교)에 사용 |

---

## `scripts/`

AI Hub 원본 데이터 → `data/` 최종 형태로 만드는 일회성 전처리 파이프라인.

| 파일 | 용도 |
|---|---|
| `download_aihub_data.sh` | AI Hub 판례 데이터셋 다운로드 |
| `extract_zip_files.py` | 압축 해제 |
| `merge_aihub_parts.py` | 분할된 파일 병합 |
| `build_train_val_dataset.py` | 최종 DB_data/Train/Val 데이터셋 생성 (v3) |

---

## `docs/`

| 파일 | 용도 |
|---|---|
| `PROJECT_STATE.md` | **새 세션 시작 시 제일 먼저 읽는 파일.** 현재 상태 요약 + 다음 계획 |
| `DECISIONS.md` | 전체 프로젝트 흐름 요약 — 각 단계 후보/결정/이유만 (세부 수치 제외, log.md가 너무 길어서 큰 그림만 보려고 만든 문서) |
| `log.md` | 날짜순 작업 로그 — 모든 실험 결과, 원인 분석, 결정 근거의 원본 기록 |
| `data_preprocessing_log.md` | 데이터 전처리 단계(`scripts/`) 관련 기록 |
| `STRUCTURE.md` | 이 파일 |

---

## `server/service/` — 실제 배포되는 서비스 코드

지금 `lexchatbot` supervisor 서비스가 실행하는 대상. 여기 안 있는 건 전부 서비스 실행에
직접 필요하지 않음(연구/실험/일회성 구축 스크립트).

| 파일 | 용도 |
|---|---|
| `api.py` | FastAPI 앱 — `/health`, `/ask`(POST, 질문→답변), `/`(간단한 HTML 데모 페이지). 서버 기동 시 모델을 한 번만 로드 |
| `pipeline.py` | **최종 채택된 구성만 남긴** RAG 파이프라인 — `load_models`, `retrieve_top_k`, `build_context`(chunks 방식 하나만), `build_prompt`(원복된 단순 버전 하나만), `generate_answer`(reasoning mode 없음, 일반 모드만), `answer_query`(요청 1건 처리하는 진입점). 비교했던 다른 선택지(문서 전체 컨텍스트, reasoning mode, 개선 시도했다 되돌린 프롬프트)는 전부 뺐음 — 그 버전들은 `server/research/generate.py`에 남아있음 |

---

## `server/eval/` — 검색(retriever) 평가 인프라 (공유 라이브러리)

`service/pipeline.py`, `research/generate.py` 양쪽이 공통으로 가져다 쓰는 core
라이브러리라 따로 안 옮김.

| 파일 | 용도 |
|---|---|
| `harness.py` | `evaluate(retriever, val_data)` — retriever 종류 안 가리고 recall@k/mrr@10/latency 계산하는 공통 로직 |
| `retrievers.py` | Retriever 구현체 모음: `DenseExactRetriever`(brute-force), `PgvectorRetriever`(dense, HNSW), `SparseRetriever`/`HybridRetriever`(BM25+dense, **미채택, 코드는 보존**), `CrossEncoderReranker`(**최종 채택**, dense top-N을 cross-encoder로 재정렬) |
| `run_eval.py` | 위 harness+retrievers를 엮은 CLI (`--mode exact\|pgvector\|sparse\|hybrid\|rerank`) — chunk_size/모델/reranker 조합 바꿔가며 실험할 때 씀 |
| `calibrate_threshold.py` | reranker cross-encoder 점수로 "관련 판례 없음"을 걸러낼 threshold를 정하기 위한 실측 스크립트 — val_query.json 전체의 recall@5 hit/miss와 top1_score를 기록(`threshold_calibration.jsonl`, gitignore) |

---

## `server/finetune/` — 임베딩/reranker 도메인 파인튜닝

| 파일 | 용도 |
|---|---|
| `prepare_data.py` | Phase A(in-batch negative)용 (query, positive_chunk) 쌍 생성 |
| `mine_hard_negatives.py` | Phase B(hard negative)용 데이터 생성 — Phase A 모델 기준으로 헷갈리는 negative 채굴 |
| `train.py` | KURE-v1 dense 임베딩 모델 파인튜닝 (Phase A/B 자동 분기) → `output/kure-v1-finetuned-hard/` |
| `prepare_reranker_data.py` | cross-encoder reranker 파인튜닝용 (anchor, positive, negative) triplet 생성 — pgvector 기반, 로컬 corpus 캐시 불필요 |
| `train_reranker.py` | `bge-reranker-v2-m3` reranker 파인튜닝 → `output/bge-reranker-v2-m3-finetuned/` |
| `eval_model.py` | dense 모델(base/파인튜닝) exact-search 평가 (참고용으로 보존, `run_eval.py --mode exact`와 같은 역할) |
| `output/kure-v1-finetuned-hard/` | **최종 채택된 dense 임베딩 모델** (git 제외, 로컬에만 존재) |
| `output/bge-reranker-v2-m3-finetuned/` | **최종 채택된 reranker 모델** (git 제외) |

---

## `server/db/` — Postgres DB 구축

| 파일 | 용도 |
|---|---|
| `schema.sql` | `make_chunk_table(suffix)` — chunk 테이블 동적 생성 함수 |
| `build_vector_db.py` | 코퍼스를 chunk로 쪼개서 임베딩 계산 후 DB 적재 (`chunks_300_overlap100`, 최종 운영 테이블) |
| `test_val_pgvector.py` | 초기 pgvector 검증 스크립트 (참고용으로 보존, retrievers.py의 `PgvectorRetriever`가 같은 로직의 재사용 가능 버전) |

---

## `server/research/` — Phase 3(생성) 일회성 평가/실험 스크립트

서비스 실행에는 필요 없음. 결과 `.jsonl`들은 전부 gitignore(핵심 수치는 log.md에 정리돼있음).

| 파일 | 용도 |
|---|---|
| `generate.py` | RAG 파이프라인 **실험용 전체 버전** — `service/pipeline.py`와 달리 비교했던 모든 선택지(문서 전체 vs chunks 컨텍스트, 일반 vs reasoning mode, 개선 시도했다 되돌린 프롬프트 v2)를 다 갖고 있음. 이 폴더의 나머지 스크립트들이 여기서 함수를 가져다 씀 |
| `build_typed_test_set.py` | val_query.json에 사건유형(DB `case_type` 컬럼)을 붙여 유형별 표본(`val_query_by_type.json`) 생성 |
| `judge_compare.py` | 쿼리 1개로 full/chunks 컨텍스트 방식을 GPT-5-mini judge로 비교(단발 테스트용) |
| `run_judge_compare.py` | 위를 배치로 — 사건유형별 표본 171개 전체 비교, resume 지원 → `judge_results.jsonl` |
| `check_miss_rejection.py` | threshold를 통과한 miss 쿼리에서 EXAONE이 "관련 판례 없음"을 스스로 인지하는지 키워드 패턴으로 확인 → `miss_rejection_results.jsonl` |
| `eval_answer_quality.py` | hit 100개+miss 100개를 GPT-5-mini로 "라벨이 아니라 실제 top-5 내용 기준" 관련성/정답 여부 평가 → `answer_quality_results.jsonl` |
| `reeval_rubric_v2.py` | 위 결과를 rubric 세분화(`grounded`+`resolves_question` 분리)해서 재평가, 답변 재생성 없이 GPT 판정만 다시 함 → `answer_quality_results_v2.jsonl` |
| `eval_prompt_v2.py` | 개선 프롬프트로 195개 답변을 다시 생성+평가 (결과: 실패, 원복됨) → `prompt_v2_results.jsonl` |

---

## `server/cache_embeddings/` — 초기 실험 결과 기록 (근거 자료)

2026-08-18 전후 임베딩 모델/청킹 비교 실험 때 나온 작은 결과 요약 JSON들. 전부 1KB 안팎,
log.md의 초기 항목들이 참조하는 근거 자료라 보존.

| 파일 | 관련 실험 |
|---|---|
| `results.json` | 임베딩 모델 선정 실험 (model_test.py, 삭제됨) |
| `jhgan-ko-sbert-sts_chunk200_results.json` | jhgan 모델 chunk 실험 (jhgan.py, 삭제됨) |
| `kure-v1_chunk_results.json` | KURE-v1 chunk 크기별 isolate 실험 (kure_chunk.py, 삭제됨) |
| `kure_chunk_overlap_results.json` | overlap 실험 (kure_chunk_overlap.py, 삭제됨) |
| `bge-m3-hybrid_results.json` | BGE-M3 dense+sparse hybrid 실험 (bge_hybrid.py) |

---

## 그 외 `server/` 최상위

| 파일 | 용도 |
|---|---|
| `bge_hybrid.py` | 최초의 dense+sparse hybrid 실험 스크립트(2026-08-18, BGE-M3 기반). 이후 Postgres tsvector+Kiwi 방식으로 대체됐지만 참고용으로 보존 |
| `requirements.txt` | 파이썬 의존성 — 핵심 파이프라인/sparse/파인튜닝/서비스/연구용으로 구분해서 정리돼있음 |

---

## 삭제된 것들 (참고 — git 히스토리에만 존재)

`model_test.py`, `jhgan.py`, `kure_chunk.py`, `kure_chunk_overlap.py`, `test_val_pgvector.py`
일부, 각종 `finetune_pairs*.jsonl`/`reranker_pairs.jsonl`/코퍼스 임베딩 `.npy` 캐시 —
전부 **재생성 가능한 파생 산출물**이라서 삭제됨(수치 결과는 log.md에 보존). "왜 지웠는지"는
각 항목이 언급된 log.md 날짜를 참고.
