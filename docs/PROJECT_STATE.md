# LexChatbot RAG 프로젝트 — 현재 상태 (여기부터 읽기)

> **새 세션(새 서버) 시작 시 이 파일 하나만 먼저 읽으면 됨.** 사용자가 프로젝트를 처음부터
> 다시 설명할 필요 없이, 이 문서 + [log.md](log.md)만 보고 바로 이어서 작업할 것.
>
> 갱신 규칙: 계획 하나가 끝나면 → ① [log.md](log.md)에 날짜/내용/근거 기록 ② 이 파일의
> "1. 지금 당장 상태"와 "5. 다음 계획"을 최신으로 고쳐쓸 것. 이 두 가지를 안 하면 다음
> 세션이 다시 헤맨다.

최종 갱신: 2026-08-21

---

## 0. 프로젝트 목표

**국내 법률 판례 데이터를 검색(RAG)해서, 사용자의 법률 질문에 관련 판례를 근거로 답해주는
챗봇**을 만든다.

- 더 큰 목적: 이 프로젝트를 **LLM Application Engineer 포트폴리오**로 완성하는 것.
  즉 "그냥 되는 것"이 아니라 각 선택(모델/chunking/DB/평가 방식 등)에 **정량적 근거**를
  남기고, 실무 수준(백엔드/운영/평가)까지 다뤄야 발표·이력서 자료로서 가치가 있음.
- 사용자가 서버를 그때그때 대여해서 작업 → 세션이 계속 끊기고 새로 시작됨. 그래서 이 문서가
  "프로젝트 기억"을 대신함.

---

## 1. 지금 당장 상태 (요약)

| 구분 | 상태 |
|---|---|
| 데이터 파이프라인 (parsing/cleaning/split) | ✅ 완료 — `scripts/build_train_val_dataset.py` v3 |
| DB_data (검색 대상 원본 코퍼스) | ✅ 44,700건 (`data/DB_data/*.json`) |
| Train/Val 쿼리셋 | ✅ `data/Train/train_query.json`, `data/Val/val_query.json` |
| Chunking / Metadata 설계 | ✅ **최종 확정: chunk_size=300, overlap=100 (글자 단위), KURE-v1 임베딩.** chunk200(overlap 없음) → chunk200_overlap50/chunk300_overlap100 overlap 실험 → 둘 다 pgvector 실측까지 마치고 chunk300_overlap100 채택 (정확도 -1%p 안팎 손해, 용량 -27% 이득, latency 무승부; 근거: [log.md](log.md) 2026-08-20) |
| 임베딩 모델 선정 실험 (no-chunk) | ✅ 완료 — `server/model_test.py`, **KURE-v1 선정** (결과: [log.md](log.md) 2026-08-18) |
| Chunk 기반 retrieval 실험 (jhgan) | ✅ 완료 — `server/jhgan.py`(chunk_size=200), no-chunk KURE-v1보다 전 지표 우세하나 **모델+chunking 동시 변경이라 confound 있음** (결과: [log.md](log.md) 2026-08-18) |
| Hybrid(dense+sparse) 실험 (BGE-M3) | ✅ 완료 — `server/bge_hybrid.py`, hybrid가 dense/sparse 단독보다 뚜렷이 우세하나 KURE-v1 no-chunk엔 못 미침 (결과: [log.md](log.md) 2026-08-18) |
| Chunking 효과 isolate 실험 (KURE-v1 chunk) | ✅ 완료 — `server/kure_chunk.py`(200/500/1000자), **confound 해소: chunking이 지배적 요인**, chunk200이 압도적 1위(recall@1 0.5541, mrr@10 0.661)이나 비용(823,763 chunk, encode 2시간)도 최대 (결과: [log.md](log.md) 2026-08-18) |
| Vector DB 구축 | ✅ **1차 완료** — FAISS 대신 PostgreSQL+pgvector 채택, Postgres 16 + supervisor 서비스 등록. `chunks_300_overlap100`(823,763행, 11GB, HNSW 인덱스)이 정식 서빙용 테이블로 확정·유지 중. `server/db/schema.sql`+`build_vector_db.py`+`test_val_pgvector.py`는 chunk_size/overlap 임의 조합 지원하도록 일반화 완료 |
| 임베딩 모델 도메인 파인튜닝 | ✅ **완료** — Phase A(in-batch) + Phase B(hard negative, `--skip-top 5`로 false negative 수정 후) 최종 모델 확정. **base 대비 recall@1 +10.0%p(상대 +17.2%), mrr@10 +9.17%p**. 결과 모델: `server/finetune/output/kure-v1-finetuned-hard/`. 근거: [log.md](log.md) 2026-08-21 |
| Retriever 평가 자동화 | ✅ 완료 — `server/eval/`(`harness.py`+`retrievers.py`+`run_eval.py`) 신설. `DenseExactRetriever`/`PgvectorRetriever` 구현, sparse/hybrid/rerank 추가 시 retrievers.py에 클래스만 추가하면 됨. 기존 임시 스크립트 7개는 대체 가능(수치는 log.md에 보존, 삭제 여부는 사용자 판단) — [log.md](log.md) 2026-08-20 |
| RAG 답변 생성 파이프라인 | ❌ 시작 전 (v1 코드 있으나 재사용 여부 미결정, 아래 2절 참고) |
| 백엔드/배포/운영 | ❌ 시작 전 (v1 코드 있음, 아래 2절 참고) |

**한 줄 요약**: 데이터 준비, 임베딩 모델(KURE-v1) 선정, chunking 전략(chunk_size=300/overlap=100),
Vector DB 구축(pgvector), **도메인 파인튜닝(Phase A in-batch + Phase B hard negative)까지
전부 완료.** base KURE-v1 대비 recall@1 +10.0%p(상대 +17.2%), mrr@10 +9.17%p — 이 프로젝트의
핵심 retriever 파이프라인 1차 완성. 재사용 가능한 평가 하니스(`server/eval/`)도 구축해서
이후 실험은 새 스크립트 없이 `run_eval.py`로 진행 가능. **다음은 sparse+dense hybrid,
reranker, MMR, 메타데이터 필터 결합 중 우선순위 정해서 진행** — 상세는 5절 참고.

---

## 2. v1-baseline (이전 시도 — 폐기하고 처음부터 다시 하는 중, 참고용으로만 보존)

### 왜 v1을 버리고 새로 시작하는가

v1은 GPT에게 그때그때 물어보면서 모델·파라미터(임베딩 모델, chunk 크기, MMR λ, rerank
임계치 등)를 별다른 검증·비교 없이 대충 정하고, 이른바 "바이브 코딩"으로 빠르게 완성만 시킨
결과물이었음. 동작은 했지만:
- 왜 그 임베딩 모델/파라미터를 골랐는지 근거가 없음 (비교 실험 자체가 없었음)
- 프로젝트 전체의 설계·구조가 없이 기능을 붙여나간 형태
→ 그래서 "동작하는 데모"는 될 수 있어도, **실제 기업 지원용 포트폴리오로서 실무 수준의
근거(왜 이 선택을 했는가, 어떻게 검증했는가)를 보여주기엔 부족**하다고 판단해서 v2로
처음부터 다시 시작함. 지금 데이터 전처리부터 근거를 남기고(§1), 임베딩 모델도 임의로
고르지 않고 정량 비교(model_test.py) 중인 것도 이 이유 때문임.

`server/main.py`, `server/acc_test.py`, `README.md`, `lib/main.dart`(Flutter 앱),
`Dockerfile`/`docker-compose.yml` 등 v1 결과물은 **저장소에서 삭제됨** (2026-08-17,
[log.md](log.md) 참고). 지금 작업 트리에는 없고, git 히스토리의 `v1-final` 태그
(`git show v1-final:server/main.py` 식으로 조회 가능)에서만 볼 수 있음. 재사용하려면
그 태그에서 필요한 파일만 `git checkout v1-final -- <path>`로 꺼내 쓸 것.

v1이 실제로 구현했던 것 (재사용 검토 대상 — 코드는 v1-final 태그에 있음):
- FastAPI 서버 + Flutter 모바일 앱 (음성 질문 → Whisper STT → 답변)
- FAISS + fine-tuned HuggingFace 임베딩(`model_bs32`) + BM25 앙상블 실험(acc_test.py)
- 3단계 검색: FAISS top-50 → MMR(λ=0.9) 다양성 10개 → CrossEncoder(`bge-reranker-v2-m3`) rerank top-3
- LLM 라우터(RAG vs DIRECT 분류, GPT-4o-mini) + 세션별 대화 히스토리
- Docker + Cloudflare Tunnel 배포

---

## 3. 로드맵 (사용자 초안 + 검토/보완)

사용자가 처음 짠 5단계 로드맵을 검토했음. 전체 구조(데이터→retriever→generation→backend→배포)는
LLM App Engineer 프로젝트로서 합리적인 순서. 아래는 **원안 그대로 두고, 빠진 부분만
"➕ 추가 제안"으로 표시**한 버전. ➕ 항목은 근거를 옆에 짧게 달아둠 — 전부 다 할 필요는 없고
발표 시간/서버 사정에 맞춰 선택하면 됨.

### Phase 1 — 데이터 + 평가 환경 구축
- [x] 프로젝트 기준선 고정 — v1 코드/구조는 위 2절에 기록해둠
- 데이터 파이프라인
  - [x] Parsing
  - [x] Cleaning
  - [x] Chunking — chunk_size=300, overlap=100 확정 ([log.md](log.md) 2026-08-20)
  - [x] Metadata — 사건번호/사건명/법원명/선고일자/사건종류명만 포함하기로 확정 (`server/db/schema.sql`)
  - [x] Embedding / Vector DB — KURE-v1 + PostgreSQL/pgvector, `chunks_300_overlap100` 서빙 중
  - ✅ **Vector DB 재선정 완료**: FAISS 대신 pgvector로 통합 (메타데이터 필터+벡터검색 결합, Phase 4 Postgres 계획과 인프라 통합)
- Retriever 평가 환경 구축
  - [x] Query–정답 판례 평가셋 (Val 쿼리셋은 이미 있음, 재사용 가능)
  - [x] Recall@k, MRR
  - [x] 평가 자동화 — `server/eval/`(harness.py+retrievers.py+run_eval.py)로 완료.
    Retriever 인터페이스만 맞으면 corpus/query/검색방법 바꿔 끼우기만 하면 됨
- [x] Retriever Baseline 구축 (최초 검색 정확도 측정) — model_test.py 결과가 이 역할

### Phase 2 — Retriever 성능 개선 (검색 정확도만 집중)
- [ ] Retriever 성능 개선
- [ ] 여러 실험 결과 비교
- [ ] 최종 Retriever 선정
- ➕ **Hybrid Search (BM25 + Dense)**: v1의 acc_test.py에 이미 시도 흔적 있음. 법률 판례는
  법조문 번호·고유명사처럼 키워드 매칭이 중요한 도메인이라 dense 단독보다 잘 나올 가능성 높음.
- ➕ **Reranker 도입 (CrossEncoder)**: v1에서 이미 검증됨(`bge-reranker-v2-m3`). 순수 dense
  대비 성능 향상 폭을 정량적으로 비교해두면 "왜 rerank를 썼는지" 발표 근거가 됨.
- ✅ **임베딩 모델 도메인 파인튜닝 완료**: `train_query.json` 기반 contrastive fine-tuning
  (in-batch → hard negative 2단계). base 대비 recall@1 +10.0%p, mrr@10 +9.17%p
  ([log.md](log.md) 2026-08-20~21).
- ✅ **Chunking 전략 실험 완료**: chunk_size 200/500/1000 → overlap 실험(200_overlap50,
  300, 300_overlap100) → pgvector 실측까지 거쳐 **chunk_size=300/overlap=100 확정**
  ([log.md](log.md) 2026-08-19~20).

### Phase 3 — RAG 답변 품질 개선
- RAG Generation Pipeline 구축
  - [ ] Context 구성
  - [ ] Prompt
  - [ ] LLM 답변 생성
  - [ ] 근거/판례 반환
  - ➕ **오케스트레이션 방식 결정**: v1은 LangChain 사용. 그대로 갈지, 커스텀 파이프라인으로
    갈지(디버깅/제어가 쉬움) 결정해두기.
  - ➕ **Prompt 버전 관리**: prompt를 코드에 하드코딩하지 말고 버전 남기기 (A/B 비교 근거 필요)
- RAG 답변 품질 평가
  - [ ] 답변 정확성
  - [ ] Groundedness
  - [ ] Citation 정확성
  - [ ] 답변 불가능한 질문 처리
  - ➕ **평가 방법/도구 결정**: "무엇으로 어떻게 채점할지"가 비어있음 — LLM-as-judge(자체
    rubric) vs RAGAS/DeepEval 같은 기존 프레임워크 중 택1 필요. 사람이 매번 눈으로 보는 건
    확장 안 됨.

### Phase 4 — 실무형 Backend / 운영 구조 구축
- Backend 서비스 구조 정리
  - [ ] FastAPI / Router·Service 분리 / Request·Response schema / 예외 처리
  - ➕ **Streaming 응답 (SSE)**: 실제 LLM 서비스는 답변을 토큰 단위로 흘려보내는 게 표준 UX.
    v1은 job polling 방식만 있었음 — 챗봇이면 streaming 하나쯤은 붙여보는 게 포트폴리오 임팩트 큼.
  - ➕ **테스트 코드**: 최소한 retriever/생성 파이프라인에 대한 unit/integration test 몇 개.
    "재현 가능한 서비스"라는 인상을 주는 데 중요.
- Logging + Observability — 원안 충분함 (request ID, 검색 결과, 모델 호출, error log, trace)
- Latency 측정 및 개선 — 원안 충분함
- Reliability — 원안 충분함
  - ➕ **Rate limiting**: 사용자별/IP별 요청 제한 (LLM 비용 통제 + 실무 표준 관행)
- Safety — 원안 충분함
- DB / 서비스 상태 관리 — 원안 충분함 (PostgreSQL/session/대화기록/feedback/Redis 검토)

### Phase 5 — 배포 + 운영 + 최종 정리
- Docker화 / 배포(AWS) / Monitoring / CI/CD — 원안 충분함 (v1에 Docker+Cloudflare 경험 있어 재활용 가능)
- ➕ **데모 UI 우선순위 결정**: v1의 Flutter 앱은 무거움(빌드/배포 손 감). 발표·시연용으로는
  Streamlit/Gradio 같은 가벼운 웹 데모를 먼저 붙이고, Flutter 앱은 시간이 남으면 마무리하는
  순서를 추천. "동작하는 걸 빨리 보여주는 것"이 검색 정확도 실험만큼 중요.
- 최종 Evaluation + 문서화 — 원안 충분함, 이 문서 + log.md가 그 근거 자료가 됨

---

## 4. 로드맵 전체에 대한 검토 의견 (LLM App Engineer 관점)

원래 5단계 리스트는 RAG 챗봇 프로젝트로서 **빠진 큰 축은 없음** (데이터→검색→생성→백엔드→배포
흐름이 실무 그대로). 위 ➕ 표시들은 전부 "있으면 더 좋은" 보완이지, 없으면 프로젝트가 안 되는
건 아님. 우선순위를 매기자면:

1. **꼭 하면 좋음** (임팩트 대비 비용 낮음): Hybrid search, Reranker, 평가 하니스 재사용화,
   Streaming, 평가 방법(프레임워크) 결정
2. **시간 되면**: 임베딩 파인튜닝, Prompt 버전관리, Rate limiting, 테스트 코드
3. **발표 전 마지막에**: 가벼운 데모 UI 우선 (Flutter는 선택)

---

## 5. 다음 계획 (Next Step)

**지금 할 일**: 검색 파이프라인의 다음 개선 요소를 우선순위 정해서 진행 — 후보와 우선순위는
아래 참고. Retriever 자체(chunking + 임베딩 파인튜닝)는 완료됨.

**Chunking 최종 결정 요약** (완료, 2026-08-19~20, 상세 근거는 [log.md](log.md)):
chunk200 → chunk1000 → chunk200_overlap50 → chunk300_overlap100까지 exact search + pgvector
실측을 거쳐 **chunk_size=300, overlap=100** 확정. chunk200_overlap50과 막판 경합했는데
정확도 차이(1%p 안팎)보다 DB 용량 27% 절감(15GB→11GB)의 실익이 크다고 판단해 채택.

**임베딩 파인튜닝 최종 결정 요약** (완료, 2026-08-20~21, 상세 근거는 [log.md](log.md)):
Phase A(in-batch negative) → Phase B(hard negative, Phase A 모델로 채굴 후 Phase A 위에
이어서 학습) 순서로 진행. Phase B 1차 시도는 유사도 최상위 negative를 그대로 써서 오히려
Phase A보다 나빴음(false negative 추정) — `mine_hard_negatives.py --skip-top 5`(상위 5개
건너뛰기)로 수정 후 재실행해서 확실히 개선됨. **최종 모델**: `server/finetune/output/kure-v1-finetuned-hard/`
— base 대비 recall@1 +10.0%p(상대 +17.2%), mrr@10 +9.17%p.

**다음 개선 후보** (2026-08-20 논의, query rewriting은 스코프 제외 결정):
1. **sparse+dense hybrid** — pgvector에 tsvector/GIN 컬럼만 추가하면 되는 additive 작업
   (아래 참고), RRF로 dense 결과와 결합. 법률 판례는 법조문 번호·고유명사 등 키워드 매칭이
   중요한 도메인이라 효과 기대됨(BGE-M3 hybrid 실험에서 sparse 단독도 준수했던 전례 있음,
   2026-08-18 log.md)
2. **MMR(다양성 재정렬)** — v1에 있었던 요소(FAISS top-50→MMR→rerank top-3)인데 v2엔 아직
   없음. chunk 단위 검색이라 같은/비슷한 chunk가 상위권을 도배할 수 있어 v1보다 오히려 더
   필요할 수 있음
3. **Reranker(CrossEncoder)** — 기성 모델(`bge-reranker-v2-m3`, v1에서 검증됨) 우선 적용,
   여유 있으면 도메인 파인튜닝도 고려
4. **메타데이터 필터 결합** — "형사 사건 중 2015년 이후" 같은 구조화 질의 처리 (SelfQuery
   방식, LLM으로 필터+의미쿼리 분리 — 초반 논의 참고). 정확도보다는 실사용 기능 완성도 항목
5. **평가는 전부 `server/eval/run_eval.py`로** — 새 검색 방법마다 `retrievers.py`에
   Retriever 클래스만 추가하면 됨 (2026-08-20 하니스 신설, 상세는 아래)

**평가 하니스** (완료, 2026-08-20, [log.md](log.md) 참고):
`server/eval/`(harness.py+retrievers.py+run_eval.py) — `DenseExactRetriever`(pgvector 없이
빠른 스크리닝), `PgvectorRetriever`(실제 서빙 latency 포함) 구현됨. 기존 임시 스크립트 중
`model_test.py`/`jhgan.py`/`kure_chunk.py`/`kure_chunk_overlap.py`는 삭제됨(결과는 log.md에
보존). `bge_hybrid.py`/`test_val_pgvector.py`/`finetune/eval_model.py`는 참고용으로 남아있음.

**디스크 관리**: 이 인스턴스(32GB)가 계속 빠듯했음(여러 차례 위기 있었음, log.md 참고) —
새 실험 전엔 `df -h`로 여유 확인 습관화. 파인튜닝 모델 1개당 ~2.2GB, 코퍼스 캐시 1개당
~1.6GB(fp16 기준) 소요.

**Vector DB 관련 참고사항**:
- FAISS 대신 **PostgreSQL+pgvector** 채택 — 이유: (1) 메타데이터 필터(법원명/사건종류명/
  선고일자 등)와 벡터 검색을 SQL 한 쿼리에서 결합 가능, (2) Phase 4에서 어차피 Postgres를 쓸
  계획이라 인프라 통합, (3) chunk200 기준 82만 벡터는 FAISS만의 우위가 필요한 규모가 아님
- 이 인스턴스에 Postgres 16 + pgvector 0.6.0 설치, supervisor 서비스(`postgres`)로 등록해
  인스턴스 재시작에도 자동 기동. DB명 `lexchatbot`, 접속정보는 `server/.env`의 `DATABASE_URL`
  (이 인스턴스는 `workspace_is_volume: false`일 경우 recycle/destroy 시 Postgres 데이터
  자체는 날아감 — DB_data JSON + 캐시된 임베딩(.npy)만 있으면 재구축 가능하니 문제 없음)
- 테이블 스키마(`server/db/schema.sql`): `make_chunk_table(suffix)` 함수로 필요한 chunk_size/overlap
  조합만 그때그때 생성(`chunks_200`, `chunks_200_overlap50` 등). 공통 필드는 사건번호/사건명/
  법원명/선고일자/사건종류명(metadata) + chunk_text + embedding(VECTOR(1024))만 포함 — 판시사항/
  판결요지/판결유형/선고 등은 의도적으로 미포함(3절 논의 범위 밖)
- `test_val_pgvector.py`의 `DISTINCT ON` dedupe 쿼리에 버그가 있었음(사건번호 순으로 잘려서
  사실상 무작위 후보군을 가져옴) — CTE로 "HNSW 최근접 500개를 먼저 뽑고 그 안에서만 dedupe"
  하도록 수정 완료. 이 패턴(top-k 먼저 뽑고 그 안에서 문서 단위 dedupe)은 이후 정식 retrieval
  코드에도 그대로 적용해야 함
- sparse(BM25/전문검색) 추가는 스키마에 컬럼+인덱스(tsvector/GIN)만 얹으면 되는 additive
  작업이라 지금 안 해도 나중에 언제든 가능 — dense 단독으로 먼저 진행

---

## 6. 세션 시작/종료 체크리스트

**세션 시작 시**: 이 파일 → [log.md](log.md) 최근 항목 → 필요시 [data_preprocessing_log.md](data_preprocessing_log.md) 순서로 읽기.

**세션 종료(또는 계획 하나 완료) 시**:
- [ ] log.md에 날짜/한 일/결과/결정 이유 기록
- [ ] 이 파일 "1. 지금 당장 상태" 표 갱신
- [ ] 이 파일 "5. 다음 계획" 갱신 (다음 세션이 뭘 해야 할지 근거와 함께)
