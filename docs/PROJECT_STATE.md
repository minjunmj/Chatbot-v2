# LexChatbot RAG 프로젝트 — 현재 상태 (여기부터 읽기)

> **새 세션(새 서버) 시작 시 이 파일 하나만 먼저 읽으면 됨.** 사용자가 프로젝트를 처음부터
> 다시 설명할 필요 없이, 이 문서 + [log.md](log.md)만 보고 바로 이어서 작업할 것.
>
> 갱신 규칙: 계획 하나가 끝나면 → ① [log.md](log.md)에 날짜/내용/근거 기록 ② 이 파일의
> "1. 지금 당장 상태"와 "5. 다음 계획"을 최신으로 고쳐쓸 것. 이 두 가지를 안 하면 다음
> 세션이 다시 헤맨다.

최종 갱신: 2026-08-17

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
| Chunking / Metadata 설계 | ❌ 미결정 (보류 중, [data_preprocessing_log.md](data_preprocessing_log.md) 3절) |
| 임베딩 모델 선정 실험 | 🔄 **진행 중** — `server/model_test.py`, 아직 실행 결과 없음 (`cache_embeddings/` 비어있음) |
| Vector DB 구축 | ❌ 시작 전 |
| Retriever 평가 자동화 | ❌ 시작 전 (model_test.py는 임시 스크립트, 정식 평가 하니스 아님) |
| RAG 답변 생성 파이프라인 | ❌ 시작 전 (v1 코드 있으나 재사용 여부 미결정, 아래 2절 참고) |
| 백엔드/배포/운영 | ❌ 시작 전 (v1 코드 있음, 아래 2절 참고) |

**한 줄 요약**: 데이터 준비는 끝났고, 지금은 "어떤 임베딩 모델을 쓸지"를 3개 모델
(Qwen3-Embedding-0.6B, BGE-M3, KURE-v1) 비교로 결정하는 단계. 이 결과가 나와야 chunking →
vector DB 구축으로 넘어갈 수 있음.

---

## 2. v1-baseline (이전 시도 — 폐기하고 처음부터 다시 하는 중, 참고용으로만 보존)

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

> v1을 왜 다시 시작하는지, 무엇이 부족해서 새로 하는지는 아직 이 문서에 기록되어 있지 않음.
> **다음 세션에서 사용자에게 확인해서 이 섹션에 이유를 채워넣을 것** (TODO).

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
  - [ ] Chunking
  - [ ] Metadata
  - [ ] Embedding / Vector DB
  - ➕ **Vector DB 재선정**: v1은 FAISS였음. Phase 4에서 PostgreSQL을 쓸 계획이라면
    pgvector로 통합할지, FAISS를 유지할지 지금 결정해두면 나중에 마이그레이션 안 해도 됨.
- Retriever 평가 환경 구축
  - [ ] Query–정답 판례 평가셋 (Val 쿼리셋은 이미 있음, 재사용 가능)
  - [ ] Recall@k, MRR
  - [ ] 평가 자동화 — ➕ `model_test.py`/`acc_test.py`처럼 매번 스크립트 새로 쓰지 말고,
    corpus/query 바꿔 끼우면 되는 **재사용 가능한 평가 하니스**로 한 번만 제대로 만들어두기
    (Phase 2에서 실험을 계속 돌려야 하므로 여기 투자가 이후 시간을 아껴줌)
- [ ] Retriever Baseline 구축 (최초 검색 정확도 측정) — model_test.py 결과가 이 역할

### Phase 2 — Retriever 성능 개선 (검색 정확도만 집중)
- [ ] Retriever 성능 개선
- [ ] 여러 실험 결과 비교
- [ ] 최종 Retriever 선정
- ➕ **Hybrid Search (BM25 + Dense)**: v1의 acc_test.py에 이미 시도 흔적 있음. 법률 판례는
  법조문 번호·고유명사처럼 키워드 매칭이 중요한 도메인이라 dense 단독보다 잘 나올 가능성 높음.
- ➕ **Reranker 도입 (CrossEncoder)**: v1에서 이미 검증됨(`bge-reranker-v2-m3`). 순수 dense
  대비 성능 향상 폭을 정량적으로 비교해두면 "왜 rerank를 썼는지" 발표 근거가 됨.
- ➕ **임베딩 모델 도메인 파인튜닝 (선택, 난이도 높음)**: `train_query.json`(질문–정답 판례
  쌍)이 이미 있어서 contrastive fine-tuning(hard negative 포함)이 가능한 조건. 포트폴리오
  차별화 포인트로 강력하지만 시간이 걸리므로 baseline 비교 후 시간 남으면 진행.
- ➕ **Chunking 전략 실험**: Phase 1에서 미룬 chunking을 여기서 결정 — 고정 길이 vs
  문단/판시사항 단위 vs 재귀적 분할, chunk 크기별 Recall 비교.

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

**지금 할 일**: `server/model_test.py` 실행 완료해서 `cache_embeddings/results.json` 확보
(Qwen3-Embedding-0.6B / BGE-M3 / KURE-v1의 Recall@1/5/10/20, MRR@10 비교).

**근거**: 이후 모든 실험(chunking, hybrid search, reranker)이 "어떤 임베딩 모델 위에서"
진행되는지에 의존함. 지금 이 비교가 안 끝나면 Phase 1의 나머지(Chunking/Metadata/Vector DB)와
Phase 2 실험을 시작할 기준선 자체가 없음. GPU 서버 대여 중이므로 이번 세션에 끝내는 게 효율적.

**결과 나온 뒤 순서**:
1. 승자 모델 선정 → [log.md](log.md)에 결과 표+선정 근거 기록
2. Chunking 전략 결정 (Phase 1 남은 체크박스) — 승자 모델의 max_seq_length 제약을 고려해서 설계
3. 정식 Vector DB 구축 (FAISS 유지 vs pgvector 결정 포함)
4. 재사용 가능한 평가 하니스로 정리 (Phase 2 실험에 계속 쓸 것이므로)

---

## 6. 세션 시작/종료 체크리스트

**세션 시작 시**: 이 파일 → [log.md](log.md) 최근 항목 → 필요시 [data_preprocessing_log.md](data_preprocessing_log.md) 순서로 읽기.

**세션 종료(또는 계획 하나 완료) 시**:
- [ ] log.md에 날짜/한 일/결과/결정 이유 기록
- [ ] 이 파일 "1. 지금 당장 상태" 표 갱신
- [ ] 이 파일 "5. 다음 계획" 갱신 (다음 세션이 뭘 해야 할지 근거와 함께)
