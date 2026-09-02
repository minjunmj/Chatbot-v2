# LexChatbot

한국 법률 판례를 근거로 답변하는 RAG(Retrieval-Augmented Generation) 챗봇입니다.
사용자의 법률 질문에 대해 관련 판례를 검색하고, 그 판례를 근거로 삼아 사건번호를
인용하며 답변합니다.


## 무엇을 하는가

- 사용자가 법률 질문을 하면, 검색 파이프라인이 44,700건의 판례 코퍼스에서 관련 판례를
  찾아 그 내용을 근거로 EXAONE 모델이 답변을 생성합니다.
- 일반적인 인사/잡담이나 답변에 대한 재질문은 검색 없이 대화 맥락만으로 처리합니다
  (router 기능).
- 질문과 관련된 판례를 찾지 못했다고 판단되면, 근거 없이 지어내는 대신 솔직히
  답변 불가를 알립니다.

## 아키텍처

```
사용자 질문
   │
   ▼
[router] 법률 검색이 필요한 질문인가? ──아니오──▶ 대화 맥락만으로 답변
   │ 예
   ▼
[dense retrieval]  KURE-v1(도메인 파인튜닝) + pgvector HNSW  → top-30 후보
   │
   ▼
[reranker]         bge-reranker-v2-m3(도메인 파인튜닝) cross-encoder → top-5 재정렬
   │
   ├─ 확신도가 threshold 미만 ──▶ "관련 판례를 찾지 못했습니다"
   │
   ▼
[컨텍스트 구성]     문서별 매칭 chunk + 앞뒤 chunk만 사용 (문서 전체 아님)
   │
   ▼
[생성]             EXAONE-4.0-1.2B 가 판례를 근거로 답변 생성, [사건번호] 인용
```

## 기술 스택

| 영역 | 선택 |
|---|---|
| 임베딩 | KURE-v1 (도메인 파인튜닝, in-batch + hard negative 2단계) |
| Vector DB | PostgreSQL + pgvector (HNSW) |
| Reranker | bge-reranker-v2-m3 (도메인 파인튜닝, cross-encoder) |
| 생성 LLM | LG AI Research `EXAONE-4.0-1.2B` |
| 평가 | GPT-5-mini를 LLM judge로 사용 (정확성/groundedness/인용정확성 등) |
| 백엔드 | FastAPI |
| 배포 | Vast.ai GPU 인스턴스, supervisor + Caddy 인증 엣지 |

각 선택의 후보와 근거는 [docs/DECISIONS.md](docs/DECISIONS.md)에 큰 흐름으로,
세부 수치는 [docs/log.md](docs/log.md)에 날짜순으로 정리돼 있습니다.

## 프로젝트 문서

- [docs/DECISIONS.md](docs/DECISIONS.md) — 전체 프로젝트 흐름 요약 (후보 → 결정 → 이유, 세부 수치 제외)
- [docs/log.md](docs/log.md) — 모든 실험/결정의 날짜순 상세 기록 (수치, 원인 분석 포함)
- [docs/PROJECT_STATE.md](docs/PROJECT_STATE.md) — 현재 상태 스냅샷 + 다음 계획
- [docs/STRUCTURE.md](docs/STRUCTURE.md) — 폴더/파일별 용도 참조

## 폴더 구조 (요약)

```
data/     원본 판례 코퍼스 + train/val 쿼리셋 (git 미포함)
scripts/  원본 데이터 전처리 파이프라인
server/
  service/   실제 배포되는 서비스 코드 (FastAPI + RAG 파이프라인)
  eval/      검색(retriever) 평가 인프라 (공유 라이브러리)
  finetune/  임베딩/reranker 도메인 파인튜닝
  db/        Postgres/pgvector 구축
  research/  생성 품질 평가용 일회성 실험 스크립트
```

자세한 파일별 설명은 [docs/STRUCTURE.md](docs/STRUCTURE.md) 참고.

## 실행

```bash
# 1. Postgres/pgvector에 chunks_300_overlap100 테이블 구축 (server/db/build_vector_db.py)
# 2. server/service/ 에서 의존성 설치 후 서버 기동
cd server/service
pip install -r ../requirements.txt
uvicorn api:app --host 0.0.0.0 --port <포트>
```

`DATABASE_URL` 등 환경변수는 `server/.env`에 설정합니다. 서버가 기동되면 `/`에서
간단한 채팅 데모 페이지를, `/health`에서 상태를 확인할 수 있습니다.

## 라이선스 관련 참고

생성 모델 `EXAONE-4.0-1.2B`는 LG AI Research의 연구/학술 목적 라이선스를 따릅니다.
상업적 사용 시 별도 확인이 필요합니다.
