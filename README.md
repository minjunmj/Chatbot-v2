# LexChatbot 🎙️⚖️

> 음성으로 묻고, 판례로 답하는 법률 AI 챗봇

음성을 녹음하면 자동으로 텍스트로 변환하고, 국내 법률 판례 데이터베이스를 검색해 관련 판례와 함께 답변을 제공하는 모바일 챗봇입니다.

---

## 🛠️ 기술 스택

| 구분 | 기술 |
|------|------|
| 클라이언트 | Flutter (Dart) |
| 서버 | Python FastAPI |
| LLM | OpenAI GPT-4o-mini |
| STT | faster-whisper (base) |
| 임베딩 | HuggingFace (fine-tuned) |
| 벡터 DB | FAISS |
| Reranker | BAAI/bge-reranker-v2-m3 |
| 배포 | Docker + Cloudflare Tunnel |

---

## 📱 주요 기능

- 🎙️ **음성 입력** — 마이크로 질문하면 Whisper STT가 자동으로 텍스트 변환
- ⚖️ **판례 검색** — 국내 법률 판례 벡터 DB에서 관련 판례 자동 검색
- 🧭 **스마트 라우팅** — 법률 질문은 RAG, 일반 질문은 LLM 직접 답변으로 자동 분류
- 💬 **대화 히스토리** — 세션 단위로 이전 대화 문맥 유지
- ⚡ **비동기 처리** — 즉시 응답 후 폴링 방식으로 Cloudflare 타임아웃 우회

---

## 🏗️ 시스템 구조

```
[Flutter 앱]
    │  음성 녹음 (AAC)
    │  POST /voice-chat
    ▼
[FastAPI 서버]
    │
    ├─ 1. ffmpeg → WAV 변환 (Whisper 맞춤)
    ├─ 2. Whisper STT → 텍스트
    ├─ 3. LLM 라우터 → RAG / DIRECT 분류
    │
    ├─ [DIRECT] GPT-4o-mini 직접 답변
    │
    └─ [RAG] 3단계 검색 파이프라인
            ├─ Stage 1: FAISS 유사도 검색 (k=50)
            ├─ Stage 2: MMR 다양성 선택 (top 10)
            └─ Stage 3: CrossEncoder Rerank (top 3)
                        → 판례 원문 로딩 → GPT 최종 답변
    │
    │  202 + job_id 즉시 반환
    ▼
[Flutter 앱]
    └─ Polling GET /voice-chat/result → 결과 표시
```
---

## 📁 프로젝트 구조

```
CaseWhisper/
├── lib/
│   └── main.dart          # Flutter 앱 전체 (UI + 녹음 + 서버 통신)
├── server/
│   ├── main.py            # FastAPI 서버 (STT + RAG + LLM)
│   ├── acc_test.py        # MRR@k 정확도 평가 스크립트
│   ├── requirements.txt   # Python 의존성
│   ├── Dockerfile
│   └── docker-compose.yml # 서버 실행 + Cloudflare Tunnel
├── android/               # Flutter Android 빌드 설정
```
---

## AI Hub 판례 데이터 다운로드

[AI Hub의 생성형AI 법률/규정 텍스트 분석 데이터(고도화) - 상황에 따른 판례 데이터](https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=71723)
(데이터셋 키 `71723`)는 용량과 이용 조건 때문에 Git에 포함하지 않습니다. 새 서버에서는
저장소를 받은 뒤 다음 순서로 다운로드합니다.

1. AI Hub 데이터셋 페이지에서 다운로드를 신청하고 승인을 받습니다.
2. [AI Hub 오픈 API 페이지](https://www.aihub.or.kr/devsport/apishell/list.do)에서 API Key를 발급받습니다.
3. 프로젝트 루트에서 아래 명령을 실행합니다.

```bash
AIHUB_API_KEY='발급받은-키' ./scripts/download_aihub_data.sh
```

데이터는 프로젝트의 `data/`에 저장되며 Git에서 자동으로 제외됩니다. 다운로드가 끝나면
`data/` 아래의 ZIP 파일도 각각의 ZIP이 위치한 폴더에 자동으로 압축 해제합니다. 원본 ZIP은
삭제하지 않습니다. 다운로드가 완료된 상태에서 스크립트를 다시 실행하면 중복 다운로드를
건너뜁니다. 다시 내려받아야 한다면 다음을 실행합니다.

```bash
AIHUB_API_KEY='발급받은-키' ./scripts/download_aihub_data.sh --force
```

이미 다운로드된 ZIP 파일만 압축 해제하려면 API Key 없이 실행할 수 있습니다.

```bash
./scripts/download_aihub_data.sh --extract-only
```

> API Key를 `.env`, 소스 코드 또는 Git 저장소에 저장하지 마세요. AI Hub 공식 다운로더는
> 압축 해제 과정까지 수행하므로 데이터 용량의 2~3배에 해당하는 여유 디스크 공간이 필요합니다.

---

## 🔍 RAG 파이프라인 상세

### 3단계 검색 과정

```
질문 입력
  │
  ▼
Stage 1 │ FAISS 유사도 검색
        │ - 전체 판례 DB에서 코사인 유사도 기준 상위 50개 추출
        │ - 사건번호 기준 중복 제거
  │
  ▼
Stage 2 │ MMR (Maximal Marginal Relevance)
        │ - 관련성 높으면서 서로 중복되지 않는 10개 선택
        │ - λ=0.9 (관련성 중심)
  │
  ▼
Stage 3 │ CrossEncoder Rerank
        │ - 질문-문서 쌍을 직접 비교해 정밀 점수 산출
        │ - 최종 상위 3개 선택
        │ - 임계치(0.20) 미달 시 "관련 판례 없음" 처리
  │
  ▼
판례 원문 JSON 로딩 (판시사항 / 판결요지 / 판례내용)
  │
  ▼
GPT-4o-mini 최종 답변 생성
```

### 라우팅 기준

| 경로 | 조건 |
|------|------|
| **RAG** | 특정 법령/판례/사례/처벌 관련 질문 (모호하면 RAG) |
| **DIRECT** | 법률과 무관한 일반 질문 |
