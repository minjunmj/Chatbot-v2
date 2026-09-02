"""
LexChatbot 백엔드 API + 채팅 데모 페이지.
dense+reranker top-5 검색 → chunks 컨텍스트(정식 채택된 방식, docs/log.md 2026-09-01) →
EXAONE-4.0-1.2B 생성. 모델은 서버 기동 시 한 번만 로드해서 메모리에 올려두고, 요청마다 재사용.

/chat은 멀티턴 채팅 — 매 턴마다 EXAONE으로 "새 판례 검색이 필요한 질문인지(search) vs
일반 대화/직전 답변에 대한 재질문인지(direct)"를 먼저 라우팅한 뒤, search면 검색+생성,
direct면 검색 없이 대화 히스토리만으로 답변(재질문 처리 — 서버에 세션을 따로 저장하지
않고, 클라이언트가 매번 전체 대화 기록을 같이 보내는 stateless 방식). 자세한 설계는
pipeline.py의 answer_chat() 참고. /ask는 단발 질문용으로 남겨둠(하위 호환).

Streamlit 대신 FastAPI가 정적 HTML 페이지까지 직접 서빙 — 이 인스턴스 디스크 여유가
1GB 안팎으로 빠듯해서 Streamlit의 무거운 의존성(pandas/pyarrow 등)을 피하려는 목적도 있음.

실행(수동 테스트용, 실제 배포는 supervisor가 담당):
    uvicorn api:app --host 127.0.0.1 --port <내부 포트>
"""
import os
from contextlib import asynccontextmanager

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from pipeline import load_models, answer_query, answer_chat, TOP_K

models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("모델 로딩 중... (dense+reranker+EXAONE, 서버 기동 시 1회만)")
    dense_model, retriever, tokenizer, llm, conn = load_models()
    models["dense_model"] = dense_model
    models["retriever"] = retriever
    models["tokenizer"] = tokenizer
    models["llm"] = llm
    models["conn"] = conn
    print("모델 로딩 완료, 요청 받을 준비됨")
    yield
    models.clear()


app = FastAPI(title="LexChatbot API", lifespan=lifespan)


class AskRequest(BaseModel):
    query: str
    top_k: int = TOP_K


class AskResponse(BaseModel):
    answer: str
    cited_cases: list[str]


class ChatMessage(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    top_k: int = TOP_K


class ChatResponse(BaseModel):
    answer: str
    cited_cases: list[str]
    route: str  # "search" | "direct"


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": bool(models)}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    answer, ranked = answer_query(models, req.query, req.top_k)
    return AskResponse(answer=answer, cited_cases=ranked)


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    history = [{"role": m.role, "content": m.content} for m in req.messages]
    answer, ranked, route = answer_chat(models, history, req.top_k)
    return ChatResponse(answer=answer, cited_cases=ranked, route=route)


DEMO_PAGE = """
<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8">
<title>LexChatbot 데모</title>
<style>
  * { box-sizing: border-box; }
  body {
    font-family: -apple-system, sans-serif; margin: 0; padding: 0; color: #222;
    height: 100vh; display: flex; flex-direction: column;
  }
  header {
    padding: 12px 16px; border-bottom: 1px solid #eee; flex-shrink: 0;
    display: flex; align-items: center; justify-content: space-between; gap: 12px;
  }
  header h1 { font-size: 1.1rem; margin: 0; }
  header p { font-size: 0.8rem; color: #777; margin: 4px 0 0; }
  #resetBtn {
    flex-shrink: 0; padding: 6px 12px; font-size: 0.8rem; border: 1px solid #ddd;
    border-radius: 14px; background: white; color: #555; cursor: pointer;
  }
  #resetBtn:hover { background: #f5f5f5; }

  #chat {
    flex: 1; overflow-y: auto; padding: 16px; display: flex; flex-direction: column; gap: 10px;
  }
  .row { display: flex; }
  .row.user { justify-content: flex-end; }
  .row.assistant { justify-content: flex-start; }
  .bubble {
    max-width: 75%; padding: 10px 14px; border-radius: 16px; line-height: 1.5;
    white-space: pre-wrap; font-size: 0.95rem;
  }
  .row.user .bubble { background: #3b82f6; color: white; border-bottom-right-radius: 4px; }
  .row.assistant .bubble { background: #f0f0f0; color: #222; border-bottom-left-radius: 4px; }
  .cases { margin-top: 6px; font-size: 0.75rem; color: #999; }
  .row.assistant .cases { padding-left: 14px; }
  .typing { font-size: 0.85rem; color: #999; padding-left: 14px; }

  #inputbar {
    flex-shrink: 0; display: flex; gap: 8px; padding: 12px 16px;
    border-top: 1px solid #eee; background: white;
  }
  #inputbar input {
    flex: 1; padding: 10px 14px; font-size: 1rem; border: 1px solid #ddd; border-radius: 20px;
  }
  #inputbar button {
    padding: 10px 20px; font-size: 1rem; border: none; border-radius: 20px;
    background: #3b82f6; color: white; cursor: pointer;
  }
  #inputbar button:disabled { background: #aaa; cursor: default; }
</style>
</head>
<body>
  <header>
    <div>
      <h1>LexChatbot — 한국 법률 판례 검색 챗봇</h1>
      <p>법률 질문은 관련 판례를 검색해서 답합니다. 이전 답변에 대한 재질문도 가능합니다.</p>
    </div>
    <button id="resetBtn" onclick="resetChat()">새 대화</button>
  </header>

  <div id="chat"></div>

  <div id="inputbar">
    <input id="input" type="text" placeholder="질문을 입력하세요" autocomplete="off">
    <button id="sendBtn" onclick="send()">전송</button>
  </div>

<script>
let history = [];  // [{role: "user"|"assistant", content: str}]
const chatEl = document.getElementById('chat');
const inputEl = document.getElementById('input');
const sendBtn = document.getElementById('sendBtn');

function addBubble(role, text, casesText) {
  const row = document.createElement('div');
  row.className = 'row ' + role;
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.textContent = text;
  row.appendChild(bubble);
  if (casesText) {
    const casesDiv = document.createElement('div');
    casesDiv.className = 'cases';
    casesDiv.textContent = casesText;
    row.appendChild(casesDiv);
  }
  chatEl.appendChild(row);
  chatEl.scrollTop = chatEl.scrollHeight;
  return row;
}

function tokenParam() {
  const token = new URLSearchParams(window.location.search).get('token');
  return token ? '?token=' + encodeURIComponent(token) : '';
}

async function send() {
  const text = inputEl.value.trim();
  if (!text) return;
  inputEl.value = '';
  sendBtn.disabled = true;

  history.push({role: 'user', content: text});
  addBubble('user', text);

  const typingRow = document.createElement('div');
  typingRow.className = 'row assistant';
  typingRow.innerHTML = '<div class="typing">답변 생성 중... (5~20초 정도 걸립니다)</div>';
  chatEl.appendChild(typingRow);
  chatEl.scrollTop = chatEl.scrollHeight;

  try {
    const resp = await fetch('/chat' + tokenParam(), {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({messages: history})
    });
    const data = await resp.json();
    typingRow.remove();

    history.push({role: 'assistant', content: data.answer});
    const casesText = (data.route === 'search' && data.cited_cases.length)
      ? '검색된 판례: ' + data.cited_cases.join(', ') : null;
    addBubble('assistant', data.answer, casesText);
  } catch (e) {
    typingRow.remove();
    addBubble('assistant', '오류가 발생했습니다: ' + e);
  } finally {
    sendBtn.disabled = false;
    inputEl.focus();
  }
}

inputEl.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.isComposing) send();
});

function resetChat() {
  history = [];
  chatEl.innerHTML = '';
  inputEl.value = '';
  inputEl.focus();
}
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index():
    return DEMO_PAGE
