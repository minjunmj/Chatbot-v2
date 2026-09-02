"""
LexChatbot RAG 파이프라인 — 최종 채택된 구성만 포함.

dense(KURE-v1 파인튜닝, ef_search=200) top-30 → reranker(bge-reranker-v2-m3 파인튜닝)
top-5 → chunks 컨텍스트(매칭 chunk+앞뒤 1개씩) → EXAONE-4.0-1.2B(일반 모드) 생성.

`server/research/generate.py`는 이 파이프라인을 만들면서 비교했던 다른 선택지들
(문서 전체 컨텍스트, reasoning mode, 개선 시도했다가 되돌린 프롬프트 v2 등)까지
전부 남겨둔 실험용 버전이고, 이 파일은 그중 실제로 채택된 것만 남긴 배포용 버전임 —
각 선택의 근거는 docs/log.md 참고:
  - chunks vs 문서 전체: 2026-09-01, chunks가 전 지표 우세 + OOM 회피
  - 일반 모드 vs reasoning mode: 2026-09-02, reasoning은 7~7.5배 느려서 배포 전환 위해 보류
  - 프롬프트: 2026-09-01, "질문 먼저 파악" 다단계 지시 추가했다가 오히려 악화돼서 원복
"""
import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # server/service/
SERVER_DIR = os.path.join(BASE_DIR, "..")  # server/

sys.path.insert(0, os.path.join(SERVER_DIR, "eval"))
from retrievers import CrossEncoderReranker, l2norm  # noqa: E402

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:lexchatbot_dev@127.0.0.1:5432/lexchatbot")
TABLE = "chunks_300_overlap100"
DENSE_MODEL_PATH = os.path.join(SERVER_DIR, "finetune", "output", "kure-v1-finetuned-hard")
RERANK_MODEL_PATH = os.path.join(SERVER_DIR, "finetune", "output", "bge-reranker-v2-m3-finetuned")
LLM_PATH = "LGAI-EXAONE/EXAONE-4.0-1.2B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TOP_K = 5
TOP_N_RERANK = 30  # dense가 1단계로 뽑는 후보 수 (log.md 2026-08-25: recall@30=0.9725로 충분)
EF_SEARCH = 200
CHUNK_WINDOW = 1  # 매칭 chunk 기준 앞뒤로 몇 개씩 붙일지 (문서당 총 3개: 앞+매칭+뒤)

# reranker top1 cross-encoder 점수가 이 미만이면 "명백히 무관한 질문"으로 보고 검색 결과를
# 컨텍스트로 쓰지 않음 (log.md 2026-08-31 calibrate_threshold.py 실측: hit 손실 0.4%,
# 무관 질문 포착 다수 — 애매한 미스 자체는 못 거르지만 완전 무관 입력에는 효과적).
# check_miss_rejection.py 실측(2%만 자진 거절)에 따라 LLM 판단에 맡기지 않고 여기서 확정 처리.
RERANK_NO_MATCH_THRESHOLD = 0.2
NO_MATCH_ANSWER = "죄송하지만 질문과 관련된 판례를 찾지 못했습니다. 좀 더 구체적인 법률 질문으로 다시 물어봐 주시겠어요?"


def load_models():
    """dense/reranker/LLM을 전부 로드하고 (dense_model, retriever, tokenizer, llm, conn)을
    반환. api.py가 서버 기동 시 한 번만 호출해서 재사용."""
    import psycopg2

    dense_model = SentenceTransformer(DENSE_MODEL_PATH, device=DEVICE,
                                       model_kwargs={"torch_dtype": torch.float16} if DEVICE == "cuda" else None)
    conn = psycopg2.connect(DATABASE_URL)
    retriever = CrossEncoderReranker(dense_model, RERANK_MODEL_PATH, conn, TABLE,
                                      top_n=TOP_N_RERANK, ef_search=EF_SEARCH, device=DEVICE)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(LLM_PATH)
    llm = AutoModelForCausalLM.from_pretrained(LLM_PATH, dtype=torch.bfloat16, device_map="auto")

    return dense_model, retriever, tokenizer, llm, conn


def retrieve_top_k(retriever, query, top_k=TOP_K):
    return retriever.retrieve_batch([query], max_k=top_k)[0]


def retrieve_top_k_with_score(retriever, query, top_k=TOP_K):
    """retrieve_top_k와 동일하지만 1등 cross-encoder 점수도 같이 반환 — 그 점수를
    RERANK_NO_MATCH_THRESHOLD와 비교해서 '명백히 무관한 질문'을 걸러내는 데 사용."""
    ranked_list, scores = retriever.retrieve_batch_with_scores([query], max_k=top_k)
    return ranked_list[0], scores[0]


def build_context(dense_model, conn, case_nos, query, window=CHUNK_WINDOW):
    """문서별로 쿼리와 가장 유사한 chunk + 그 앞뒤 window개씩만 붙여서 컨텍스트 구성.
    CrossEncoderReranker는 case_no 랭킹만 반환하고 어떤 chunk가 매칭됐는지는 버리므로,
    여기서 문서별로 다시 한 번 최근접 chunk_index를 찾는다(코퍼스 전체가 아니라 case_no
    하나로 좁혀서 찾는 거라 비용이 크지 않음)."""
    query_emb = l2norm(dense_model.encode(
        [query], convert_to_numpy=True, normalize_embeddings=True
    ).astype(np.float32))[0]
    lit = "[" + ",".join(f"{x:.6f}" for x in query_emb) + "]"

    parts = []
    with conn.cursor() as cur:
        for cn in case_nos:
            cur.execute(
                f"SELECT chunk_index, embedding <=> %s AS dist FROM {TABLE} "
                f"WHERE case_no = %s ORDER BY dist LIMIT 1",
                (lit, cn),
            )
            best_idx, _ = cur.fetchone()
            cur.execute(
                f"SELECT chunk_index, chunk_text, case_name, court_name, judgment_date FROM {TABLE} "
                f"WHERE case_no = %s AND chunk_index BETWEEN %s AND %s ORDER BY chunk_index",
                (cn, best_idx - window, best_idx + window),
            )
            rows = cur.fetchall()
            meta = rows[0]
            chunk_text = " ".join(r[1] for r in rows)  # 앞/매칭/뒤 chunk를 순서대로 이어붙임
            parts.append(f"[사건번호: {cn}] {meta[2]} ({meta[3]}, {meta[4]})\n{chunk_text}")
    return "\n\n---\n\n".join(parts)


def build_prompt(query, context):
    return (
        "당신은 한국 법률 판례를 근거로 답변하는 법률 어시스턴트입니다. "
        "아래 제공된 판례들만 근거로 삼아 질문에 답하세요. "
        "판례에 없는 내용은 절대 지어내지 마세요. "
        "답변에서 근거로 사용한 부분은 반드시 [사건번호]를 명시해서 인용하세요. "
        "제공된 판례로 답할 수 없으면 그렇게 솔직히 답하세요.\n\n"
        f"=== 참고 판례 ===\n{context}\n\n=== 질문 ===\n{query}"
    )


def generate_answer(tokenizer, llm, query, context, max_new_tokens=1024):
    prompt = build_prompt(query, context)
    messages = [{"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=False
    ).to(llm.device)
    output = llm.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)


def answer_query(models, query, top_k=TOP_K):
    """단발 질문용 진입점(하위 호환, /ask가 사용) — 검색부터 답변 생성까지 한 번에."""
    ranked, top1_score = retrieve_top_k_with_score(models["retriever"], query, top_k)
    if not ranked or top1_score < RERANK_NO_MATCH_THRESHOLD:
        return NO_MATCH_ANSWER, []
    context = build_context(models["dense_model"], models["conn"], ranked, query)
    answer = generate_answer(models["tokenizer"], models["llm"], query, context)
    return answer, ranked


# ---------- 채팅(멀티턴): 라우팅 + 대화 히스토리 ----------

ROUTER_SYSTEM = (
    "다음은 사용자가 챗봇에게 보낸 메시지입니다. 이 메시지에 답하기 위해 한국 법률 판례를 "
    "새로 검색해야 하는지 판단하세요.\n\n"
    "예시:\n"
    "- \"안녕하세요\" → DIRECT(일반적인 질문)\n"
    "- \"너는 뭘 할 수 있어?\" → DIRECT\n"
    "- \"방금 답변 좀 더 쉽게 설명해줘\" → DIRECT (직전 대화에 대한 재질문)\n"
    "- \"진정성립이 인정되는 처분문서는 증명력이 있는가?\" → SEARCH\n"
    "- \"계약 해지 시 위약금은 어떻게 되나요?\" → SEARCH\n"
    "- 특정 법률 용어·조항·쟁점, 계약/소송/판결 등 법적 상황을 구체적으로 묻는 경우 → SEARCH\n\n"
    "**판단이 애매하면 반드시 SEARCH를 선택하세요** — 법률 질문을 놓치고 근거 없이 답하는 "
    "것이 불필요한 검색보다 훨씬 큰 문제입니다.\n"
    "SEARCH 또는 DIRECT, 그 단어 하나만 답하세요."
)

GENERAL_SYSTEM = (
    "당신은 한국 법률 판례 검색 챗봇입니다. 지금 사용자의 메시지는 법률 판례 검색이 "
    "필요 없는 일반적인 대화(인사, 잡담, 이전 답변에 대한 설명 요청 등)로 판단됐습니다. "
    "대화 맥락에 맞게 자연스럽고 간결하게 답하세요. 새로운 법률 쟁점을 묻는 것 같으면 "
    "구체적인 법률 질문을 해달라고 안내하세요."
)

RAG_SYSTEM_TEMPLATE = (
    "당신은 한국 법률 판례를 근거로 답변하는 법률 어시스턴트입니다. "
    "아래 제공된 판례들만 근거로 삼아 질문에 답하세요. "
    "판례에 없는 내용은 절대 지어내지 마세요. "
    "답변에서 근거로 사용한 부분은 반드시 [사건번호]를 명시해서 인용하세요. "
    "제공된 판례로 답할 수 없으면 그렇게 솔직히 답하세요.\n\n"
    "=== 참고 판례 ===\n{context}"
)


def _chat_generate(tokenizer, llm, messages, max_new_tokens=1024):
    input_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=False
    ).to(llm.device)
    output = llm.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)


def classify_query(tokenizer, llm, latest_query):
    """새 판례 검색이 필요한 질문인지("search") 아니면 일반 대화/재질문인지("direct")
    라우팅. 별도 모델 없이 이미 로드된 EXAONE을 짧은 분류 프롬프트로 재사용 —
    max_new_tokens을 작게 잡아서 라우팅 자체의 지연은 최소화."""
    messages = [
        {"role": "system", "content": ROUTER_SYSTEM},
        {"role": "user", "content": latest_query},
    ]
    result = _chat_generate(tokenizer, llm, messages, max_new_tokens=10)
    return "direct" if "DIRECT" in result.upper() else "search"


def answer_chat(models, history, top_k=TOP_K):
    """멀티턴 채팅 진입점. history: [{"role": "user"/"assistant", "content": str}, ...]
    (마지막 원소가 이번에 새로 온 사용자 메시지). 매 턴마다 라우팅부터 다시 하므로, 같은
    대화 안에서도 어떤 턴은 검색이 필요하고 어떤 턴은 필요 없을 수 있음(예: 법률 질문 →
    그 답변에 대한 재질문 → 새로운 법률 질문). 재질문 처리는 별도 세션 저장 없이 history를
    그대로 모델에 다시 넣어주는 방식으로 처리 — 검색이 필요 없는 턴이면 직전 답변 내용이
    이미 대화 맥락에 있으니 그걸로 부연설명 가능.

    반환: (answer, cited_cases, route) — route는 "search"|"direct", UI에서 참고용으로 노출 가능.
    """
    latest_query = history[-1]["content"]
    route = classify_query(models["tokenizer"], models["llm"], latest_query)

    if route == "search":
        ranked, top1_score = retrieve_top_k_with_score(models["retriever"], latest_query, top_k)
        if not ranked or top1_score < RERANK_NO_MATCH_THRESHOLD:
            return NO_MATCH_ANSWER, [], route
        context = build_context(models["dense_model"], models["conn"], ranked, latest_query)
        system_msg = {"role": "system", "content": RAG_SYSTEM_TEMPLATE.format(context=context)}
    else:
        ranked = []
        system_msg = {"role": "system", "content": GENERAL_SYSTEM}

    messages = [system_msg] + history
    answer = _chat_generate(models["tokenizer"], models["llm"], messages)
    return answer, ranked, route
