"""
RAG 답변 생성 파이프라인 — Phase 3.
검색(dense+reranker, server/eval/retrievers.py의 CrossEncoderReranker 재사용) → top-5 판례의
컨텍스트(전체 원문 또는 문서당 chunk 3개) → LLM(EXAONE-4.0-1.2B)이 사건번호를 인용하며 답변 생성.

두 컨텍스트 방식("문서 전체" vs "매칭 chunk+앞뒤")을 비교하기 위한 두 함수를 다 제공한다
(--context-mode). judge_compare.py가 이 파일의 함수들을 재사용해서 두 방식을 한 번에 비교한다.
LLM 선택 근거(EXAONE-4.0-1.2B, transformers 원본, GGUF 아님)는 docs/log.md 2026-08-25 참고.

사용법:
    python generate.py                          # val_query.json 0번째, 문서 전체 컨텍스트
    python generate.py --index 5 --context-mode chunks
    python generate.py --query "직접 입력한 질문"
"""
import argparse
import json
import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # server/research/
SERVER_DIR = os.path.join(BASE_DIR, "..")  # server/

sys.path.insert(0, os.path.join(SERVER_DIR, "eval"))
from retrievers import CrossEncoderReranker, l2norm  # noqa: E402

DB_DATA_DIR = os.path.join(SERVER_DIR, "..", "data", "DB_data")
VAL_PATH = os.path.join(SERVER_DIR, "..", "data", "Val", "val_query.json")

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:lexchatbot_dev@127.0.0.1:5432/lexchatbot")
TABLE = "chunks_300_overlap100"
DENSE_MODEL_PATH = os.path.join(SERVER_DIR, "finetune", "output", "kure-v1-finetuned-hard")
RERANK_MODEL_PATH = os.path.join(SERVER_DIR, "finetune", "output", "bge-reranker-v2-m3-finetuned")
LLM_PATH = "LGAI-EXAONE/EXAONE-4.0-1.2B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TOP_K = 5
TOP_N_RERANK = 30  # dense가 1단계로 뽑는 후보 수 (log.md 2026-08-25: recall@30=0.9725로 충분)
EF_SEARCH = 200
CHUNK_WINDOW = 1  # 매칭 chunk 기준 앞뒤로 몇 개씩 붙일지 (1이면 문서당 총 3개: 앞+매칭+뒤)


# ---------- 모델 로딩 ----------

def load_models():
    """dense/reranker/LLM을 전부 로드하고 (retriever, tokenizer, llm, conn)을 반환.
    judge_compare.py처럼 한 프로세스에서 여러 번 생성할 때 모델을 한 번만 로드하려고 분리함."""
    import psycopg2

    print("dense 모델 로딩...")
    dense_model = SentenceTransformer(DENSE_MODEL_PATH, device=DEVICE,
                                       model_kwargs={"torch_dtype": torch.float16} if DEVICE == "cuda" else None)

    conn = psycopg2.connect(DATABASE_URL)

    print("reranker 로딩...")
    retriever = CrossEncoderReranker(dense_model, RERANK_MODEL_PATH, conn, TABLE,
                                      top_n=TOP_N_RERANK, ef_search=EF_SEARCH, device=DEVICE)

    print(f"LLM 로딩: {LLM_PATH}")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(LLM_PATH)
    llm = AutoModelForCausalLM.from_pretrained(LLM_PATH, dtype=torch.bfloat16, device_map="auto")

    return dense_model, retriever, tokenizer, llm, conn


# ---------- 검색 ----------

def retrieve_top_k(retriever, query, top_k=TOP_K):
    return retriever.retrieve_batch([query], max_k=top_k)[0]


# ---------- 컨텍스트 구성 ----------

def load_full_doc(case_no):
    path = os.path.join(DB_DATA_DIR, f"{case_no}.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_context_full(case_nos):
    """방식 1: 판례 원문 전체."""
    parts = []
    for cn in case_nos:
        d = load_full_doc(cn)
        parts.append(
            f"[사건번호: {d['사건번호']}] {d['사건명']} ({d['법원명']}, {d['선고일자']})\n{d['판례내용']}"
        )
    return "\n\n---\n\n".join(parts)


def build_context_chunks(dense_model, conn, case_nos, query, window=CHUNK_WINDOW):
    """방식 2: 문서별로 쿼리와 가장 유사한 chunk + 그 앞뒤 window개씩만.
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
    # 2026-09-01: "질문을 먼저 파악하고 그것에 직접 답하라"는 다단계 지시를 추가한 v2를
    # 195건 정식 비교했더니 오히려 전 지표 악화(hit resolves_question 89.0%→65.7%,
    # 둘다(진짜 정답) 78.0%→59.6% 등, docs/log.md 참고) — EXAONE-4.0-1.2B처럼 작은
    # 모델은 지시가 복잡해질수록 오히려 핵심(질문에 직접 답하기)에 쓸 주의력이 분산되는
    # 것으로 추정. 원래의 단순한 프롬프트로 되돌림.
    return (
        "당신은 한국 법률 판례를 근거로 답변하는 법률 어시스턴트입니다. "
        "아래 제공된 판례들만 근거로 삼아 질문에 답하세요. "
        "판례에 없는 내용은 절대 지어내지 마세요. "
        "답변에서 근거로 사용한 부분은 반드시 [사건번호]를 명시해서 인용하세요. "
        "제공된 판례로 답할 수 없으면 그렇게 솔직히 답하세요.\n\n"
        f"=== 참고 판례 ===\n{context}\n\n=== 질문 ===\n{query}"
    )


# ---------- 생성 ----------

def generate_answer(tokenizer, llm, query, context, max_new_tokens=1024, reasoning=False):
    """reasoning=True면 EXAONE-4.0의 hybrid reasoning mode를 켬(`enable_thinking=True`).
    프롬프트에 다단계 지시를 직접 써넣는 방식(2026-09-01, 실패)과 달리, 모델 자체의
    내장 사고 과정을 쓰는 거라 "작은 모델이 복잡한 지시에 혼란스러워하는" 문제를 피할
    가능성이 있음 — docs/log.md 참고. reasoning mode는 모델 카드 권장대로 그리디가 아닌
    샘플링(temperature=0.6, top_p=0.95)을 씀."""
    prompt = build_prompt(query, context)
    messages = [{"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=False,
        enable_thinking=reasoning,
    ).to(llm.device)
    gen_kwargs = {"max_new_tokens": max_new_tokens}
    if reasoning:
        gen_kwargs.update(do_sample=True, temperature=0.6, top_p=0.95)
    else:
        gen_kwargs.update(do_sample=False)
    output = llm.generate(input_ids, **gen_kwargs)
    decoded = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
    if reasoning and "</think>" in decoded:
        # enable_thinking=True면 사고 과정과 최종 답변이 </think>로만 구분된 채 하나의
        # 텍스트로 나옴(skip_special_tokens로도 안 지워짐, 실측 확인) — 최종 답변만 반환
        decoded = decoded.split("</think>", 1)[1].strip()
    return decoded


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, default=0, help="val_query.json에서 몇 번째 쿼리를 쓸지")
    parser.add_argument("--query", type=str, default=None, help="직접 쿼리 입력(지정 시 --index 무시)")
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--context-mode", choices=["full", "chunks"], default="full")
    parser.add_argument("--reasoning", action="store_true", help="EXAONE-4.0 reasoning mode 사용")
    args = parser.parse_args()

    if args.query:
        query = args.query
    else:
        with open(VAL_PATH, encoding="utf-8") as f:
            val_data = json.load(f)
        query = val_data[args.index]["query"]
        print(f"[val_query.json #{args.index}] 정답 case_ids: {val_data[args.index]['case_ids']}")

    print(f"쿼리: {query}")

    dense_model, retriever, tokenizer, llm, conn = load_models()

    print("검색+재정렬 중...")
    ranked = retrieve_top_k(retriever, query, args.top_k)
    print(f"top-{args.top_k} 판례: {ranked}")

    print(f"컨텍스트 구성 중 ({args.context_mode})...")
    if args.context_mode == "full":
        context = build_context_full(ranked)
    else:
        context = build_context_chunks(dense_model, conn, ranked, query)

    print("답변 생성 중...")
    max_new = 2048 if args.reasoning else 1024  # reasoning 과정이 토큰을 더 씀
    answer = generate_answer(tokenizer, llm, query, context, max_new_tokens=max_new, reasoning=args.reasoning)

    print("\n===== 답변 =====")
    print(answer)


if __name__ == "__main__":
    main()
