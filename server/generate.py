"""
RAG 답변 생성 최소 동작 버전 — Phase 3 첫 스크립트.
검색(dense+reranker, server/eval/retrievers.py의 CrossEncoderReranker 재사용) → top-5
판례의 원문 전체를 컨텍스트로 → LLM(EXAONE-4.0-1.2B)이 사건번호를 인용하며 답변 생성.

"top-5 문서 전체 vs top-5 문서당 chunk 3개" 비교 실험의 베이스라인(문서 전체 버전)이자,
end-to-end 파이프라인이 실제로 돌아가는지 확인하는 용도. LLM 선택 근거(EXAONE-4.0-1.2B,
transformers 원본, GGUF 아님)는 docs/log.md 2026-08-25 참고.

사용법:
    python generate.py                   # val_query.json의 0번째 쿼리로 테스트
    python generate.py --index 5         # val_query.json의 5번째 쿼리로 테스트
    python generate.py --query "직접 입력한 질문"
"""
import argparse
import json
import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from sentence_transformers import SentenceTransformer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval"))
from retrievers import CrossEncoderReranker  # noqa: E402

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # server/
DB_DATA_DIR = os.path.join(BASE_DIR, "..", "data", "DB_data")
VAL_PATH = os.path.join(BASE_DIR, "..", "data", "Val", "val_query.json")

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:lexchatbot_dev@127.0.0.1:5432/lexchatbot")
TABLE = "chunks_300_overlap100"
DENSE_MODEL_PATH = os.path.join(BASE_DIR, "finetune", "output", "kure-v1-finetuned-hard")
RERANK_MODEL_PATH = os.path.join(BASE_DIR, "finetune", "output", "bge-reranker-v2-m3-finetuned")
LLM_PATH = "LGAI-EXAONE/EXAONE-4.0-1.2B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TOP_K = 5
TOP_N_RERANK = 30  # dense가 1단계로 뽑는 후보 수 (log.md 2026-08-25: recall@30=0.9725로 충분)
EF_SEARCH = 200


def load_full_doc(case_no):
    path = os.path.join(DB_DATA_DIR, f"{case_no}.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_prompt(query, docs):
    context_parts = [
        f"[사건번호: {d['사건번호']}] {d['사건명']} ({d['법원명']}, {d['선고일자']})\n{d['판례내용']}"
        for d in docs
    ]
    context = "\n\n---\n\n".join(context_parts)
    return (
        "당신은 한국 법률 판례를 근거로 답변하는 법률 어시스턴트입니다. "
        "아래 제공된 판례들만 근거로 삼아 질문에 답하세요. "
        "판례에 없는 내용은 절대 지어내지 마세요. "
        "답변에서 근거로 사용한 부분은 반드시 [사건번호]를 명시해서 인용하세요. "
        "제공된 판례로 답할 수 없으면 그렇게 솔직히 답하세요.\n\n"
        f"=== 참고 판례 ===\n{context}\n\n=== 질문 ===\n{query}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, default=0, help="val_query.json에서 몇 번째 쿼리를 쓸지")
    parser.add_argument("--query", type=str, default=None, help="직접 쿼리 입력(지정 시 --index 무시)")
    parser.add_argument("--top-k", type=int, default=TOP_K)
    args = parser.parse_args()

    if args.query:
        query = args.query
    else:
        with open(VAL_PATH, encoding="utf-8") as f:
            val_data = json.load(f)
        query = val_data[args.index]["query"]
        print(f"[val_query.json #{args.index}] 정답 case_ids: {val_data[args.index]['case_ids']}")

    print(f"쿼리: {query}")

    print("dense 모델 로딩...")
    dense_model = SentenceTransformer(DENSE_MODEL_PATH, device=DEVICE,
                                       model_kwargs={"torch_dtype": torch.float16} if DEVICE == "cuda" else None)

    import psycopg2
    conn = psycopg2.connect(DATABASE_URL)

    print("reranker 로딩...")
    retriever = CrossEncoderReranker(dense_model, RERANK_MODEL_PATH, conn, TABLE,
                                      top_n=TOP_N_RERANK, ef_search=EF_SEARCH, device=DEVICE)

    print("검색+재정렬 중...")
    ranked = retriever.retrieve_batch([query], max_k=args.top_k)[0]
    print(f"top-{args.top_k} 판례: {ranked}")

    docs = [load_full_doc(cn) for cn in ranked]

    print(f"LLM 로딩: {LLM_PATH}")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(LLM_PATH)
    llm = AutoModelForCausalLM.from_pretrained(LLM_PATH, dtype=torch.bfloat16, device_map="auto")

    prompt = build_prompt(query, docs)
    messages = [{"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=False
    ).to(llm.device)

    print("답변 생성 중...")
    output = llm.generate(input_ids, max_new_tokens=1024, do_sample=False)
    answer = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)

    print("\n===== 답변 =====")
    print(answer)


if __name__ == "__main__":
    main()
