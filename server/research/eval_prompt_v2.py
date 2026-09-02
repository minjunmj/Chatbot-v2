"""
개선된 프롬프트(generate.py build_prompt v2 — "질문이 뭘 묻는지 먼저 파악하고 그것에
직접 답하라" 지시 추가, docs/log.md 2026-09-01 참고)로 기존 195개 쿼리(hit 100+miss 100,
answer_quality_results.jsonl과 동일한 질문/문서 세트)의 답변을 새로 생성하고, 개선 전
결과(grounded/resolves_question 분리 rubric, reeval_rubric_v2.py 결과)와 비교한다.

재사용/재계산 최적화:
  - ranked(top-5)와 actually_relevant는 프롬프트와 무관 → 그대로 재사용
  - decline_justified는 정의상 actually_relevant의 반대이므로 GPT 호출 없이 계산
  - 거절 여부는 새로 생성된 답변에 keyword 패턴(check_miss_rejection.py와 동일)을 다시 적용
  - grounded/resolves_question만 새 답변에 대해 GPT로 새로 판정

사용법:
    python eval_prompt_v2.py
    python eval_prompt_v2.py --limit 20
"""
import json
import os
import re

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from sentence_transformers import SentenceTransformer

from generate import (
    DENSE_MODEL_PATH, LLM_PATH, DEVICE,
    load_full_doc, build_context_chunks, generate_answer,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # server/
SRC_PATH = os.path.join(BASE_DIR, "answer_quality_results.jsonl")
OUT_PATH = os.path.join(BASE_DIR, "prompt_v2_results.jsonl")

# check_miss_rejection.py와 동일한 거절 탐지 패턴 재사용
DECLINE_PATTERNS = [
    r"답변(할|드리기|드릴)\s*수\s*없", r"찾을\s*수\s*없", r"확인(할|되지)\s*수?\s*없",
    r"판단(하기|이)\s*어렵", r"관련\s*(판례|자료).{0,10}(없|부족|찾지)",
    r"제공된\s*판례.{0,15}(없|부족|어렵|충분하지)", r"알\s*수\s*없",
]


def is_declined(answer):
    return any(re.search(p, answer) for p in DECLINE_PATTERNS)


JUDGE_SYSTEM = (
    "당신은 한국 법률 RAG 챗봇의 답변 품질을 검증하는 심사위원입니다. 아래 top-5 판례 "
    "원문과 질문, 그리고 시스템이 생성한 답변을 보고 두 가지를 판단하세요.\n\n"
    "1. grounded: 답변 내용이 판례 원문에 실제로 있는 내용에 부합하고 지어낸 게 없는지 "
    "— 즉 할루시네이션이 없는지(true/false)\n"
    "2. resolves_question: 답변이 판례 내용을 단순히 나열한 게 아니라 **질문이 구체적으로 "
    "묻는 쟁점에 직접 대응해서 정확한 결론을 냈는지**(true/false). 예를 들어 관련 판례 "
    "내용을 충실히 옮겼더라도 질문의 핵심 쟁점을 안 짚었거나 엉뚱한 결론을 냈다면 false\n\n"
    '반드시 JSON으로만 답하세요: {"grounded": true|false, "resolves_question": true|false, "reasoning": "..."}'
)


def call_judge(query, reference_docs, answer):
    from openai import OpenAI
    client = OpenAI()
    ref_text = "\n\n---\n\n".join(
        f"[사건번호: {d['사건번호']}] {d['사건명']} ({d['법원명']}, {d['선고일자']})\n{d['판례내용']}"
        for d in reference_docs
    )
    user_msg = f"=== 질문 ===\n{query}\n\n=== top-5 판례 원문 ===\n{ref_text}\n\n=== 시스템이 생성한 답변 ===\n{answer}"
    resp = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user_msg}],
    )
    raw = resp.choices[0].message.content
    try:
        return json.loads(raw), raw
    except json.JSONDecodeError:
        return None, raw


def load_already_done(out_path):
    done = set()
    if not os.path.exists(out_path):
        return done
    with open(out_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                done.add(json.loads(line)["query"])
    return done


def main():
    import argparse
    from tqdm import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    with open(SRC_PATH, encoding="utf-8") as f:
        src_records = [json.loads(line) for line in f]
    if args.limit:
        src_records = src_records[:args.limit]

    already_done = load_already_done(OUT_PATH)
    todo = [r for r in src_records if r["query"] not in already_done]
    print(f"전체 {len(src_records)}개 중 이미 완료 {len(already_done)}개, 남은 {len(todo)}개 실행")

    if not todo:
        summarize()
        return

    print("dense 모델 로딩...")  # reranker는 재검색 안 하니 불필요, dense만 있으면 chunk context 구성 가능
    dense_model = SentenceTransformer(DENSE_MODEL_PATH, device=DEVICE,
                                       model_kwargs={"torch_dtype": torch.float16} if DEVICE == "cuda" else None)
    import psycopg2
    conn = psycopg2.connect(os.environ.get("DATABASE_URL",
                                            "postgresql://postgres:lexchatbot_dev@127.0.0.1:5432/lexchatbot"))
    print(f"LLM 로딩: {LLM_PATH}")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(LLM_PATH)
    llm = AutoModelForCausalLM.from_pretrained(LLM_PATH, dtype=torch.bfloat16, device_map="auto")

    called, derived = 0, 0
    with open(OUT_PATH, "a", encoding="utf-8") as f:
        for item in tqdm(todo, desc="새 프롬프트로 재평가", ncols=80):
            try:
                actually_relevant = (item.get("verdict") or {}).get("actually_relevant")
                context = build_context_chunks(dense_model, conn, item["ranked"], item["query"])
                answer = generate_answer(tokenizer, llm, item["query"], context)
                declined = is_declined(answer)

                if declined:
                    grounded, resolves_question = None, None
                    decline_justified = (not actually_relevant) if actually_relevant is not None else None
                    raw = None
                    derived += 1
                else:
                    reference_docs = [load_full_doc(cn) for cn in item["ranked"]]
                    verdict, raw = call_judge(item["query"], reference_docs, answer)
                    grounded = verdict.get("grounded") if verdict else None
                    resolves_question = verdict.get("resolves_question") if verdict else None
                    decline_justified = None
                    called += 1
            except Exception as e:  # noqa: BLE001
                print(f"\n[에러] {item['query'][:30]}...: {e}")
                continue

            rec = {
                "category": item["category"], "query": item["query"], "ranked": item["ranked"],
                "answer": answer, "declined": declined,
                "verdict": {
                    "actually_relevant": actually_relevant,
                    "grounded": grounded,
                    "resolves_question": resolves_question,
                    "decline_justified": decline_justified,
                },
                "judge_raw": raw,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()

    print(f"GPT 새로 호출: {called}개, 거절이라 계산으로 대체: {derived}개")
    summarize()


def summarize():
    records = [json.loads(line) for line in open(OUT_PATH, encoding="utf-8")]
    hit_recs = [r for r in records if r["category"] == "hit"]
    miss_declined = [r for r in records if r["category"] == "miss" and r["declined"]]
    miss_answered = [r for r in records if r["category"] == "miss" and not r["declined"]]

    def rate(recs, key):
        vals = [r["verdict"].get(key) for r in recs if r.get("verdict")]
        vals = [v for v in vals if v is not None]
        return f"{sum(vals)}/{len(vals)} ({sum(vals)/len(vals)*100:.1f}%)" if vals else "N/A"

    def both_rate(recs):
        vals = [(r["verdict"].get("grounded"), r["verdict"].get("resolves_question"))
                for r in recs if r.get("verdict")]
        vals = [(g, q) for g, q in vals if g is not None and q is not None]
        ok = sum(1 for g, q in vals if g and q)
        return f"{ok}/{len(vals)} ({ok/len(vals)*100:.1f}%)" if vals else "N/A"

    print("\n===== 결과 (새 프롬프트, rubric v2) =====")
    print(f"[hit, n={len(hit_recs)}] grounded: {rate(hit_recs, 'grounded')}, "
          f"resolves_question: {rate(hit_recs, 'resolves_question')}, "
          f"둘다(진짜 정답): {both_rate(hit_recs)}")
    print(f"[miss-거절, n={len(miss_declined)}] decline_justified: {rate(miss_declined, 'decline_justified')}")
    relevant_answered = [r for r in miss_answered if r["verdict"] and r["verdict"].get("actually_relevant")]
    print(f"[miss-미거절, n={len(miss_answered)}] actually_relevant: {rate(miss_answered, 'actually_relevant')}")
    print(f"  (관련 있던 것 중, n={len(relevant_answered)}) grounded: {rate(relevant_answered, 'grounded')}, "
          f"resolves_question: {rate(relevant_answered, 'resolves_question')}, "
          f"둘다(진짜 정답): {both_rate(relevant_answered)}")


if __name__ == "__main__":
    main()
