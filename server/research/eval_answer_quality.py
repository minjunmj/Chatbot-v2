"""
hit 100개 + miss(LLM 거절/미거절 섞어서) 100개, 총 ~200개에 대해 GPT로 최종 답변 품질을
평가한다. 라벨(hit/miss)이 아니라 **실제 top-5 문서 내용 기준 진짜 관련성**을 GPT가
새로 판단하게 해서, 시스템의 3가지 가능한 행동(정상 답변/거절)이 실제로 옳았는지 본다
(docs/log.md 2026-08-31 threshold/거절 실험 후속 — 설계 논의 참고).

세 카테고리를 하나의 판정 스키마로 통일:
  - hit: 정답 문서가 top-5에 있었던 쿼리 → 답변 생성 후 정확성 평가
  - miss(미거절): top-5에 정답 라벨은 없지만 EXAONE이 확신 있게 답변한 쿼리 →
    "진짜 관련 문서가 있었는지" + "있었다면 답이 맞았는지" 평가
  - miss(거절): EXAONE이 "관련 판례 없음"이라고 거절한 쿼리 →
    "진짜 관련 문서가 없어서 거절이 타당했는지" 평가

GPT 판정 스키마(공통): {"actually_relevant": bool, "answer_correct": bool|null,
"decline_justified": bool|null, "reasoning": str}
- actually_relevant: top-5 판례 중 이 질문에 답할 근거가 되는 게 실제로 있는가 (라벨 무관)
- answer_correct: 시스템이 답변을 냈다면, 그 답변이 정확한가 (안 냈으면 null)
- decline_justified: 시스템이 거절했다면, 그 거절이 타당한가 (안 거절했으면 null,
  보통 actually_relevant의 반대여야 함)

사용법:
    python eval_answer_quality.py                    # hit 100 + miss(거절/미거절 섞어) 100
    python eval_answer_quality.py --hit-n 50 --miss-n 50
"""
import argparse
import json
import os
import random

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from tqdm import tqdm

from generate import load_models, retrieve_top_k, load_full_doc, build_context_chunks, generate_answer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # server/research/
CALIB_PATH = os.path.join(BASE_DIR, "..", "eval", "threshold_calibration.jsonl")
MISS_REJECTION_PATH = os.path.join(BASE_DIR, "miss_rejection_results.jsonl")
OUT_PATH = os.path.join(BASE_DIR, "answer_quality_results.jsonl")

JUDGE_SYSTEM = (
    "당신은 한국 법률 RAG 챗봇의 답변 품질을 검증하는 심사위원입니다. 아래 top-5 판례 "
    "원문과 질문, 그리고 시스템의 실제 행동(답변 생성 또는 거절)을 보고 판단하세요.\n\n"
    "1. actually_relevant: 미리 정해진 정답 라벨과 무관하게, 제공된 top-5 판례 중 실제로 "
    "이 질문에 답할 근거가 되는 것이 하나라도 있는지(true/false)\n"
    "2. answer_correct: 시스템이 답변을 생성했다면, 그 답변이 판례 내용에 비추어 정확하고 "
    "근거 있는지(true/false). 시스템이 답변을 생성하지 않았다면 null\n"
    "3. decline_justified: 시스템이 '관련 판례 없음'이라고 거절했다면, actually_relevant가 "
    "false라서 거절이 타당했는지(true/false). 시스템이 거절하지 않았다면 null\n\n"
    "반드시 JSON으로만 답하세요: "
    '{"actually_relevant": true|false, "answer_correct": true|false|null, '
    '"decline_justified": true|false|null, "reasoning": "..."}'
)


def build_judge_user_message(query, reference_docs, answer, declined):
    ref_text = "\n\n---\n\n".join(
        f"[사건번호: {d['사건번호']}] {d['사건명']} ({d['법원명']}, {d['선고일자']})\n{d['판례내용']}"
        for d in reference_docs
    )
    if declined:
        system_action = f"시스템은 '제공된 판례로는 답변할 수 없다'며 거절했습니다.\n원본 응답: {answer}"
    else:
        system_action = f"시스템이 생성한 답변:\n{answer}"
    return f"=== 질문 ===\n{query}\n\n=== top-5 판례 원문 ===\n{ref_text}\n\n=== 시스템 행동 ===\n{system_action}"


def call_judge(query, reference_docs, answer, declined):
    from openai import OpenAI
    client = OpenAI()
    resp = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": build_judge_user_message(query, reference_docs, answer, declined)},
        ],
    )
    raw = resp.choices[0].message.content
    try:
        return json.loads(raw), raw
    except json.JSONDecodeError:
        return None, raw


def load_hit_queries(n, seed):
    with open(CALIB_PATH, encoding="utf-8") as f:
        calib = [json.loads(line) for line in f]
    hits = [r["query"] for r in calib if r["recall5_hit"]]
    return random.Random(seed).sample(hits, min(n, len(hits)))


def load_miss_sample(n, seed):
    """miss_rejection_results.jsonl에서 거절한 것/못한 것을 최대한 균형 있게 섞어서 n개
    뽑는다 — 거절 비율이 낮으면(스모크 테스트에서 10%) 순수 랜덤은 거절 케이스가 너무
    적게 뽑혀서 '거절이 타당했는지' 쪽 평가가 부실해지므로, 반씩 채우는 걸 우선함."""
    if not os.path.exists(MISS_REJECTION_PATH):
        raise FileNotFoundError(f"{MISS_REJECTION_PATH} 없음 — check_miss_rejection.py를 먼저 완료해야 함")
    with open(MISS_REJECTION_PATH, encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
    declined = [r for r in records if r["declined"]]
    not_declined = [r for r in records if not r["declined"]]

    rng = random.Random(seed)
    half = n // 2
    picked_declined = rng.sample(declined, min(half, len(declined)))
    remaining = n - len(picked_declined)
    picked_not_declined = rng.sample(not_declined, min(remaining, len(not_declined)))
    remaining2 = n - len(picked_declined) - len(picked_not_declined)
    if remaining2 > 0:  # declined이 부족했으면 not_declined에서 더 채움
        extra_pool = [r for r in not_declined if r not in picked_not_declined]
        picked_not_declined += rng.sample(extra_pool, min(remaining2, len(extra_pool)))

    print(f"miss 샘플: 거절 {len(picked_declined)}개 + 미거절 {len(picked_not_declined)}개")
    return picked_declined + picked_not_declined


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hit-n", type=int, default=100)
    parser.add_argument("--miss-n", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    hit_queries = load_hit_queries(args.hit_n, args.seed)
    miss_records = load_miss_sample(args.miss_n, args.seed)
    print(f"hit {len(hit_queries)}개 + miss {len(miss_records)}개 = 총 {len(hit_queries) + len(miss_records)}개")

    dense_model, retriever, tokenizer, llm, conn = load_models()

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        # hit: 아직 답변 생성 전이라 여기서 새로 생성
        for query in tqdm(hit_queries, desc="hit 답변 생성+평가", ncols=80):
            try:
                ranked = retrieve_top_k(retriever, query, args.top_k)
                reference_docs = [load_full_doc(cn) for cn in ranked]
                context = build_context_chunks(dense_model, conn, ranked, query)
                answer = generate_answer(tokenizer, llm, query, context)
                verdict, raw = call_judge(query, reference_docs, answer, declined=False)
            except Exception as e:  # noqa: BLE001
                print(f"\n[에러] {query[:30]}...: {e}")
                continue
            rec = {"category": "hit", "query": query, "ranked": ranked, "answer": answer,
                   "declined": False, "verdict": verdict, "judge_raw": raw}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()

        # miss: 이미 생성해둔 답변/거절 결과 재사용
        for item in tqdm(miss_records, desc="miss 평가", ncols=80):
            try:
                reference_docs = [load_full_doc(cn) for cn in item["ranked"]]
                verdict, raw = call_judge(item["query"], reference_docs, item["answer"], item["declined"])
            except Exception as e:  # noqa: BLE001
                print(f"\n[에러] {item['query'][:30]}...: {e}")
                continue
            rec = {"category": "miss", "query": item["query"], "ranked": item["ranked"],
                   "answer": item["answer"], "declined": item["declined"],
                   "verdict": verdict, "judge_raw": raw}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()

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

    print("\n===== 결과 =====")
    print(f"[hit, n={len(hit_recs)}] answer_correct: {rate(hit_recs, 'answer_correct')}")
    print(f"[miss-거절, n={len(miss_declined)}] decline_justified: {rate(miss_declined, 'decline_justified')}")
    print(f"[miss-미거절, n={len(miss_answered)}] actually_relevant: {rate(miss_answered, 'actually_relevant')}, "
          f"answer_correct(관련 있었던 것 중): "
          f"{rate([r for r in miss_answered if r['verdict'] and r['verdict'].get('actually_relevant')], 'answer_correct')}")


if __name__ == "__main__":
    main()
