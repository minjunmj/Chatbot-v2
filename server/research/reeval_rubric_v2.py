"""
eval_answer_quality.py의 rubric을 보완해서 기존 195개 답변을 재평가한다.

기존 answer_correct는 "판례 내용에 비추어 정확한가"만 물어서, "판례 내용은 충실히
옮겼지만 정작 질문이 묻는 핵심 쟁점엔 제대로 답 안 함"인 경우를 놓칠 수 있다는 문제
제기(사용자, docs/log.md 참고)에 따라 두 축으로 분리:
  - grounded: 판례 원문에 실제로 있는 내용에 부합하고 지어낸 게 없는가(할루시네이션 없음)
  - resolves_question: 질문이 구체적으로 묻는 쟁점에 직접 대응해서 정확히 결론을 냈는가
    (판례를 나열만 하고 질문의 핵심은 안 짚었으면 false)
둘 다 true여야 "진짜 쓸만한 정답"으로 침.

actually_relevant/decline_justified는 정의가 안 바뀌었으니 기존 v1 판정을 그대로
재사용하고(재질문 안 함), **거절한 케이스는 애초에 채점할 답변이 없어서 GPT 호출 자체를
생략**하고 v1 값만 이어붙임 — grounded/resolves_question이 필요한 "답변을 실제로 생성한"
케이스에만 새로 GPT를 부른다. EXAONE 재생성도 없음(기존 answer 재사용).

사용법:
    python reeval_rubric_v2.py
"""
import json
import os

from generate import load_full_doc

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # server/
SRC_PATH = os.path.join(BASE_DIR, "answer_quality_results.jsonl")
OUT_PATH = os.path.join(BASE_DIR, "answer_quality_results_v2.jsonl")

# grounded/resolves_question 두 개만 새로 물음 — actually_relevant/decline_justified는
# v1 결과를 그대로 재사용(아래 main()에서 병합).
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


def build_judge_user_message(query, reference_docs, answer):
    ref_text = "\n\n---\n\n".join(
        f"[사건번호: {d['사건번호']}] {d['사건명']} ({d['법원명']}, {d['선고일자']})\n{d['판례내용']}"
        for d in reference_docs
    )
    return f"=== 질문 ===\n{query}\n\n=== top-5 판례 원문 ===\n{ref_text}\n\n=== 시스템이 생성한 답변 ===\n{answer}"


def call_judge(query, reference_docs, answer):
    from openai import OpenAI
    client = OpenAI()
    resp = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": build_judge_user_message(query, reference_docs, answer)},
        ],
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
    parser.add_argument("--limit", type=int, default=None, help="빠른 확인용 개수 제한")
    args = parser.parse_args()

    with open(SRC_PATH, encoding="utf-8") as f:
        src_records = [json.loads(line) for line in f]
    if args.limit:
        src_records = src_records[:args.limit]

    already_done = load_already_done(OUT_PATH)
    todo = [r for r in src_records if r["query"] not in already_done]
    print(f"전체 {len(src_records)}개 중 이미 완료 {len(already_done)}개, 남은 {len(todo)}개 실행")

    skipped, called = 0, 0
    with open(OUT_PATH, "a", encoding="utf-8") as f:
        for item in tqdm(todo, desc="rubric v2 재평가", ncols=80):
            v1_verdict = item.get("verdict") or {}
            actually_relevant = v1_verdict.get("actually_relevant")
            decline_justified = v1_verdict.get("decline_justified")

            if item["declined"]:
                # 거절한 케이스는 채점할 답변이 없음 — GPT 호출 없이 v1 판정만 이어붙임
                grounded, resolves_question, raw = None, None, None
                skipped += 1
            else:
                try:
                    reference_docs = [load_full_doc(cn) for cn in item["ranked"]]
                    verdict, raw = call_judge(item["query"], reference_docs, item["answer"])
                except Exception as e:  # noqa: BLE001
                    print(f"\n[에러] {item['query'][:30]}...: {e}")
                    continue
                grounded = verdict.get("grounded") if verdict else None
                resolves_question = verdict.get("resolves_question") if verdict else None
                called += 1

            rec = {
                "category": item["category"], "query": item["query"], "ranked": item["ranked"],
                "answer": item["answer"], "declined": item["declined"],
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

    print(f"GPT 새로 호출: {called}개, 거절 케이스라 재사용만: {skipped}개")
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

    print("\n===== 결과 (rubric v2: grounded / resolves_question 분리) =====")
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
