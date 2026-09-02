"""
threshold=0.2로 안 걸러지는 miss 쿼리들(recall@5 실패 + top1_score>=0.2)에 대해, 실제
답변 생성 모델인 EXAONE이 "관련 판례를 못 찾았다"고 솔직히 인정하는지, 아니면 (틀린 문서를
근거로) 확신 있게 답을 지어내는지 확인한다. threshold 단독 필터가 못 잡는 애매한 영역을
LLM이 얼마나 커버해주는지 보는 실험 (docs/log.md 2026-08-31 threshold 실험의 후속).

거절 여부 판정은 키워드 기반 규칙으로 함(GPT API 비용/의존성 없이 빠르게 확인하려는 목적 —
답변을 생성하는 실제 모델은 어디까지나 EXAONE 하나뿐이고, 이건 그 출력을 카운트만 하는
보조 로직). 완벽하진 않지만(모델이 예상 밖 표현으로 거절할 경우 놓칠 수 있음), 결과가
애매하면 나중에 GPT judge로 재검증할 수 있게 원본 답변을 전부 저장해둠.

사용법:
    python check_miss_rejection.py --limit 30      # 먼저 소규모로 확인
    python check_miss_rejection.py                  # threshold=0.2로 안 걸러지는 490개 전체
"""
import argparse
import json
import os
import re

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from tqdm import tqdm

from generate import load_models, retrieve_top_k, build_context_chunks, generate_answer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # server/research/
CALIB_PATH = os.path.join(BASE_DIR, "..", "eval", "threshold_calibration.jsonl")
OUT_PATH = os.path.join(BASE_DIR, "miss_rejection_results.jsonl")

# generate.py의 build_prompt() 지시("제공된 판례로 답할 수 없으면 그렇게 솔직히 답하세요")를
# 따를 때 모델이 실제로 쓸 법한 표현들. 완전한 목록은 아니라서, 애매한 경우는
# judge_raw(원본 답변)를 나중에 직접 읽거나 GPT judge로 재검증 가능하게 다 저장해둠.
DECLINE_PATTERNS = [
    r"답변(할|드리기|드릴)\s*수\s*없", r"찾을\s*수\s*없", r"확인(할|되지)\s*수?\s*없",
    r"판단(하기|이)\s*어렵", r"관련\s*(판례|자료).{0,10}(없|부족|찾지)",
    r"제공된\s*판례.{0,15}(없|부족|어렵|충분하지)", r"알\s*수\s*없",
]


def classify_decline(answer):
    matched = [p for p in DECLINE_PATTERNS if re.search(p, answer)]
    return bool(matched), matched


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="빠른 확인용 쿼리 수 제한")
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    with open(CALIB_PATH, encoding="utf-8") as f:
        calib = [json.loads(line) for line in f]
    targets = [r for r in calib if not r["recall5_hit"] and r["top1_score"] >= args.threshold]
    print(f"threshold={args.threshold}로 안 걸러지는 miss: {len(targets)}개")
    if args.limit:
        targets = targets[:args.limit]

    already_done = load_already_done(OUT_PATH)
    todo = [t for t in targets if t["query"] not in already_done]
    print(f"이미 완료 {len(already_done)}개, 남은 {len(todo)}개 실행")

    if todo:
        dense_model, retriever, tokenizer, llm, conn = load_models()
        with open(OUT_PATH, "a", encoding="utf-8") as f:
            for item in tqdm(todo, desc="miss 거절 판단", ncols=80):
                query = item["query"]
                try:
                    ranked = retrieve_top_k(retriever, query, args.top_k)
                    context = build_context_chunks(dense_model, conn, ranked, query)
                    answer = generate_answer(tokenizer, llm, query, context)
                    declined, matched_patterns = classify_decline(answer)
                except Exception as e:  # noqa: BLE001
                    print(f"\n[에러] {query[:30]}...: {e}")
                    continue
                rec = {
                    "query": query,
                    "top1_score": item["top1_score"],
                    "ranked": ranked,
                    "answer": answer,
                    "declined": declined,
                    "matched_patterns": matched_patterns,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()

    declined = sum(1 for line in open(OUT_PATH, encoding="utf-8") if json.loads(line).get("declined") is True)
    total = sum(1 for _ in open(OUT_PATH, encoding="utf-8"))
    print(f"\n===== 결과 =====")
    if total == 0:
        print("완료된 기록이 없음 — 위 에러 메시지 확인 필요")
    else:
        print(f"전체 {total}개 중 솔직히 거절: {declined}개 ({declined/total*100:.1f}%), "
              f"확신 있게 (틀린) 답 지어냄: {total-declined}개 ({(total-declined)/total*100:.1f}%)")


if __name__ == "__main__":
    main()
