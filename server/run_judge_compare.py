"""
val_query_by_type.json(사건유형별 샘플, build_typed_test_set.py 산출물)의 각 쿼리마다
full/chunks 두 컨텍스트 방식으로 답변을 생성하고 GPT-5-mini judge로 비교한 뒤,
1) 쿼리별 원본 기록을 JSONL로 저장하고 2) 마지막에 집계(전체 + 사건유형별) 결과를 출력한다.

judge_compare.py의 단일 쿼리 로직을 재사용하되, 모델을 한 번만 로드해서 171개를 순회한다.
중간에 중단돼도 이미 처리한 (query, case_id)는 건너뛰고 이어서 실행됨(재실행 시 자동 resume).

사용법:
    python run_judge_compare.py
    python run_judge_compare.py --limit 20      # 시간 가늠용 소규모 실행
    python run_judge_compare.py --out my_results.jsonl
"""
import argparse
import json
import os
import random
from collections import Counter, defaultdict

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from tqdm import tqdm

from generate import (
    load_models, retrieve_top_k, load_full_doc,
    build_context_full, build_context_chunks, generate_answer,
)
from judge_compare import call_judge

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # server/
TYPED_VAL_PATH = os.path.join(BASE_DIR, "..", "data", "Val", "val_query_by_type.json")
DEFAULT_OUT_PATH = os.path.join(BASE_DIR, "judge_results.jsonl")

CRITERIA = ("more_accurate", "more_grounded", "more_correct_citation", "overall_better")


def load_already_done(out_path):
    done = set()
    if not os.path.exists(out_path):
        return done
    with open(out_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                done.add((rec["query"], rec["case_id"]))
    return done


def run_one(dense_model, retriever, tokenizer, llm, conn, item, top_k, seed):
    query, case_id, case_type = item["query"], item["case_id"], item["사건유형"]

    ranked = retrieve_top_k(retriever, query, top_k)
    reference_docs = [load_full_doc(cn) for cn in ranked]

    context_full = build_context_full(ranked)
    context_chunks = build_context_chunks(dense_model, conn, ranked, query)

    answer_full = generate_answer(tokenizer, llm, query, context_full)
    answer_chunks = generate_answer(tokenizer, llm, query, context_chunks)

    rng = random.Random(seed)
    if rng.random() < 0.5:
        label_map = {"A": "full", "B": "chunks"}
        answer_a, answer_b = answer_full, answer_chunks
    else:
        label_map = {"A": "chunks", "B": "full"}
        answer_a, answer_b = answer_chunks, answer_full

    verdict, raw = call_judge(query, reference_docs, answer_a, answer_b)

    resolved = {}
    if verdict:
        for key in CRITERIA:
            v = verdict.get(key)
            resolved[key] = label_map.get(v, v)

    return {
        "query": query,
        "case_id": case_id,
        "사건유형": case_type,
        "ranked": ranked,
        "answer_full": answer_full,
        "answer_chunks": answer_chunks,
        "label_map": label_map,
        "judge_raw": raw,
        "judge_verdict_resolved": resolved,
        "judge_reasoning": verdict.get("reasoning") if verdict else None,
    }


def summarize(out_path):
    overall = {c: Counter() for c in CRITERIA}
    by_type = defaultdict(lambda: {c: Counter() for c in CRITERIA})
    n = 0
    with open(out_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            n += 1
            resolved = rec.get("judge_verdict_resolved") or {}
            for c in CRITERIA:
                v = resolved.get(c)
                if v in ("full", "chunks", "tie"):
                    overall[c][v] += 1
                    by_type[rec["사건유형"]][c][v] += 1

    print(f"\n===== 전체 집계 (n={n}) =====")
    for c in CRITERIA:
        print(f"{c}: {dict(overall[c])}")

    print("\n===== 사건유형별 overall_better =====")
    for case_type, counters in sorted(by_type.items(), key=lambda kv: -sum(kv[1]["overall_better"].values())):
        cnt = counters["overall_better"]
        total = sum(cnt.values())
        print(f"{case_type} (n={total}): {dict(cnt)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="빠른 확인용 쿼리 수 제한")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--out", default=DEFAULT_OUT_PATH, help="결과 JSONL 저장 경로")
    parser.add_argument("--seed", type=int, default=0, help="A/B 순서 셔플 기준 시드")
    args = parser.parse_args()

    with open(TYPED_VAL_PATH, encoding="utf-8") as f:
        typed_data = json.load(f)["data"]
    if args.limit:
        typed_data = typed_data[:args.limit]

    already_done = load_already_done(args.out)
    todo = [item for item in typed_data if (item["query"], item["case_id"]) not in already_done]
    print(f"전체 {len(typed_data)}개 중 이미 완료 {len(already_done)}개, 남은 {len(todo)}개 실행")

    if todo:
        dense_model, retriever, tokenizer, llm, conn = load_models()
        with open(args.out, "a", encoding="utf-8") as f:
            for i, item in enumerate(tqdm(todo, desc="judge 비교", ncols=80)):
                try:
                    rec = run_one(dense_model, retriever, tokenizer, llm, conn, item, args.top_k, args.seed + i)
                except Exception as e:  # noqa: BLE001 — 배치 중 하나 실패해도 나머지는 계속 진행
                    print(f"\n[에러] query={item['query'][:30]}... : {e}")
                    continue
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()

    summarize(args.out)


if __name__ == "__main__":
    main()
