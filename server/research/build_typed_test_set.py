"""
generate.py/judge_compare.py의 "full vs chunks 컨텍스트" 비교 실험을 사건유형별로 나눠서
보기 위한 테스트셋 생성. val_query.json에는 사건유형 필드가 없어서, DB에서 조회해 붙여
새 파일로 만든다.

사건유형 22개 전부 대상으로 유형당 10개씩 뽑되, val_query.json에 10개가 안 되는 유형은
있는 만큼만 뽑는다(즉 총 개수는 220개가 아니라 실제로 뽑힌 합계가 됨 — 사건유형 분포가
매우 편향돼있어서 희귀 유형은 10개가 안 나옴, docs/log.md 2026-08-25 참고).

사용법:
    python build_typed_test_set.py
    python build_typed_test_set.py --per-type 10 --seed 42
"""
import argparse
import json
import os
import random
from collections import Counter, defaultdict

import psycopg2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # server/research/
VAL_PATH = os.path.join(BASE_DIR, "..", "..", "data", "Val", "val_query.json")
OUT_PATH = os.path.join(BASE_DIR, "..", "..", "data", "Val", "val_query_by_type.json")

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:lexchatbot_dev@127.0.0.1:5432/lexchatbot")
TABLE = "chunks_300_overlap100"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-type", type=int, default=10, help="사건유형당 뽑을 개수(상한)")
    parser.add_argument("--seed", type=int, default=42, help="샘플링 재현용 시드")
    args = parser.parse_args()

    with open(VAL_PATH, encoding="utf-8") as f:
        val_data = json.load(f)

    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()

    # 정답이 여러 개면 첫 번째만 사용 (prepare_data.py 등과 동일한 단순화)
    case_ids = list({item["case_ids"][0] for item in val_data})
    cur.execute(f"SELECT DISTINCT case_no, case_type FROM {TABLE} WHERE case_no = ANY(%s)", (case_ids,))
    type_by_case = dict(cur.fetchall())

    by_type = defaultdict(list)
    for item in val_data:
        case_id = item["case_ids"][0]
        case_type = type_by_case.get(case_id)
        if case_type:
            by_type[case_type].append({"query": item["query"], "case_id": case_id, "사건유형": case_type})

    rng = random.Random(args.seed)
    sampled = []
    counts = Counter()
    for case_type, items in sorted(by_type.items(), key=lambda kv: -len(kv[1])):
        k = min(args.per_type, len(items))
        picked = rng.sample(items, k)
        sampled.extend(picked)
        counts[case_type] = k

    print(f"사건유형 수: {len(counts)}, 총 샘플 수: {len(sampled)} (목표: 유형당 최대 {args.per_type}개)")
    print("--- 유형별 샘플 개수 ---")
    for case_type, k in counts.most_common():
        available = len(by_type[case_type])
        flag = "" if k == args.per_type else f"  <- val에 {available}개뿐이라 부족"
        print(f"{case_type}: {k}{flag}")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump({"counts": dict(counts), "data": sampled}, f, ensure_ascii=False, indent=2)
    print(f"\n저장: {OUT_PATH}")


if __name__ == "__main__":
    main()
