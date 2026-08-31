"""
"관련 판례 없음" 판단을 reranker의 cross-encoder 점수 threshold로 걸러도 되는지 검증.

val_query.json 전체를 CrossEncoderReranker.retrieve_batch_with_scores로 돌려서, 각 쿼리의
1등 점수(top1_score)와 "recall@5 성공/실패 여부"(top-5 안에 정답 case_id가 있었는지)를
같이 기록한다. 그 다음 threshold를 낮은 값부터 천천히 올려가며:
  - recall@5 성공 케이스 중 몇 개가 걸러지는지 (false rejection — 낮을수록 좋음,
    실제로 답할 수 있었는데 "모른다"고 해버리는 경우)
  - recall@5 실패 케이스 중 몇 개가 걸러지는지 (correct rejection — 높을수록 좋음,
    애초에 못 찾았던 걸 "모른다"고 정직하게 인정하는 경우)
를 보여준다. 점수는 sigmoid 활성화라 대략 0~1 범위(실측: 관련 질문 0.66, "오늘 날씨 어때?"
같은 무관 질문 0.23 — docs/log.md 참고).

사용법:
    python calibrate_threshold.py                  # val_query.json 전체 (7,280개, ~1시간)
    python calibrate_threshold.py --limit 500       # 빠른 확인용
"""
import argparse
import json
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import psycopg2
import torch
from sentence_transformers import SentenceTransformer

from retrievers import CrossEncoderReranker

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # server/
VAL_PATH = os.path.join(BASE_DIR, "..", "data", "Val", "val_query.json")
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "threshold_calibration.jsonl")

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:lexchatbot_dev@127.0.0.1:5432/lexchatbot")
TABLE = "chunks_300_overlap100"
DENSE_MODEL_PATH = os.path.join(BASE_DIR, "finetune", "output", "kure-v1-finetuned-hard")
RERANK_MODEL_PATH = os.path.join(BASE_DIR, "finetune", "output", "bge-reranker-v2-m3-finetuned")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TOP_K = 5
TOP_N_RERANK = 30
EF_SEARCH = 200
BATCH_SIZE = 64


def run_and_save(limit):
    with open(VAL_PATH, encoding="utf-8") as f:
        val_data = json.load(f)
    if limit:
        val_data = val_data[:limit]
    print(f"쿼리 수: {len(val_data)}")

    dense_model = SentenceTransformer(DENSE_MODEL_PATH, device=DEVICE,
                                       model_kwargs={"torch_dtype": torch.float16} if DEVICE == "cuda" else None)
    conn = psycopg2.connect(DATABASE_URL)
    retriever = CrossEncoderReranker(dense_model, RERANK_MODEL_PATH, conn, TABLE,
                                      top_n=TOP_N_RERANK, ef_search=EF_SEARCH, device=DEVICE)

    from tqdm import tqdm
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for start in tqdm(range(0, len(val_data), BATCH_SIZE), desc="calibrate", ncols=80):
            batch = val_data[start:start + BATCH_SIZE]
            queries = [item["query"] for item in batch]
            ranked_list, top1_scores = retriever.retrieve_batch_with_scores(queries, TOP_K)
            for item, ranked, score in zip(batch, ranked_list, top1_scores):
                true_ids = set(item["case_ids"])
                hit = any(cid in true_ids for cid in ranked)
                f.write(json.dumps({"query": item["query"], "top1_score": score, "recall5_hit": hit},
                                    ensure_ascii=False) + "\n")
    print(f"저장: {OUT_PATH}")


def sweep(thresholds):
    hits, misses = [], []
    with open(OUT_PATH, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            (hits if rec["recall5_hit"] else misses).append(rec["top1_score"])
    hits, misses = np.array(hits), np.array(misses)
    print(f"recall@5 성공(hit): {len(hits)}개, 실패(miss): {len(misses)}개")
    print(f"hit 점수 분포: min={hits.min():.3f} 25%={np.percentile(hits,25):.3f} "
          f"중앙값={np.median(hits):.3f} 75%={np.percentile(hits,75):.3f} max={hits.max():.3f}")
    print(f"miss 점수 분포: min={misses.min():.3f} 25%={np.percentile(misses,25):.3f} "
          f"중앙값={np.median(misses):.3f} 75%={np.percentile(misses,75):.3f} max={misses.max():.3f}")

    print(f"\n{'threshold':>10} | {'걸러진 hit(놓침)':>16} | {'걸러진 miss(정직하게 거절)':>22}")
    for t in thresholds:
        filtered_hits = int((hits < t).sum())
        filtered_misses = int((misses < t).sum())
        print(f"{t:>10.2f} | {filtered_hits:>6}/{len(hits)} ({filtered_hits/len(hits)*100:5.1f}%) | "
              f"{filtered_misses:>6}/{len(misses)} ({filtered_misses/len(misses)*100:5.1f}%)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="빠른 확인용 쿼리 수 제한")
    parser.add_argument("--skip-run", action="store_true",
                         help="이미 저장된 threshold_calibration.jsonl로 sweep만 다시 출력")
    parser.add_argument("--thresholds", type=str, default="0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.6,0.7",
                         help="스윕할 threshold 목록, 콤마 구분")
    args = parser.parse_args()

    if not args.skip_run:
        run_and_save(args.limit)

    thresholds = [float(x) for x in args.thresholds.split(",")]
    sweep(thresholds)


if __name__ == "__main__":
    main()
