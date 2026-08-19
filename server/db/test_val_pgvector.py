"""
pgvector에 적재된 chunk 테이블(--table로 임의 지정, 예: chunks_200, chunks_300_overlap100)을
val_query.json으로 검증한다.

두 가지를 확인:
1. 정확도 sanity check — kure_chunk.py가 numpy로 계산했던 recall@k/mrr@10과 거의 같은 값이
   나오는지 (다르면 적재/인덱스 쪽에 버그가 있다는 뜻)
2. 실제 검색 latency — HNSW 인덱스를 통한 SQL 쿼리 1건이 평균/p50/p95 몇 ms 걸리는지
   (chunk 크기 최종 결정을 위한 트레이드오프 자료, docs/PROJECT_STATE.md 5절 참고)

주의: HNSW는 근사 최근접 탐색(ANN)이라 kure_chunk.py의 정확한(exact) top-k 계산과
100% 동일한 값이 안 나올 수 있음 — 소폭 차이는 정상, 크게 벗어나면 문제로 봐야 함.

사용법:
    python test_val_pgvector.py --table chunks_200_overlap50
    python test_val_pgvector.py --table chunks_300_overlap100 --limit 500   # 빠른 확인용 쿼리 수 제한
"""
import argparse
import json
import os
import time

import numpy as np
import psycopg2
import torch
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # server/
VAL_PATH = os.path.join(BASE_DIR, "..", "data", "Val", "val_query.json")

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:lexchatbot_dev@127.0.0.1:5432/lexchatbot")
REPO = "nlpai-lab/KURE-v1"
K_LIST = [1, 5, 10, 20]
MRR_K = 10
DEDUPE_POOL = 500  # HNSW로 먼저 뽑아올 최근접 chunk 후보 수 (kure_chunk.py의 pool_k=500과 동일하게 맞춤)


def l2norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    np.maximum(n, 1e-12, out=n)
    return x / n


def pg_vector_literal(vec):
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", required=True,
                         help="검증할 pgvector 테이블명 (예: chunks_200, chunks_300_overlap100)")
    parser.add_argument("--limit", type=int, default=None, help="테스트할 쿼리 수 제한(기본: 전체 7280개)")
    args = parser.parse_args()
    table = args.table

    with open(VAL_PATH, encoding="utf-8") as f:
        val_data = json.load(f)
    if args.limit:
        val_data = val_data[:args.limit]
    print(f"쿼리 수: {len(val_data)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(REPO, device=device,
                                 model_kwargs={"torch_dtype": torch.float16} if device == "cuda" else None)
    query_embs = l2norm(model.encode(
        [item["query"] for item in val_data], batch_size=32, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=True,
    ).astype(np.float32))

    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    # HNSW 검색 품질 파라미터 (기본값보다 넉넉히) — ef_search가 클수록 정확하지만 느려짐
    cur.execute("SET hnsw.ef_search = 100")

    max_k = max(K_LIST + [MRR_K])
    recall_hits = {k: 0 for k in K_LIST}
    mrr_sum = 0.0
    latencies_ms = []

    for i, item in enumerate(val_data):
        true_ids = set(item["case_ids"])
        vec_literal = pg_vector_literal(query_embs[i])

        t0 = time.perf_counter()
        cur.execute(
            # 1) HNSW 인덱스로 진짜 최근접 chunk DEDUPE_POOL개를 먼저 뽑고(ORDER BY 거리 LIMIT),
            # 2) 그 후보군 안에서만 case_no 기준으로 dedupe (DISTINCT ON 문법상 ORDER BY가
            #    case_no로 시작해야 해서, 전체 테이블에 바로 걸면 거리 순이 아니라 사건번호
            #    순으로 잘려버림 — 그래서 반드시 서브쿼리로 먼저 거리순 pool을 확정해야 함)
            f"WITH nearest AS ("
            f"    SELECT case_no, embedding <=> %s AS dist FROM {table} "
            f"    ORDER BY embedding <=> %s LIMIT {DEDUPE_POOL}"
            f") "
            f"SELECT DISTINCT ON (case_no) case_no, dist FROM nearest ORDER BY case_no, dist",
            (vec_literal, vec_literal),
        )
        rows = cur.fetchall()
        latencies_ms.append((time.perf_counter() - t0) * 1000)

        ranked_case_ids = [r[0] for r in sorted(rows, key=lambda r: r[1])][:max_k]

        for k in K_LIST:
            if any(cid in true_ids for cid in ranked_case_ids[:k]):
                recall_hits[k] += 1
        for rank, cid in enumerate(ranked_case_ids[:MRR_K], start=1):
            if cid in true_ids:
                mrr_sum += 1.0 / rank
                break

        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(val_data)}")

    conn.close()
    n = len(val_data)
    lat = np.array(latencies_ms)

    print(f"\n===== {table} — pgvector 검증 결과 =====")
    for k in K_LIST:
        print(f"recall@{k}: {recall_hits[k] / n:.4f}")
    print(f"mrr@{MRR_K}: {mrr_sum / n:.4f}")
    print(f"\nlatency(ms) — mean: {lat.mean():.2f}, p50: {np.percentile(lat, 50):.2f}, "
          f"p95: {np.percentile(lat, 95):.2f}, p99: {np.percentile(lat, 99):.2f}")
    print("\n(참고: kure_chunk.py의 numpy exact-search 결과와 소폭 다를 수 있음 — HNSW는 근사 탐색)")


if __name__ == "__main__":
    main()
