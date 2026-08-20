"""
재사용 가능한 retrieval 평가 하니스.

지금까지 model_test.py/jhgan.py/bge_hybrid.py/kure_chunk.py/kure_chunk_overlap.py/
test_val_pgvector.py/finetune/eval_model.py가 전부 "val_query.json 읽어서 → 검색 →
recall@k/mrr@10 계산 → 출력"이라는 같은 로직을 매번 새로 짰음(docs/log.md 2026-08-20 논의).
이 파일이 그 공통 로직을 한 곳에 모은 것 — 새 검색 방법(sparse, hybrid, rerank 등)을 테스트할
때는 retrievers.py에 Retriever 하나만 새로 구현하면 되고, 이 harness.py는 안 건드림.

사용법은 run_eval.py 참고.
"""
import time

import numpy as np


def evaluate(retriever, val_data, k_list=(1, 5, 10, 20), mrr_k=10, batch_size=64, measure_latency=False):
    """
    retriever: retrieve_batch(queries: list[str], max_k: int) -> list[list[str]] 메서드를 가진 객체.
               각 쿼리에 대해 문서(case_no) 단위로 dedupe된 순위 리스트를 유사도 순으로 반환해야 함.
    val_data: [{"query": ..., "case_ids": [...]}, ...] (val_query.json 형식)
    """
    max_k = max(list(k_list) + [mrr_k])
    recall_hits = {k: 0 for k in k_list}
    mrr_sum = 0.0
    n = len(val_data)
    latencies_ms = []

    for start in range(0, n, batch_size):
        batch = val_data[start:start + batch_size]
        queries = [item["query"] for item in batch]

        t0 = time.perf_counter()
        results = retriever.retrieve_batch(queries, max_k)
        elapsed = time.perf_counter() - t0
        if measure_latency:
            latencies_ms.append(elapsed / len(queries) * 1000)

        for ranked_case_ids, item in zip(results, batch):
            true_ids = set(item["case_ids"])
            for k in k_list:
                if any(cid in true_ids for cid in ranked_case_ids[:k]):
                    recall_hits[k] += 1
            for rank, cid in enumerate(ranked_case_ids[:mrr_k], start=1):
                if cid in true_ids:
                    mrr_sum += 1.0 / rank
                    break

    result = {f"recall@{k}": round(recall_hits[k] / n, 4) for k in k_list}
    result[f"mrr@{mrr_k}"] = round(mrr_sum / n, 4)
    if measure_latency and latencies_ms:
        arr = np.array(latencies_ms)
        result["latency_mean_ms"] = round(float(arr.mean()), 2)
        result["latency_p50_ms"] = round(float(np.percentile(arr, 50)), 2)
        result["latency_p95_ms"] = round(float(np.percentile(arr, 95)), 2)
        result["latency_p99_ms"] = round(float(np.percentile(arr, 99)), 2)
    return result
