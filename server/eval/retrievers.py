"""
harness.py의 evaluate()에 꽂아 쓰는 Retriever 구현체들. 새 검색 방법(sparse, hybrid, rerank
등)을 추가할 때는 여기에 retrieve_batch(queries, max_k) -> list[list[str]]를 구현하는
클래스만 추가하면 되고, harness.py는 안 건드려도 됨.

공통 규칙: retrieve_batch가 반환하는 리스트는 각 쿼리마다 "문서(case_no) 단위로 중복 제거된
순위 리스트"여야 함 — 여러 chunk가 같은 문서에서 나와도 그 문서는 한 번만, 제일 점수 높은
순서로.
"""
import numpy as np
import torch


def l2norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    np.maximum(n, 1e-12, out=n)
    return x / n


class DenseExactRetriever:
    """numpy로 corpus 전체와 exact(brute-force) 유사도 검색 — kure_chunk.py/kure_chunk_overlap.py/
    finetune/eval_model.py가 하던 방식과 동일. pgvector 없이 정확도만 빠르게 스크리닝할 때 사용."""

    def __init__(self, model, chunk_case_ids, chunk_texts=None, doc_embs=None,
                 pool_k=500, device="cuda", query_batch_size=32):
        self.model = model
        self.chunk_case_ids = np.array(chunk_case_ids)
        self.pool_k = min(pool_k, len(chunk_case_ids))
        self.device = device
        self.query_batch_size = query_batch_size

        if doc_embs is None:
            if chunk_texts is None:
                raise ValueError("chunk_texts 또는 doc_embs 둘 중 하나는 필요함")
            doc_embs = model.encode(chunk_texts, batch_size=32, show_progress_bar=True,
                                     convert_to_numpy=True, normalize_embeddings=True)
        self.doc_t = torch.from_numpy(l2norm(doc_embs.astype(np.float32))).to(device)

    def retrieve_batch(self, queries, max_k):
        query_embs = l2norm(self.model.encode(
            queries, batch_size=self.query_batch_size, show_progress_bar=False,
            convert_to_numpy=True, normalize_embeddings=True,
        ).astype(np.float32))
        query_t = torch.from_numpy(query_embs).to(self.device)

        sims = query_t @ self.doc_t.T
        _, top_idx = torch.topk(sims, k=self.pool_k, dim=1)
        top_idx = top_idx.cpu().numpy()

        results = []
        for row in top_idx:
            seen = set()
            ranked = []
            for j in row:
                cid = self.chunk_case_ids[j]
                if cid not in seen:
                    seen.add(cid)
                    ranked.append(cid)
                    if len(ranked) >= max_k:
                        break
            results.append(ranked)
        return results


class PgvectorRetriever:
    """Postgres+pgvector 테이블(HNSW 인덱스)로 검색 — test_val_pgvector.py가 하던 방식.
    실제 서빙 latency까지 재고 싶을 때 사용. 쿼리마다 SQL 왕복이 발생하므로 DenseExactRetriever
    보다 느림(그게 정상 — 근사 인덱스 실측 latency를 보는 게 목적)."""

    def __init__(self, model, conn, table, pool_k=500, ef_search=100, query_batch_size=32):
        self.model = model
        self.conn = conn
        self.table = table
        self.pool_k = pool_k
        self.query_batch_size = query_batch_size
        with self.conn.cursor() as cur:
            cur.execute("SET hnsw.ef_search = %s", (ef_search,))

    @staticmethod
    def _pg_vector_literal(vec):
        return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"

    def retrieve_batch(self, queries, max_k):
        query_embs = l2norm(self.model.encode(
            queries, batch_size=self.query_batch_size, show_progress_bar=False,
            convert_to_numpy=True, normalize_embeddings=True,
        ).astype(np.float32))

        results = []
        with self.conn.cursor() as cur:
            for vec in query_embs:
                lit = self._pg_vector_literal(vec)
                cur.execute(
                    f"WITH nearest AS ("
                    f"    SELECT case_no, embedding <=> %s AS dist FROM {self.table} "
                    f"    ORDER BY embedding <=> %s LIMIT {self.pool_k}"
                    f") "
                    f"SELECT DISTINCT ON (case_no) case_no, dist FROM nearest ORDER BY case_no, dist",
                    (lit, lit),
                )
                rows = sorted(cur.fetchall(), key=lambda r: r[1])[:max_k]
                results.append([r[0] for r in rows])
        return results
