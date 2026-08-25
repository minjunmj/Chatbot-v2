"""
harness.py의 evaluate()에 꽂아 쓰는 Retriever 구현체들. 새 검색 방법(sparse, hybrid, rerank
등)을 추가할 때는 여기에 retrieve_batch(queries, max_k) -> list[list[str]]를 구현하는
클래스만 추가하면 되고, harness.py는 안 건드려도 됨.

공통 규칙: retrieve_batch가 반환하는 리스트는 각 쿼리마다 "문서(case_no) 단위로 중복 제거된
순위 리스트"여야 함 — 여러 chunk가 같은 문서에서 나와도 그 문서는 한 번만, 제일 점수 높은
순서로.
"""
import math
from collections import Counter

import numpy as np
import torch
from tqdm import tqdm


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


# server/db/build_vector_db.py의 KIWI_KEEP_PREFIXES와 반드시 동일해야 함 — 문서 쪽(DB 적재
# 시)과 검색어 쪽(여기)의 토큰화 방식이 다르면 tsvector 매칭 자체가 안 됨(2026-08-21 실측
# 확인: "손해배상"을 Kiwi 없이 그대로 검색하면 "손해"+"배상"으로 쪼개 저장된 문서와 매칭 실패).
KIWI_KEEP_PREFIXES = ("NN", "NP", "NR", "VV", "VA", "VX", "MAG", "MAJ", "SL", "SH", "SN", "XR")


class SparseRetriever:
    """Postgres 내장 전문검색(tsvector/GIN) + Kiwi 형태소 분석으로 후보를 빠르게 추리고,
    그 후보들만 진짜 BM25 공식(IDF + tf 포화 + 문서길이 보정)으로 재점수/재정렬하는 sparse 검색.
    build_vector_db.py가 content_tsv/chunk_text_kiwi 컬럼을 채워둔 테이블이 필요함.

    Postgres의 ts_rank_cd는 진짜 BM25가 아님(IDF/k1/b 없이 단순 위치·빈도 가중치라 학습도
    튜닝도 안 됨) — 그래서 1단계(SQL, GIN)로는 후보만 빠르게 뽑고, 2단계(Python)에서
    표준 Okapi BM25로 다시 점수를 매김:
        score(q,d) = sum_t IDF(t) * tf(t,d)*(k1+1) / (tf(t,d) + k1*(1-b+b*|d|/avgdl))
    IDF/avgdl은 ts_stat()으로 코퍼스 전체를 한 번 스캔해 __init__에서 미리 계산해둠(쿼리마다
    다시 계산 안 함).

    너무 흔한 단어(예: "회사"/"경우"/"이유" 등 법률 문서 어디에나 있는 단어)가 1단계 OR
    검색어에 섞이면 매칭이 전체의 절반 이상으로 폭발해서 GIN 인덱스를 안 쓰고 순차 스캔(seq
    scan)으로 떨어짐 — 쿼리 1개에 5초씩 걸리는 문제로 실측 확인(2026-08-24). 그래서 문서빈도
    (ndoc) 상위 top_common_pct%(기본 3%)를 "너무 흔한 단어"로 걸러내고 1단계 후보 검색에서만
    제외(실제 BM25 점수 계산에는 여전히 전체 코퍼스 기준 IDF를 그대로 사용 — 흔한 단어는
    IDF가 낮아서 어차피 점수에 거의 기여 안 하므로 걸러내도 랭킹 품질 손실 없음)."""

    def __init__(self, conn, table, pool_k=500, top_common_pct=0.03, k1=1.5, b=0.75):
        from kiwipiepy import Kiwi
        self.conn = conn
        self.table = table
        self.pool_k = pool_k
        self.k1 = k1
        self.b = b
        self.kiwi = Kiwi(num_workers=-1)
        self.common_words, self.idf, self.avgdl = self._load_corpus_stats(top_common_pct)

    def _load_corpus_stats(self, top_common_pct):
        with self.conn.cursor() as cur:
            cur.execute(f"SELECT count(*) FROM {self.table}")
            n_docs = cur.fetchone()[0]
            # ts_stat: 코퍼스 전체를 스캔해 단어별 ndoc(등장 문서 수)/nentry(총 등장 횟수)를
            # 한 번에 계산 — 여기서 나온 값으로 IDF와 평균 문서 길이(avgdl)를 미리 구해둠.
            cur.execute(f"SELECT word, ndoc, nentry FROM ts_stat('SELECT content_tsv FROM {self.table}')")
            rows = cur.fetchall()

        common_words = {w for w, ndoc, _ in rows if ndoc > n_docs * top_common_pct}
        # Okapi BM25 IDF: 흔한 단어(ndoc 큼)일수록 0에 가까워지고, 드문 단어일수록 커짐.
        idf = {w: math.log((n_docs - ndoc + 0.5) / (ndoc + 0.5) + 1) for w, ndoc, _ in rows}
        avgdl = sum(nentry for _, _, nentry in rows) / n_docs if n_docs else 0.0
        return common_words, idf, avgdl

    def _tokenize(self, texts):
        results = []
        for tokens in self.kiwi.tokenize(texts):
            kept = [t.form for t in tokens if t.tag.startswith(KIWI_KEEP_PREFIXES)]
            results.append(" ".join(kept))
        return results

    def _bm25_score(self, query_terms, doc_text):
        doc_terms = doc_text.split() if doc_text else []
        doc_len = len(doc_terms)
        if doc_len == 0 or self.avgdl == 0:
            return 0.0
        tf_counter = Counter(doc_terms)
        score = 0.0
        for t in query_terms:
            tf = tf_counter.get(t, 0)
            if tf == 0:  # 이 chunk에 아예 안 나오는 단어는 기여 0(IDF 몰라도 상관없음)
                continue
            idf = self.idf.get(t, 0.0)
            denom = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
            score += idf * tf * (self.k1 + 1) / denom
        return score

    def retrieve_batch(self, queries, max_k):
        queries_kiwi = self._tokenize(queries)

        results = []
        with self.conn.cursor() as cur:
            for q in queries_kiwi:
                terms = q.split()
                if not terms:  # 형태소 분석 결과 실질형태소가 하나도 안 남은 경우(드묾)
                    results.append([])
                    continue
                # 너무 흔한 단어(__init__ 주석 참고)는 1단계 후보 검색에서만 제외 — 다
                # 걸러졌으면(드묾, 짧은 질문 등) 어쩔 수 없이 원래 단어 그대로 사용
                filtered = [t for t in terms if t not in self.common_words]
                query_terms = filtered if filtered else terms
                # 1단계(SQL/GIN): plainto_tsquery는 전부 AND라 40~50개 키워드가 있는 val
                # 질문에는 안 맞음(2026-08-24 실측, 매칭 0건) — OR(|)로 후보 pool_k개를 빠르게
                # 확보만 하고, 최종 순위는 아래 2단계 진짜 BM25로 다시 매김.
                or_query = " | ".join(query_terms)
                cur.execute(
                    f"WITH matched AS ("
                    f"    SELECT case_no, chunk_text_kiwi, ts_rank_cd(content_tsv, query) AS rank "
                    f"    FROM {self.table}, to_tsquery('simple', %s) query "
                    f"    WHERE content_tsv @@ query "
                    f"    ORDER BY rank DESC LIMIT {self.pool_k}"
                    f") "
                    f"SELECT case_no, chunk_text_kiwi FROM matched",
                    (or_query,),
                )
                candidates = cur.fetchall()

                # 2단계(Python): 후보들만 대상으로 진짜 BM25 재점수. 점수 내림차순으로 정렬한
                # 뒤 case_no 중복 제거하면 각 문서의 "가장 점수 높은 chunk"가 자동으로 남음.
                scored = [(case_no, self._bm25_score(query_terms, text)) for case_no, text in candidates]
                scored.sort(key=lambda r: r[1], reverse=True)

                seen = set()
                ranked = []
                for case_no, _ in scored:
                    if case_no not in seen:
                        seen.add(case_no)
                        ranked.append(case_no)
                        if len(ranked) >= max_k:
                            break
                results.append(ranked)
        return results


class HybridRetriever:
    """dense(예: PgvectorRetriever)와 sparse(SparseRetriever)를 가중 RRF(Reciprocal Rank
    Fusion)로 결합. 원래 점수의 스케일이 서로 다른(코사인 거리 vs ts_rank) 두 결과를,
    "몇 위인지"라는 공통 기준으로 바꿔서 합침 — bge_hybrid.py(2026-08-18)와 같은 방식에
    가중치(dense_weight/sparse_weight)를 추가.

    동일 가중치(1.0/1.0)로 처음 돌려봤더니 dense 단독보다 오히려 크게 나빠짐(2026-08-24,
    recall@1 0.6826→0.4727) — sparse 단독 성능이 dense보다 훨씬 약한데(recall@1 0.203)
    RRF가 "sparse가 1위로 꼽았다"는 사실만으로 오답에도 점수를 얹어줘서, dense가 정확히
    맞춘 결과를 오히려 밀어내는 문제였음. sparse가 dense보다 확연히 약할 때는 sparse
    기여도를 낮게 잡아야 함 — 두 retriever 실력이 비슷할 때 잘 맞는 RRF의 알려진 한계."""

    def __init__(self, dense_retriever, sparse_retriever, k=60, pool_k=100,
                 dense_weight=1.0, sparse_weight=1.0):
        self.dense = dense_retriever
        self.sparse = sparse_retriever
        self.k = k
        self.pool_k = pool_k
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight

    def retrieve_batch(self, queries, max_k):
        dense_results = self.dense.retrieve_batch(queries, self.pool_k)
        sparse_results = self.sparse.retrieve_batch(queries, self.pool_k)

        results = []
        for dense_ranked, sparse_ranked in zip(dense_results, sparse_results):
            scores = {}
            for rank, cid in enumerate(dense_ranked, start=1):
                scores[cid] = scores.get(cid, 0.0) + self.dense_weight / (self.k + rank)
            for rank, cid in enumerate(sparse_ranked, start=1):
                scores[cid] = scores.get(cid, 0.0) + self.sparse_weight / (self.k + rank)
            ranked = sorted(scores, key=scores.get, reverse=True)[:max_k]
            results.append(ranked)
        return results


class CrossEncoderReranker:
    """1단계(dense, pgvector/HNSW)로 top_n개 후보를 빠르게 추리고, 2단계(cross-encoder)로
    (query, chunk_text) 쌍을 그 자리에서 같이 인코딩해 재점수·재정렬하는 reranker.

    ColBERT와 달리 코퍼스를 미리 인코딩해서 저장해두는 게 없음(그래서 디스크 추가 소요 0,
    ColBERT는 chunk당 토큰 수만큼 벡터 저장이 필요해 이 인스턴스 디스크 예산(32GB 고정,
    2026-08-25 기준 여유 3.9GB)을 훌쩍 넘어서 배제됨) — 대신 cross-encoder는 후보 하나하나를
    쿼리와 함께 매번 forward pass 해야 해서 후보 수에 비례해 느려짐. `top_n` 기본값 30은
    ef_search=200 기준 recall@30=0.9725(log.md 2026-08-25)면 대부분의 정답이 이미 이 pool
    안에 있다는 실측 근거로 잡은 값 — 이보다 넓혀도 얻는 recall은 적고 latency만 커짐.

    1단계 SQL은 PgvectorRetriever와 같은 패턴(넉넉한 pool을 거리순으로 먼저 뽑고 그 안에서
    case_no dedupe)이지만, 재랭킹에 필요한 chunk_text까지 같이 가져와야 해서 별도로 구현함."""

    def __init__(self, dense_model, cross_encoder_path, conn, table, top_n=30, ef_search=200,
                 device="cuda", query_batch_size=32, rerank_batch_size=32):
        from sentence_transformers import CrossEncoder
        self.dense_model = dense_model
        self.cross_encoder = CrossEncoder(cross_encoder_path, device=device, max_length=512)
        self.conn = conn
        self.table = table
        self.top_n = top_n
        self.query_batch_size = query_batch_size
        self.rerank_batch_size = rerank_batch_size
        with self.conn.cursor() as cur:
            cur.execute("SET hnsw.ef_search = %s", (ef_search,))

    @staticmethod
    def _pg_vector_literal(vec):
        return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"

    def retrieve_batch(self, queries, max_k):
        query_embs = l2norm(self.dense_model.encode(
            queries, batch_size=self.query_batch_size, show_progress_bar=False,
            convert_to_numpy=True, normalize_embeddings=True,
        ).astype(np.float32))

        # case_no dedupe 후에도 top_n개가 남으려면 넉넉한 pool에서 시작해야 함(PgvectorRetriever
        # 주석 참고) — top_n의 10배 정도면 충분히 여유 있음
        pool_k = self.top_n * 10

        results = []
        with self.conn.cursor() as cur:
            # harness.py의 tqdm은 배치(64개) 단위라 rerank처럼 쿼리 하나당 cross-encoder를
            # top_n번씩 돌려야 해서 느린 경우엔 갱신 간격이 너무 뜸함 — 쿼리 단위로 한 번 더
            # 감쌈(leave=False라 배치 진행률 바로 아래 한 줄만 쓰고 사라짐, 여러 줄 안 쌓임)
            for query, vec in tqdm(list(zip(queries, query_embs)), desc="rerank", leave=False, ncols=80):
                lit = self._pg_vector_literal(vec)
                cur.execute(
                    f"WITH nearest AS ("
                    f"    SELECT case_no, chunk_text, embedding <=> %s AS dist FROM {self.table} "
                    f"    ORDER BY embedding <=> %s LIMIT {pool_k}"
                    f") "
                    f"SELECT DISTINCT ON (case_no) case_no, chunk_text, dist FROM nearest "
                    f"ORDER BY case_no, dist",
                    (lit, lit),
                )
                rows = sorted(cur.fetchall(), key=lambda r: r[2])[:self.top_n]
                if not rows:
                    results.append([])
                    continue

                pairs = [(query, r[1]) for r in rows]
                scores = self.cross_encoder.predict(
                    pairs, batch_size=self.rerank_batch_size, show_progress_bar=False)
                order = np.argsort(-np.asarray(scores))
                ranked = [rows[i][0] for i in order][:max_k]
                results.append(ranked)
        return results
