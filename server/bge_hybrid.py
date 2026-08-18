"""
BGE-M3 hybrid retrieval 실험: dense-only vs sparse-only vs hybrid(dense+sparse)

목적: model_test.py에서 이미 측정한 bge-m3 dense-only 결과(2026-08-18, log.md) 대비, BGE-M3가
자체적으로 지원하는 sparse(lexical weight) 벡터를 dense와 결합했을 때 검색 성능이 실제로
개선되는지 확인한다. (Qwen3/KURE-v1은 sparse 출력을 지원하지 않는 별개 모델이라 이 실험은
BGE-M3 한정)

- dense: BGE-M3의 dense 벡터, cosine similarity (기존 model_test.py와 동일 방식)
- sparse: BGE-M3의 lexical weight(단어별 학습된 가중치) 벡터, 내적으로 유사도 계산
  (BM25 아님 — BGE-M3 모델 자체가 학습한 sparse representation)
- hybrid: dense 랭킹과 sparse 랭킹을 RRF(Reciprocal Rank Fusion, k=60)로 融합.
  dense/sparse는 점수 스케일이 서로 달라(dense는 cosine 0~1, sparse는 겹치는 토큰 수에 따라
  범위가 들쭉날쭉) 점수를 그대로 가중합하면 스케일 차이가 결과를 왜곡함. RRF는 점수가 아니라
  "순위"만 사용해서 이 문제를 피하는 표준적인 hybrid fusion 방식이라 이걸 채택함.

dense/sparse는 같은 forward pass에서 함께 나오므로(model.encode에 return_dense=True,
return_sparse=True를 같이 주면 됨), model_test.py의 bge-m3 실행과 인코딩 비용은 비슷함.

주의: sentence-transformers가 아니라 FlagEmbedding 패키지의 BGEM3FlagModel을 사용함
(sentence-transformers 래퍼는 dense만 지원, sparse/colbert는 FlagEmbedding에서만 뽑을 수 있음).
미설치 시: pip install -U FlagEmbedding

지표: Recall@1/5/10/20, MRR@10 (val_query.json, 쿼리당 정답 case_id 1개) — 세 방식 각각.
"""

import os
import json
import glob
import time
import gc
import pickle
from itertools import chain

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from tqdm import tqdm
from scipy.sparse import csr_matrix, save_npz, load_npz
from FlagEmbedding import BGEM3FlagModel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(BASE_DIR, "..", "data", "DB_data")
VAL_PATH = os.path.join(BASE_DIR, "..", "data", "Val", "val_query.json")
CACHE_DIR = os.path.join(BASE_DIR, "cache_embeddings")
os.makedirs(CACHE_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_KEY = "bge-m3-hybrid"
REPO = "BAAI/bge-m3"
MAX_LEN_DOC = 8192  # model_test.py의 bge-m3 max_seq_length와 동일하게 맞춤 (비교 가능하도록)
MAX_LEN_QUERY = 512  # 쿼리는 짧은 한 문장이라 넉넉함
CORPUS_BATCH_SIZE = 16  # model_test.py에서 bge-m3에 실측으로 안전했던 배치 크기 그대로 사용
QUERY_BATCH_SIZE = 32
CHECKPOINT_EVERY = 2000  # 문서 수 기준 중간 저장 주기

K_LIST = [1, 5, 10, 20]
MRR_K = 10
RRF_K = 60  # RRF 표준 상수(원 논문/Elasticsearch 등에서 통용되는 기본값)
TOP_POOL = 100  # dense/sparse 각각에서 몇 개까지 뽑아서 RRF에 넣을지 (max_k=20보다 넉넉히)

EVAL_BATCH = 128


def load_corpus():
    files = sorted(glob.glob(os.path.join(DB_DIR, "*.json")))
    files = [f for f in files if not os.path.basename(f).startswith("._")]

    case_ids, texts = [], []
    for f in tqdm(files, desc="corpus 로딩", ncols=80):
        with open(f, encoding="utf-8") as fh:
            d = json.load(fh)
        content = (d.get("판례내용") or "").strip()
        if not content:
            continue
        case_id = d.get("사건번호") or os.path.splitext(os.path.basename(f))[0]
        case_ids.append(case_id)
        texts.append(content)
    return case_ids, texts


def load_val():
    with open(VAL_PATH, encoding="utf-8") as f:
        return json.load(f)


def dicts_to_sparse(dict_list, vocab_size):
    """lexical_weights([{token_id_str: weight}, ...])를 scipy csr_matrix로 변환."""
    lengths = [len(d) for d in dict_list]
    rows = np.repeat(np.arange(len(dict_list)), lengths)
    cols = np.fromiter(chain.from_iterable(d.keys() for d in dict_list), dtype=np.int64, count=sum(lengths))
    data = np.fromiter(chain.from_iterable(d.values() for d in dict_list), dtype=np.float32, count=sum(lengths))
    return csr_matrix((data, (rows, cols)), shape=(len(dict_list), vocab_size))


def encode_corpus_hybrid(model, texts, vocab_size, dense_path, sparse_path, batch_size,
                          checkpoint_every=CHECKPOINT_EVERY):
    """
    model_test.py의 encode_corpus와 같은 전략(길이순 정렬 배치 + 매 배치 GPU 메모리 회수 +
    중간 checkpoint)을 dense+sparse 동시 인코딩에 맞게 확장한 버전.
    """
    if os.path.exists(dense_path) and os.path.exists(sparse_path):
        print(f"  코퍼스 임베딩 캐시 로드: {dense_path}, {sparse_path}")
        return np.load(dense_path), load_npz(sparse_path)

    order = sorted(range(len(texts)), key=lambda i: len(texts[i]), reverse=True)
    sorted_texts = [texts[i] for i in order]

    ckpt_path = dense_path + ".partial.pkl"
    start_idx = 0
    all_dense = []
    all_sparse_dicts = []
    if os.path.exists(ckpt_path):
        with open(ckpt_path, "rb") as f:
            ckpt = pickle.load(f)
        all_dense = [ckpt["dense"]]
        all_sparse_dicts = ckpt["sparse_dicts"]
        start_idx = ckpt["done"]
        print(f"  중단된 지점에서 재개: {start_idx}/{len(texts)}건 완료된 상태")

    total_batches = (len(sorted_texts) - start_idx + batch_size - 1) // batch_size
    since_ckpt = 0

    for start in tqdm(range(start_idx, len(sorted_texts), batch_size), desc="corpus encode",
                       ncols=80, total=total_batches):
        batch = sorted_texts[start:start + batch_size]
        out = model.encode(
            batch, batch_size=batch_size, max_length=MAX_LEN_DOC,
            return_dense=True, return_sparse=True, return_colbert_vecs=False,
        )
        all_dense.append(out["dense_vecs"].astype(np.float32))
        all_sparse_dicts.extend(out["lexical_weights"])
        since_ckpt += len(batch)

        del out, batch
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

        if since_ckpt >= checkpoint_every:
            merged_dense = np.concatenate(all_dense, axis=0)
            with open(ckpt_path, "wb") as f:
                pickle.dump({"dense": merged_dense, "sparse_dicts": all_sparse_dicts,
                             "done": start + batch_size}, f)
            all_dense = [merged_dense]
            since_ckpt = 0

    sorted_dense = np.concatenate(all_dense, axis=0)
    dense_embs = np.empty_like(sorted_dense)
    dense_embs[order] = sorted_dense

    sparse_mat_sorted = dicts_to_sparse(all_sparse_dicts, vocab_size)
    inv_order = np.empty(len(order), dtype=np.int64)
    inv_order[order] = np.arange(len(order))
    sparse_mat = sparse_mat_sorted[inv_order]

    np.save(dense_path, dense_embs)
    save_npz(sparse_path, sparse_mat)
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
    return dense_embs, sparse_mat


def rrf_fuse(dense_top_idx, sparse_top_idx, rrf_k=RRF_K, top_n=MRR_K):
    scores = {}
    for rank, idx in enumerate(dense_top_idx, start=1):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (rrf_k + rank)
    for rank, idx in enumerate(sparse_top_idx, start=1):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (rrf_k + rank)
    ranked = sorted(scores.items(), key=lambda x: -x[1])[:top_n]
    return [idx for idx, _ in ranked]


def evaluate(case_ids, doc_dense, doc_sparse, val_data):
    model = BGEM3FlagModel(REPO, use_fp16=(DEVICE == "cuda"))
    vocab_size = model.model.tokenizer.vocab_size

    print("\n===== 쿼리 인코딩 =====")
    queries = [item["query"] for item in val_data]
    out_q = model.encode(
        queries, batch_size=QUERY_BATCH_SIZE, max_length=MAX_LEN_QUERY,
        return_dense=True, return_sparse=True, return_colbert_vecs=False,
    )
    query_dense = out_q["dense_vecs"].astype(np.float32)
    query_sparse = dicts_to_sparse(out_q["lexical_weights"], vocab_size)

    doc_dense_t = torch.from_numpy(doc_dense).to(DEVICE)
    query_dense_t = torch.from_numpy(query_dense).to(DEVICE)
    max_k = max(K_LIST + [MRR_K])

    methods = ["dense_only", "sparse_only", "hybrid_rrf"]
    recall_hits = {m: {k: 0 for k in K_LIST} for m in methods}
    mrr_sum = {m: 0.0 for m in methods}
    n = len(val_data)

    for start in tqdm(range(0, n, EVAL_BATCH), desc="평가", ncols=80):
        end = min(start + EVAL_BATCH, n)

        dense_sims = query_dense_t[start:end] @ doc_dense_t.T
        _, dense_top = torch.topk(dense_sims, k=TOP_POOL, dim=1)
        dense_top = dense_top.cpu().numpy()

        sparse_sims = (query_sparse[start:end] @ doc_sparse.T).toarray().astype(np.float32)
        sparse_sims_t = torch.from_numpy(sparse_sims).to(DEVICE)
        _, sparse_top = torch.topk(sparse_sims_t, k=TOP_POOL, dim=1)
        sparse_top = sparse_top.cpu().numpy()

        for d_row, s_row, item in zip(dense_top, sparse_top, val_data[start:end]):
            true_ids = set(item["case_ids"])

            ranked = {
                "dense_only": [case_ids[j] for j in d_row[:max_k]],
                "sparse_only": [case_ids[j] for j in s_row[:max_k]],
                "hybrid_rrf": [case_ids[j] for j in rrf_fuse(d_row, s_row, top_n=max_k)],
            }

            for m in methods:
                rcids = ranked[m]
                for k in K_LIST:
                    if any(cid in true_ids for cid in rcids[:k]):
                        recall_hits[m][k] += 1
                for rank, cid in enumerate(rcids[:MRR_K], start=1):
                    if cid in true_ids:
                        mrr_sum[m] += 1.0 / rank
                        break

    del model, doc_dense_t, query_dense_t
    torch.cuda.empty_cache()

    results = []
    for m in methods:
        results.append({
            "model": f"{MODEL_KEY}:{m}",
            **{f"recall@{k}": round(recall_hits[m][k] / n, 4) for k in K_LIST},
            f"mrr@{MRR_K}": round(mrr_sum[m] / n, 4),
        })
    return results


def main():
    print("코퍼스 로딩 중...")
    case_ids, doc_texts = load_corpus()
    print(f"코퍼스 문서 수: {len(doc_texts)}")

    val_data = load_val()
    print(f"평가 쿼리 수: {len(val_data)}")

    print("\n===== 코퍼스 인코딩 (dense + sparse) =====")
    model = BGEM3FlagModel(REPO, use_fp16=(DEVICE == "cuda"))
    vocab_size = model.model.tokenizer.vocab_size

    dense_path = os.path.join(CACHE_DIR, f"{MODEL_KEY}_dense_corpus.npy")
    sparse_path = os.path.join(CACHE_DIR, f"{MODEL_KEY}_sparse_corpus.npz")
    t0 = time.time()
    doc_dense, doc_sparse = encode_corpus_hybrid(
        model, doc_texts, vocab_size, dense_path, sparse_path, CORPUS_BATCH_SIZE
    )
    encode_time = time.time() - t0
    print(f"코퍼스 인코딩 소요: {encode_time:.1f}초")

    del model
    gc.collect()
    torch.cuda.empty_cache()

    results = evaluate(case_ids, doc_dense, doc_sparse, val_data)

    print("\n\n===== 최종 비교 결과 (dense-only vs sparse-only vs hybrid) =====")
    cols = ["model"] + [f"recall@{k}" for k in K_LIST] + [f"mrr@{MRR_K}"]
    print(" | ".join(cols))
    for r in results:
        print(" | ".join(str(r[c]) for c in cols))

    out_path = os.path.join(CACHE_DIR, f"{MODEL_KEY}_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n결과 저장: {out_path}")


if __name__ == "__main__":
    main()
