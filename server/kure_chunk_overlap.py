"""
KURE-v1로 chunk_size + overlap 조합을 추가 실험한다: (200,overlap50) / (300,overlap0) / (300,overlap100)

kure_chunk.py와 동일한 방식(exact search, numpy)으로 recall@k/mrr@10을 측정하되, chunk 크기별
독립 실험이 아니라 "chunk_size 확정 이후 overlap이 추가로 도움이 되는가"를 보기 위한 후속 실험.

3개 설정 다 합쳐도 npy 용량이 DB 빌드(HNSW 인덱스 포함)보다 훨씬 작아서(~5GB 안팎), 굳이
설정마다 지울 필요 없이 **끝까지 다 남겨둔다** — 이후 이긴 후보를 pgvector에 실제로 올릴 때
재인코딩 없이 이 npy를 그대로 재사용하기 위함. 비교 끝나고 진 후보의 npy를 지울지는 결과 보고
사용자가 직접 판단(각 config의 npy 경로는 cache_embeddings/kure-v1_chunk{size}_overlap{overlap}_corpus.npy).

중단됐다가 재실행하면 이미 결과가 저장된 설정은 건너뛰고 이어서 진행한다.

사용법:
    python kure_chunk_overlap.py
"""
import gc
import glob
import json
import os
import time

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(BASE_DIR, "..", "data", "DB_data")
VAL_PATH = os.path.join(BASE_DIR, "..", "data", "Val", "val_query.json")
CACHE_DIR = os.path.join(BASE_DIR, "cache_embeddings")
os.makedirs(CACHE_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
REPO = "nlpai-lab/KURE-v1"
MAX_SEQ_LENGTH = 1024
CORPUS_BATCH_SIZE = 32
QUERY_BATCH_SIZE = 32
CHECKPOINT_EVERY = 20000

K_LIST = [1, 5, 10, 20]
MRR_K = 10
DEDUPE_POOL = 500  # kure_chunk.py와 동일 (문서 단위 dedupe용 후보 pool)
EVAL_BATCH = 64

# (chunk_size, overlap) 조합 — 필요하면 이 리스트만 수정해서 재사용 가능
CONFIGS = [(200, 50), (300, 0), (300, 100)]

RESULTS_PATH = os.path.join(CACHE_DIR, "kure_chunk_overlap_results.json")


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


def l2norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    np.maximum(n, 1e-12, out=n)
    return x / n


def chunk_document(text, chunk_size, overlap):
    """overlap만큼 겹치게 자름. chunk 길이는 항상 chunk_size로 고정, step만 (chunk_size-overlap)로 줄어듦."""
    step = chunk_size - overlap
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), step)]
    return [c for c in chunks if c.strip()]


def build_chunks(case_ids, doc_texts, chunk_size, overlap):
    chunk_texts, chunk_case_ids = [], []
    for cid, text in zip(case_ids, doc_texts):
        for c in chunk_document(text, chunk_size, overlap):
            chunk_texts.append(c)
            chunk_case_ids.append(cid)
    return chunk_texts, chunk_case_ids


def encode_corpus(model, texts, cache_path, batch_size, checkpoint_every=CHECKPOINT_EVERY):
    """kure_chunk.py와 동일한 전략: 길이순 정렬 배치 + 매 배치 GPU 메모리 회수 + 중간 checkpoint."""
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]), reverse=True)
    sorted_texts = [texts[i] for i in order]

    ckpt_path = cache_path + ".partial.npz"
    start_idx = 0
    all_embs = []
    if os.path.exists(ckpt_path):
        ckpt = np.load(ckpt_path)
        all_embs = [ckpt["embs"]]
        start_idx = int(ckpt["done"])
        print(f"  중단된 지점에서 재개: {start_idx}/{len(texts)}건 완료된 상태")

    total_batches = (len(sorted_texts) - start_idx + batch_size - 1) // batch_size
    since_ckpt = 0

    for start in tqdm(range(start_idx, len(sorted_texts), batch_size), desc="corpus encode",
                       ncols=80, total=total_batches):
        batch = sorted_texts[start:start + batch_size]
        e = model.encode(
            batch, batch_size=batch_size, show_progress_bar=False,
            convert_to_numpy=True, normalize_embeddings=True,
        )
        all_embs.append(e)
        since_ckpt += len(batch)

        del e, batch
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

        if since_ckpt >= checkpoint_every:
            merged = np.concatenate(all_embs, axis=0)
            np.savez(ckpt_path, embs=merged, done=start + batch_size)
            all_embs = [merged]
            since_ckpt = 0

    sorted_embs = np.concatenate(all_embs, axis=0)
    embs = np.empty_like(sorted_embs)
    embs[order] = sorted_embs

    np.save(cache_path, embs)
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
    return embs


def evaluate_config(model, case_ids, doc_texts, val_data, chunk_size, overlap):
    config_key = f"kure-v1_chunk{chunk_size}_overlap{overlap}"
    print(f"\n===== {config_key} (chunk_size={chunk_size}, overlap={overlap}) =====")

    chunk_texts, chunk_case_ids = build_chunks(case_ids, doc_texts, chunk_size, overlap)
    print(f"chunk 개수: {len(chunk_texts)} (문서 {len(doc_texts)}건)")
    chunk_case_ids = np.array(chunk_case_ids)

    # 끝나도 안 지움 — 이긴 후보를 나중에 build_vector_db.py류 스크립트로 pgvector에 올릴 때 재사용
    cache_path = os.path.join(CACHE_DIR, f"{config_key}_corpus.npy")
    t0 = time.time()
    doc_embs = l2norm(
        encode_corpus(model, chunk_texts, cache_path, CORPUS_BATCH_SIZE).astype(np.float32)
    )
    encode_time = time.time() - t0

    queries = [item["query"] for item in val_data]
    query_embs = l2norm(model.encode(
        queries, batch_size=QUERY_BATCH_SIZE, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=True,
    ).astype(np.float32))

    doc_t = torch.from_numpy(doc_embs).to(DEVICE)
    query_t = torch.from_numpy(query_embs).to(DEVICE)
    max_k = max(K_LIST + [MRR_K])
    pool_k = min(DEDUPE_POOL, doc_t.shape[0])

    recall_hits = {k: 0 for k in K_LIST}
    mrr_sum = 0.0
    n = len(val_data)

    for start in tqdm(range(0, n, EVAL_BATCH), desc="평가", ncols=80):
        end = min(start + EVAL_BATCH, n)
        sims = query_t[start:end] @ doc_t.T
        _, top_idx = torch.topk(sims, k=pool_k, dim=1)
        top_idx = top_idx.cpu().numpy()

        for row, item in zip(top_idx, val_data[start:end]):
            true_ids = set(item["case_ids"])
            seen = set()
            ranked_case_ids = []
            for j in row:
                cid = chunk_case_ids[j]
                if cid not in seen:
                    seen.add(cid)
                    ranked_case_ids.append(cid)
                    if len(ranked_case_ids) >= max_k:
                        break

            for k in K_LIST:
                if any(cid in true_ids for cid in ranked_case_ids[:k]):
                    recall_hits[k] += 1
            for rank, cid in enumerate(ranked_case_ids[:MRR_K], start=1):
                if cid in true_ids:
                    mrr_sum += 1.0 / rank
                    break

    del doc_t, query_t
    gc.collect()
    torch.cuda.empty_cache()

    result = {
        "config": config_key, "chunk_size": chunk_size, "overlap": overlap,
        "num_chunks": len(chunk_texts), "embedding_dim": int(doc_embs.shape[1]),
        "corpus_encode_time_sec": round(encode_time, 1),
        **{f"recall@{k}": round(recall_hits[k] / n, 4) for k in K_LIST},
        f"mrr@{MRR_K}": round(mrr_sum / n, 4),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"임베딩 캐시 보존됨: {cache_path} (이긴 후보만 남기고 나머지는 비교 후 수동 삭제 권장)")

    return result


def main():
    print("코퍼스 로딩 중...")
    case_ids, doc_texts = load_corpus()
    print(f"코퍼스 문서 수: {len(doc_texts)}")

    val_data = load_val()
    print(f"평가 쿼리 수: {len(val_data)}")

    model = SentenceTransformer(REPO, device=DEVICE,
                                 model_kwargs={"torch_dtype": torch.float16} if DEVICE == "cuda" else None)
    model.max_seq_length = MAX_SEQ_LENGTH

    results = []
    done_configs = set()
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, encoding="utf-8") as f:
            results = json.load(f)
        done_configs = {r["config"] for r in results}

    for chunk_size, overlap in CONFIGS:
        config_key = f"kure-v1_chunk{chunk_size}_overlap{overlap}"
        if config_key in done_configs:
            print(f"{config_key} 이미 완료됨 — 건너뜀 (재실험하려면 {RESULTS_PATH}에서 해당 항목 지우고 재실행)")
            continue
        result = evaluate_config(model, case_ids, doc_texts, val_data, chunk_size, overlap)
        results.append(result)
        with open(RESULTS_PATH, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"결과 저장: {RESULTS_PATH}")

    print("\n\n===== 최종 비교 결과 =====")
    cols = ["config"] + [f"recall@{k}" for k in K_LIST] + [f"mrr@{MRR_K}", "num_chunks", "corpus_encode_time_sec"]
    print(" | ".join(cols))
    for r in results:
        print(" | ".join(str(r[c]) for c in cols))

    print("\n남은 npy 캐시 (이긴 후보만 남기고 나머지는 수동 삭제 권장):")
    for f in sorted(glob.glob(os.path.join(CACHE_DIR, "kure-v1_chunk*_overlap*_corpus.npy"))):
        print(f"  {f}  ({os.path.getsize(f) / 1024**3:.2f} GB)")


if __name__ == "__main__":
    main()
