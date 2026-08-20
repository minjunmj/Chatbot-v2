"""
임의 모델(base KURE-v1 또는 파인튜닝 결과 체크포인트)로 chunk_size=300/overlap=100 코퍼스를
인코딩해서 val_query.json 기준 recall@k/mrr@10을 계산한다 — kure_chunk_overlap.py와 동일한
exact-search(numpy) 방법론이라 지금까지의 chunk300_overlap100 실험 결과와 직접 비교 가능.

파인튜닝된 모델은 가중치가 달라져서 임베딩도 달라지므로 코퍼스를 새로 인코딩해야 하지만
(chunk300_overlap100 기준 실측 약 2시간 10분/7833초), **base KURE-v1은 이미
kure_chunk_overlap.py가 만든 kure-v1_chunk300_overlap100_corpus.npy가 정확히 이 코퍼스를
base 모델로 인코딩한 결과라 그대로 재사용**한다 — 재인코딩 없이 몇 초 만에 로드됨.
그 외 임의 모델(파인튜닝 체크포인트 등)은 최초 1회 인코딩 후 캐시해서, 같은 모델로 재평가할
때는 다시 빠르게 돈다.

사용법:
    python eval_model.py --model-path nlpai-lab/KURE-v1                        # base 모델 (캐시 재사용, 빠름)
    python eval_model.py --model-path ./output/kure-v1-finetuned-inbatch       # 파인튜닝 결과 (최초 1회만 느림)
"""
import argparse
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

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # server/
DB_DIR = os.path.join(BASE_DIR, "..", "data", "DB_data")
VAL_PATH = os.path.join(BASE_DIR, "..", "data", "Val", "val_query.json")
CACHE_DIR = os.path.join(BASE_DIR, "cache_embeddings")

BASE_REPO = "nlpai-lab/KURE-v1"
BASE_MODEL_CACHE = os.path.join(CACHE_DIR, "kure-v1_chunk300_overlap100_corpus.npy")  # kure_chunk_overlap.py 산출물

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHUNK_SIZE = 300
OVERLAP = 100
MAX_SEQ_LENGTH = 1024
CORPUS_BATCH_SIZE = 32
QUERY_BATCH_SIZE = 32

K_LIST = [1, 5, 10, 20]
MRR_K = 10
DEDUPE_POOL = 500
EVAL_BATCH = 64


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
    step = chunk_size - overlap
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), step)]
    return [c for c in chunks if c.strip()]


def build_chunks(case_ids, doc_texts):
    chunk_texts, chunk_case_ids = [], []
    for cid, text in zip(case_ids, doc_texts):
        for c in chunk_document(text, CHUNK_SIZE, OVERLAP):
            chunk_texts.append(c)
            chunk_case_ids.append(cid)
    return chunk_texts, chunk_case_ids


def encode_all(model, texts, batch_size):
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]), reverse=True)
    sorted_texts = [texts[i] for i in order]

    all_embs = []
    for start in tqdm(range(0, len(sorted_texts), batch_size), desc="encode", ncols=80):
        batch = sorted_texts[start:start + batch_size]
        e = model.encode(batch, batch_size=batch_size, show_progress_bar=False,
                          convert_to_numpy=True, normalize_embeddings=True)
        all_embs.append(e)
        del e, batch
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    sorted_embs = np.concatenate(all_embs, axis=0)
    embs = np.empty_like(sorted_embs)
    embs[order] = sorted_embs
    return embs


def cache_path_for(model_path):
    if model_path == BASE_REPO:
        return BASE_MODEL_CACHE
    safe_name = model_path.strip("/").replace("/", "_").replace(os.sep, "_")
    return os.path.join(CACHE_DIR, f"eval_{safe_name}_chunk300_overlap100_corpus.npy")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True,
                         help="HF hub 이름(예: nlpai-lab/KURE-v1) 또는 로컬 체크포인트 경로")
    args = parser.parse_args()

    print("코퍼스 로딩 중...")
    case_ids, doc_texts = load_corpus()
    print(f"코퍼스 문서 수: {len(doc_texts)}")

    val_data = load_val()
    print(f"평가 쿼리 수: {len(val_data)}")

    chunk_texts, chunk_case_ids = build_chunks(case_ids, doc_texts)
    print(f"chunk 개수: {len(chunk_texts)}")
    chunk_case_ids = np.array(chunk_case_ids)

    cache_path = cache_path_for(args.model_path)
    if os.path.exists(cache_path):
        print(f"코퍼스 임베딩 캐시 재사용: {cache_path} (재인코딩 없음)")
        t0 = time.time()
        doc_embs = l2norm(np.load(cache_path).astype(np.float32))
        encode_time = time.time() - t0
        assert doc_embs.shape[0] == len(chunk_texts), (
            f"캐시 row 수({doc_embs.shape[0]})와 재생성된 chunk 수({len(chunk_texts)})가 안 맞음 — "
            f"{cache_path}가 다른 chunk 설정으로 만들어졌을 수 있음")
        model = SentenceTransformer(args.model_path, device=DEVICE,
                                     model_kwargs={"torch_dtype": torch.float16} if DEVICE == "cuda" else None)
        model.max_seq_length = MAX_SEQ_LENGTH
    else:
        print(f"모델 로딩: {args.model_path}")
        model = SentenceTransformer(args.model_path, device=DEVICE,
                                     model_kwargs={"torch_dtype": torch.float16} if DEVICE == "cuda" else None)
        model.max_seq_length = MAX_SEQ_LENGTH

        t0 = time.time()
        doc_embs = l2norm(encode_all(model, chunk_texts, CORPUS_BATCH_SIZE).astype(np.float32))
        encode_time = time.time() - t0

        os.makedirs(CACHE_DIR, exist_ok=True)
        np.save(cache_path, doc_embs.astype(np.float16))  # 디스크 절반으로 절약 (kure_chunk_overlap.py 캐시와 동일 정밀도)
        print(f"코퍼스 임베딩 캐시 저장: {cache_path} (다음 평가 시 재사용됨)")

    query_embs = l2norm(model.encode(
        [item["query"] for item in val_data], batch_size=QUERY_BATCH_SIZE, show_progress_bar=True,
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

    result = {
        "model_path": args.model_path, "num_chunks": len(chunk_texts),
        "corpus_encode_time_sec": round(encode_time, 1),
        **{f"recall@{k}": round(recall_hits[k] / n, 4) for k in K_LIST},
        f"mrr@{MRR_K}": round(mrr_sum / n, 4),
    }
    print("\n===== 결과 =====")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
