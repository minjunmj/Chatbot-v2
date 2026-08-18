"""
KURE-v1 chunk 기반 retrieval 실험 (chunk 크기 isolate 버전)

목적: jhgan.py 결과(chunk_size=200, jhgan/ko-sbert-sts, recall@1 0.4082)가 KURE-v1 no-chunk
baseline(recall@1 0.4012, model_test.py)을 앞섰지만, 이건 "모델"과 "chunking 여부"를 동시에
바꾼 비교라 chunking 자체의 효과인지 jhgan이라는 모델의 우연한 궁합인지 구분이 안 됨(confound,
2026-08-18 log.md 참고). 이 스크립트는 모델을 KURE-v1(no-chunk 1위 모델)로 고정하고 chunking
여부/크기만 바꿔서 isolate한다.

chunk 크기 3종 비교 (전부 글자 수 기준, 겹침 없음):
- 200자: jhgan.py와 정확히 같은 크기로 맞춰서 "같은 chunking, 다른 모델" 비교가 가능하게 함
- 500자: 문단 하나 정도에 가까운 중간 크기
- 1000자: 여러 문단/한 섹션 정도 — 세 크기 중 가장 chunking을 약하게 적용한 극단

문서 단위 dedupe는 jhgan.py와 동일하게 score-level max-pooling(쿼리-chunk 유사도 랭킹에서
같은 case_id는 가장 먼저 등장한(=가장 점수 높은) chunk만 유지). KURE-v1은 max_seq_length=8192라
위 세 크기 모두(최대 1000자 ≈ 600토큰 내외) truncation 걱정이 없음 — jhgan/klue-bert-base
(128토큰 한도) 때와 달리 별도 검증이 필요 없음.

지표: Recall@1/5/10/20, MRR@10 (val_query.json, 쿼리당 정답 case_id 1개) — chunk 크기별로 각각.
"""

import os
import json
import glob
import time
import gc

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
MODEL_KEY_BASE = "kure-v1"
CHUNK_SIZES = [200, 500, 1000]
MAX_SEQ_LENGTH = 1024  # 실제 chunk 최대 길이(1000자 ≈ 600토큰 내외)보다 넉넉히 잡은 값
CORPUS_BATCH_SIZE = 32  # chunk가 짧아서(no-chunk 때의 batch=16보다) 크게 잡아도 안전함
QUERY_BATCH_SIZE = 32
CHECKPOINT_EVERY = 20000  # chunk 개수 기준 중간 저장 주기

K_LIST = [1, 5, 10, 20]
MRR_K = 10
DEDUPE_POOL = 500  # chunk 랭킹에서 문서 단위로 dedupe할 때 볼 후보 chunk 개수
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


def chunk_document(text, chunk_size):
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    return [c for c in chunks if c.strip()]


def build_chunks(case_ids, doc_texts, chunk_size):
    chunk_texts, chunk_case_ids = [], []
    for cid, text in zip(case_ids, doc_texts):
        for c in chunk_document(text, chunk_size):
            chunk_texts.append(c)
            chunk_case_ids.append(cid)
    return chunk_texts, chunk_case_ids


def encode_corpus(model, texts, cache_path, batch_size, checkpoint_every=CHECKPOINT_EVERY):
    """model_test.py/jhgan.py와 동일한 전략: 길이순 정렬 배치 + 매 배치 GPU 메모리 회수 +
    중간 checkpoint."""
    if os.path.exists(cache_path):
        print(f"  코퍼스 임베딩 캐시 로드: {cache_path}")
        return np.load(cache_path)

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


def evaluate_chunk_size(model, case_ids, doc_texts, val_data, chunk_size):
    model_key = f"{MODEL_KEY_BASE}_chunk{chunk_size}"
    print(f"\n===== {model_key} ({REPO}, chunk_size={chunk_size}) =====")

    chunk_texts, chunk_case_ids = build_chunks(case_ids, doc_texts, chunk_size)
    print(f"chunk 개수: {len(chunk_texts)} (문서 {len(doc_texts)}건)")
    chunk_case_ids = np.array(chunk_case_ids)

    cache_path = os.path.join(CACHE_DIR, f"{model_key}_corpus.npy")
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
        "model": model_key,
        "chunk_size": chunk_size,
        "num_chunks": len(chunk_texts),
        "embedding_dim": int(doc_embs.shape[1]),
        "corpus_encode_time_sec": round(encode_time, 1),
        **{f"recall@{k}": round(recall_hits[k] / n, 4) for k in K_LIST},
        f"mrr@{MRR_K}": round(mrr_sum / n, 4),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
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
    for chunk_size in CHUNK_SIZES:
        results.append(evaluate_chunk_size(model, case_ids, doc_texts, val_data, chunk_size))

    print("\n\n===== 최종 비교 결과 (chunk 크기별) =====")
    cols = ["model"] + [f"recall@{k}" for k in K_LIST] + [f"mrr@{MRR_K}", "num_chunks", "corpus_encode_time_sec"]
    print(" | ".join(cols))
    for r in results:
        print(" | ".join(str(r[c]) for c in cols))

    out_path = os.path.join(CACHE_DIR, f"{MODEL_KEY_BASE}_chunk_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n결과 저장: {out_path}")


if __name__ == "__main__":
    main()
