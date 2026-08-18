"""
임베딩 모델 비교 실험 (chunk 버전): jhgan/ko-sbert-sts

model_test.py와의 차이점 (model_test.py는 chunking 없이 문서 전문을 통으로 encode함):
- 문서를 CHUNK_SIZE(글자 수) 단위로 겹치지 않게 잘라서 각 chunk를 개별 encode
- 검색 시: 쿼리 임베딩과 "모든 chunk" 임베딩의 유사도를 계산한 뒤, 같은 문서(case_id)에서
  나온 chunk 중 가장 높은 점수만 그 문서의 점수로 사용(max-pooling)해서 문서 단위로
  중복 제거한 순위를 만듦. val_query.json의 정답은 chunk가 아니라 case_id 단위이므로
  이렇게 문서 단위로 환원해야 model_test.py와 동일한 기준(Recall@k/MRR@10)으로 비교 가능함.
- chunk 크기(글자 수)/방식은 아직 프로젝트에서 정식으로 정하지 않은 상태라, 이 실험은
  "chunking을 하면 no-chunk 대비 얼마나 달라지는가"를 보기 위한 1회성 비교용 스크립트임.

지표: Recall@1/5/10/20, MRR@10 (val_query.json, 쿼리당 정답 case_id 1개)
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

MODEL_KEY = "jhgan-ko-sbert-sts_chunk200"
REPO = "jhgan/ko-sbert-sts"
MAX_SEQ_LENGTH = 128  # 이 모델의 사전학습 기본값(klue/bert-base 계열)
# 글자 수 기준, 겹침 없음. 300자로 테스트해보니 이 모델 토크나이저 기준 약 143토큰이 나와
# MAX_SEQ_LENGTH(128)를 넘어서 청크 내부가 다시 잘리는 문제가 있었음 -> 128토큰 안에 들어오는
# 200자로 낮춰서 truncation 없이 청크 전체가 임베딩에 반영되도록 함
CHUNK_SIZE = 200
CORPUS_BATCH_SIZE = 64  # encoder-only 소형 모델이라 model_test.py보다 크게 잡음
QUERY_BATCH_SIZE = 32
CHECKPOINT_EVERY = 20000  # chunk 개수 기준 (문서 수보다 훨씬 많아짐)

K_LIST = [1, 5, 10, 20]
MRR_K = 10
# 문서 단위로 dedupe하기 전에, chunk 랭킹에서 몇 개까지 볼지. 같은 문서의 chunk가
# 상위권을 여러 개 차지해도 max_k(=20)개의 "서로 다른" 문서를 확보할 수 있도록 여유 있게 잡음.
DEDUPE_POOL = 500


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


def chunk_document(text, chunk_size=CHUNK_SIZE):
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    return [c for c in chunks if c.strip()]


def build_chunks(case_ids, doc_texts):
    """문서 리스트를 chunk 리스트로 펼침. chunk_case_ids[i]는 chunk_texts[i]의 원문서 case_id."""
    chunk_texts, chunk_case_ids = [], []
    for cid, text in tqdm(zip(case_ids, doc_texts), total=len(doc_texts), desc="chunking", ncols=80):
        for c in chunk_document(text):
            chunk_texts.append(c)
            chunk_case_ids.append(cid)
    return chunk_texts, chunk_case_ids


def encode_corpus(model, texts, cache_path, batch_size, checkpoint_every=CHECKPOINT_EVERY):
    """
    model_test.py의 encode_corpus와 동일한 전략: 길이순 정렬 후 배치(패딩 낭비 감소),
    매 배치 후 GPU 메모리 강제 회수, checkpoint_every개마다 중간 저장(중단 시 재개 가능).
    """
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


def evaluate(case_ids, doc_texts, val_data):
    print(f"\n===== {MODEL_KEY} ({REPO}, chunk_size={CHUNK_SIZE}) =====")

    chunk_texts, chunk_case_ids = build_chunks(case_ids, doc_texts)
    print(f"chunk 개수: {len(chunk_texts)} (문서 {len(doc_texts)}건)")
    chunk_case_ids = np.array(chunk_case_ids)

    model = SentenceTransformer(REPO, device=DEVICE)
    model.max_seq_length = MAX_SEQ_LENGTH

    cache_path = os.path.join(CACHE_DIR, f"{MODEL_KEY}_corpus.npy")
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
    eval_batch = 64  # chunk 임베딩 개수가 많아 model_test.py보다 배치를 작게 잡음

    for start in tqdm(range(0, n, eval_batch), desc="평가", ncols=80):
        end = min(start + eval_batch, n)
        sims = query_t[start:end] @ doc_t.T  # (b, num_chunks)
        top_scores, top_idx = torch.topk(sims, k=pool_k, dim=1)
        top_idx = top_idx.cpu().numpy()

        for row, item in zip(top_idx, val_data[start:end]):
            true_ids = set(item["case_ids"])

            # chunk 랭킹 -> 문서 단위 dedupe (같은 case_id는 처음 등장한 순위만 유지, max-pooling)
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

    del model, doc_t, query_t
    torch.cuda.empty_cache()

    result = {
        "model": MODEL_KEY,
        "chunk_size": CHUNK_SIZE,
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

    result = evaluate(case_ids, doc_texts, val_data)

    out_path = os.path.join(CACHE_DIR, f"{MODEL_KEY}_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump([result], f, ensure_ascii=False, indent=2)
    print(f"\n결과 저장: {out_path}")


if __name__ == "__main__":
    main()
