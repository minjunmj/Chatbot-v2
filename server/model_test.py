"""
임베딩 모델 비교 실험: Qwen3-Embedding-0.6B vs BGE-M3 vs KURE-v1(BGE-M3 한국어 튜닝)

조건:
- chunking 없음 (문서를 통으로 encode, 모델 max_seq_length를 넘으면 기본 truncation)
- 판례내용(전문)만 사용
- dense retrieval만 사용 (BM25/rerank 없이 순수 임베딩 성능만 비교)

지표: Recall@1/5/10/20, MRR@10 (val_query.json, 쿼리당 정답 case_id 1개)
"""

import os
import json
import glob
import time
import gc

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# 긴 시퀀스 배치를 여러 번 할당/해제하면서 생기는 CUDA 메모리 파편화 완화 (torch import 전에 설정해야 적용됨)
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
# 쿼리는 항상 한 문장짜리 짧은 텍스트라 배치를 키워도 안전함
QUERY_BATCH_SIZE = 32
K_LIST = [1, 5, 10, 20]
MRR_K = 10

# 법률 판례 검색에 맞춘 커스텀 instruction (Qwen 쪽 권장: task별 instruction 커스터마이징 시 1~5% 개선 보고됨)
QWEN_QUERY_INSTRUCTION = (
    "Instruct: Given a Korean legal question, retrieve court case precedents "
    "relevant to resolving the legal issue.\nQuery:"
)

MODELS = {
    "qwen3-embedding-0.6b": {
        "repo": "Qwen/Qwen3-Embedding-0.6B",
        "query_prompt": QWEN_QUERY_INSTRUCTION,  # 쿼리 쪽에만 instruction 적용 (asymmetric)
        "max_seq_length": 32000,
        # use_cache=False(아래 evaluate_model 참고) 적용 후에도, 32000토큰까지 꽉 채운 배치가
        # batch_size=16으로 연달아 두 번 나오면 empty_cache() 이후에도 ~20GB가 회수되지 않고
        # 남아 두 번째 배치에서 OOM이 실측됨(cuBLAS/cuDNN 워크스페이스 캐시로 추정, PyTorch
        # 캐싱 할당자 레벨이 아님). batch_size=4는 동일 조건에서 5연속 배치 모두 GPU 메모리가
        # 매번 1.2GB로 완전히 회수되는 것을 실측 확인함 — 느리지만 안전한 값으로 고정.
        "corpus_batch_size": 4,
    },
    "bge-m3": {
        "repo": "BAAI/bge-m3",
        "query_prompt": None,  # BGE-M3는 별도 prefix 불필요
        "max_seq_length": 8192,
        "corpus_batch_size": 16,
    },
    "kure-v1": {  # BGE-M3를 한국어로 추가 튜닝한 모델 ("bge ko")
        "repo": "nlpai-lab/KURE-v1",
        "query_prompt": None,
        "max_seq_length": 8192,
        "corpus_batch_size": 16,
    },
}


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


def encode_corpus(model, texts, cache_path, batch_size, checkpoint_every=2000):
    """
    model.encode(texts, ...)를 통째로 한 번에 호출하면, 내부 배치 루프를 도는 동안
    (Qwen3 같은 causal LM 기반 임베딩 모델에서) GPU 메모리가 배치를 거칠수록 서서히 쌓여
    결국 CUDA OOM으로 죽는 현상이 실측됨(use_cache=False로도 완전히 막히지 않음).
    그래서 배치를 직접 순회하며 매 배치 후 gc.collect()+empty_cache()로 강제 회수하고,
    checkpoint_every 문서마다 중간 저장을 남겨서 중간에 죽어도 처음부터 다시 인코딩하지
    않도록 함.

    길이순 정렬 후 배치를 만듦 — 그래야 배치 안의 시퀀스 길이가 비슷해서 padding 낭비가
    없음. 정렬 없이 원래 순서(파일 로딩 순서, 사실상 무작위) 그대로 배치를 자르면, 아주 긴
    문서 하나가 짧은 문서 여럿과 같은 배치에 우연히 섞일 때마다 그 배치 전체가 긴 문서
    길이만큼 padding되어(=배치 내 최댓값에 맞춰짐) 매번 최악의 경우처럼 무거워지고, 이게
    44,700건 전체에 걸쳐 예측 불가능하게 반복되면서 OOM으로 이어지는 게 실측으로 확인됨.
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

        # 매 배치 끝나고 메모리 정리
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
    embs[order] = sorted_embs  # 원래 texts/case_ids 순서로 복원

    np.save(cache_path, embs)
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
    return embs


def evaluate_model(model_key, cfg, case_ids, doc_texts, val_data):
    print(f"\n===== {model_key} ({cfg['repo']}) =====")

    model = SentenceTransformer(
        cfg["repo"],
        device=DEVICE,
        trust_remote_code=True,
        model_kwargs={"torch_dtype": torch.float16} if DEVICE == "cuda" else None,
    )
    model.max_seq_length = cfg["max_seq_length"]

    # Qwen3-Embedding처럼 causal LM을 기반으로 한 임베딩 모델은 기본적으로 use_cache=True라서,
    # 생성(generation)에 쓰이지도 않을 KV 캐시를 매 forward마다 긴 시퀀스 길이만큼 만들어 붙잡고
    # 있다가 GPU 메모리를 계속 잠식함(batch=4에서도 OOM 재현됨). 임베딩 추출에는 불필요하므로 끔.
    # (BGE-M3/KURE-v1처럼 encoder-only 구조는 이 속성이 없거나 무시되므로 무해함.)
    if hasattr(model[0].auto_model.config, "use_cache"):
        model[0].auto_model.config.use_cache = False

    cache_path = os.path.join(CACHE_DIR, f"{model_key}_corpus.npy")
    t0 = time.time()
    doc_embs = l2norm(
        encode_corpus(model, doc_texts, cache_path, cfg["corpus_batch_size"]).astype(np.float32)
    )
    encode_time = time.time() - t0

    queries = [item["query"] for item in val_data]
    encode_kwargs = dict(
        batch_size=QUERY_BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    if cfg["query_prompt"]:
        encode_kwargs["prompt"] = cfg["query_prompt"]
    query_embs = l2norm(model.encode(queries, **encode_kwargs).astype(np.float32))

    # 코사인 유사도: (num_query, num_docs). GPU에서 matmul 수행 후 CPU로 회수.
    doc_t = torch.from_numpy(doc_embs).to(DEVICE)
    query_t = torch.from_numpy(query_embs).to(DEVICE)
    max_k = max(K_LIST + [MRR_K])

    recall_hits = {k: 0 for k in K_LIST}
    mrr_sum = 0.0
    n = len(val_data)
    eval_batch = 256

    for start in tqdm(range(0, n, eval_batch), desc="평가", ncols=80):
        end = min(start + eval_batch, n)
        sims = query_t[start:end] @ doc_t.T  # (b, num_docs)
        top_scores, top_idx = torch.topk(sims, k=max_k, dim=1)
        top_idx = top_idx.cpu().numpy()

        for row, item in zip(top_idx, val_data[start:end]):
            true_ids = set(item["case_ids"])
            ranked_case_ids = [case_ids[j] for j in row]

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
        "model": model_key,
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

    results = []
    for model_key, cfg in MODELS.items():
        results.append(evaluate_model(model_key, cfg, case_ids, doc_texts, val_data))

    print("\n\n===== 최종 비교 결과 =====")
    cols = ["model"] + [f"recall@{k}" for k in K_LIST] + [f"mrr@{MRR_K}", "embedding_dim", "corpus_encode_time_sec"]
    print(" | ".join(cols))
    for r in results:
        print(" | ".join(str(r[c]) for c in cols))

    out_path = os.path.join(CACHE_DIR, "results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n결과 저장: {out_path}")


if __name__ == "__main__":
    main()
