"""
harness.py + retrievers.py를 묶은 CLI. model_test.py/jhgan.py/kure_chunk.py/
kure_chunk_overlap.py/test_val_pgvector.py/finetune/eval_model.py가 하던 걸 이 스크립트
하나로 대체 — 새 chunk_size/overlap/모델/테이블 조합을 테스트할 때 새 .py 파일을 안 만들어도 됨.

사용법:
    # exact search (pgvector 없이, 코퍼스 직접 인코딩) — chunk_size/overlap 실험, 파인튜닝 모델 비교 등
    python run_eval.py --mode exact --model-path nlpai-lab/KURE-v1 --chunk-size 300 --overlap 100
    python run_eval.py --mode exact --model-path ../finetune/output/kure-v1-finetuned-hard --chunk-size 300 --overlap 100

    # pgvector (실제 서빙 latency까지 포함) — 정식 DB 테이블 대상
    python run_eval.py --mode pgvector --model-path nlpai-lab/KURE-v1 --table chunks_300_overlap100 --measure-latency

    # sparse (Kiwi+tsvector, dense 모델 불필요)
    python run_eval.py --mode sparse --table chunks_300_overlap100

    # hybrid (dense+sparse, RRF 결합) — --model-path 필요(dense용)
    python run_eval.py --mode hybrid --model-path ../finetune/output/kure-v1-finetuned-hard --table chunks_300_overlap100

    # 빠른 확인용 쿼리 수 제한
    python run_eval.py --mode exact --model-path nlpai-lab/KURE-v1 --chunk-size 300 --overlap 100 --limit 500
"""
import argparse
import glob
import json
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from harness import evaluate
from retrievers import DenseExactRetriever, PgvectorRetriever, SparseRetriever, HybridRetriever

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # server/
DB_DIR = os.path.join(BASE_DIR, "..", "data", "DB_data")
VAL_PATH = os.path.join(BASE_DIR, "..", "data", "Val", "val_query.json")
CACHE_DIR = os.path.join(BASE_DIR, "cache_embeddings")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:lexchatbot_dev@127.0.0.1:5432/lexchatbot")


def load_corpus():
    files = sorted(glob.glob(os.path.join(DB_DIR, "*.json")))
    files = [f for f in files if not os.path.basename(f).startswith("._")]
    case_ids, texts = [], []
    for f in files:
        with open(f, encoding="utf-8") as fh:
            d = json.load(fh)
        content = (d.get("판례내용") or "").strip()
        if not content:
            continue
        case_id = d.get("사건번호") or os.path.splitext(os.path.basename(f))[0]
        case_ids.append(case_id)
        texts.append(content)
    return case_ids, texts


def chunk_document(text, chunk_size, overlap):
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


def load_val(limit=None):
    with open(VAL_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return data[:limit] if limit else data


def cache_path_for(model_path, chunk_size, overlap):
    """model_path+chunk 설정별 코퍼스 임베딩 캐시 경로. 기존 kure_chunk.py류 스크립트가 만든
    관례적 파일명(overlap=0이면 kure-v1_chunk{size}_corpus.npy 등)과도 최대한 맞춰서, 이미
    있는 캐시를 재사용할 수 있게 함."""
    safe_name = model_path.strip("/").replace("/", "_").replace(os.sep, "_")
    if overlap:
        return os.path.join(CACHE_DIR, f"{safe_name}_chunk{chunk_size}_overlap{overlap}_corpus.npy")
    return os.path.join(CACHE_DIR, f"{safe_name}_chunk{chunk_size}_corpus.npy")


def encode_corpus_cached(model, chunk_texts, cache_path, batch_size=32):
    if os.path.exists(cache_path):
        print(f"코퍼스 임베딩 캐시 재사용: {cache_path} (재인코딩 없음)")
        embs = np.load(cache_path).astype(np.float32)
        assert embs.shape[0] == len(chunk_texts), (
            f"캐시 row 수({embs.shape[0]})와 chunk 수({len(chunk_texts)})가 안 맞음 — "
            f"다른 chunk 설정으로 만들어진 캐시일 수 있음")
        return embs

    from tqdm import tqdm
    print("코퍼스 인코딩 중 (캐시 없음)...")
    order = sorted(range(len(chunk_texts)), key=lambda i: len(chunk_texts[i]), reverse=True)
    sorted_texts = [chunk_texts[i] for i in order]
    all_embs = []
    for start in tqdm(range(0, len(sorted_texts), batch_size), desc="encode", ncols=80):
        batch = sorted_texts[start:start + batch_size]
        e = model.encode(batch, batch_size=batch_size, show_progress_bar=False,
                          convert_to_numpy=True, normalize_embeddings=True)
        all_embs.append(e)
    sorted_embs = np.concatenate(all_embs, axis=0).astype(np.float32)
    embs = np.empty_like(sorted_embs)
    embs[order] = sorted_embs

    os.makedirs(CACHE_DIR, exist_ok=True)
    np.save(cache_path, embs.astype(np.float16))  # fp16로 저장 — 디스크 절반
    print(f"코퍼스 임베딩 캐시 저장: {cache_path}")
    return embs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["exact", "pgvector", "sparse", "hybrid"])
    parser.add_argument("--model-path", help="exact/pgvector/hybrid 필수 (sparse는 불필요, dense 모델 안 씀)")
    parser.add_argument("--chunk-size", type=int, help="--mode exact 필수")
    parser.add_argument("--overlap", type=int, default=0, help="--mode exact 전용, 기본 0")
    parser.add_argument("--table", help="--mode pgvector/sparse/hybrid 필수 (예: chunks_300_overlap100)")
    parser.add_argument("--limit", type=int, default=None, help="빠른 확인용 쿼리 수 제한")
    parser.add_argument("--k-list", type=str, default="1,5,10,20",
                         help="recall@k를 계산할 k값들, 콤마로 구분 (예: 1,5,10,20,30,40,50)")
    parser.add_argument("--ef-search", type=int, default=100,
                         help="--mode pgvector/hybrid 전용, HNSW ef_search (기본 100, 클수록 정확하나 느려짐)")
    parser.add_argument("--measure-latency", action="store_true")
    parser.add_argument("--dense-weight", type=float, default=1.0, help="--mode hybrid 전용, RRF 가중치")
    parser.add_argument("--sparse-weight", type=float, default=1.0,
                         help="--mode hybrid 전용, RRF 가중치 — sparse가 dense보다 훨씬 약하면 낮게 잡을 것 "
                              "(2026-08-24: 1.0/1.0 동일 가중치는 dense 단독보다 크게 나빴음)")
    args = parser.parse_args()

    val_data = load_val(args.limit)
    print(f"평가 쿼리 수: {len(val_data)}")

    model = None
    if args.mode in ("exact", "pgvector", "hybrid"):
        if not args.model_path:
            raise ValueError(f"--mode {args.mode}는 --model-path 필수")
        print(f"모델 로딩: {args.model_path}")
        model = SentenceTransformer(args.model_path, device=DEVICE,
                                     model_kwargs={"torch_dtype": torch.float16} if DEVICE == "cuda" else None)

    if args.mode == "exact":
        if not args.chunk_size:
            raise ValueError("--mode exact는 --chunk-size 필수")
        model.max_seq_length = 1024
        print("코퍼스 로딩 중...")
        case_ids, doc_texts = load_corpus()
        chunk_texts, chunk_case_ids = build_chunks(case_ids, doc_texts, args.chunk_size, args.overlap)
        print(f"chunk 개수: {len(chunk_texts)}")
        cache_path = cache_path_for(args.model_path, args.chunk_size, args.overlap)
        doc_embs = encode_corpus_cached(model, chunk_texts, cache_path)
        retriever = DenseExactRetriever(model, chunk_case_ids, doc_embs=doc_embs, device=DEVICE)

    elif args.mode == "pgvector":
        if not args.table:
            raise ValueError("--mode pgvector는 --table 필수")
        import psycopg2
        conn = psycopg2.connect(DATABASE_URL)
        retriever = PgvectorRetriever(model, conn, args.table, ef_search=args.ef_search)

    elif args.mode == "sparse":
        if not args.table:
            raise ValueError("--mode sparse는 --table 필수")
        import psycopg2
        conn = psycopg2.connect(DATABASE_URL)
        retriever = SparseRetriever(conn, args.table)

    else:  # hybrid
        if not args.table:
            raise ValueError("--mode hybrid는 --table 필수")
        import psycopg2
        conn = psycopg2.connect(DATABASE_URL)
        dense = PgvectorRetriever(model, conn, args.table, ef_search=args.ef_search)
        sparse = SparseRetriever(conn, args.table)
        retriever = HybridRetriever(dense, sparse, dense_weight=args.dense_weight, sparse_weight=args.sparse_weight)

    k_list = tuple(int(k) for k in args.k_list.split(","))
    result = evaluate(retriever, val_data, k_list=k_list, measure_latency=args.measure_latency)
    print("\n===== 결과 =====")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
