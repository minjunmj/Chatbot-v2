"""
Phase A(in-batch negative만으로 파인튜닝한 모델)로 hard negative를 채굴해서
finetune_pairs.jsonl에 hard_negative_chunk 필드를 추가한 학습 데이터를 만든다.

Phase A 모델로 채굴하는 이유: base 모델 기준으로 채굴하면 Phase A가 이미 잘 구분하게 된
부분까지 또 negative로 잡아서 학습 신호가 낭비됨 — "지금 모델이 헷갈려하는 것"을 반영해야
다음 단계 학습에 의미가 있음.

전체 코퍼스(823,763 chunk)에서 각 학습 쿼리와 유사도가 높은 상위 후보(--search-pool)를 뽑고,
그중 정답 문서(case_no)가 아닌 것들 중 상위 --num-negatives개를 hard negative로 붙인다.

eval_model.py가 파인튜닝 모델 평가 시 저장해둔 코퍼스 캐시
(cache_embeddings/eval_..._corpus.npy)를 재사용 — 재인코딩 없음.

사용법:
    python mine_hard_negatives.py
    python mine_hard_negatives.py --num-negatives 1 --search-pool 100
"""
import argparse
import glob
import json
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # server/
DB_DIR = os.path.join(BASE_DIR, "..", "data", "DB_data")
TRAIN_PATH = os.path.join(BASE_DIR, "..", "data", "Train", "train_query.json")
FINETUNE_DIR = os.path.dirname(os.path.abspath(__file__))
PAIRS_PATH = os.path.join(FINETUNE_DIR, "finetune_pairs.jsonl")
OUT_PATH = os.path.join(FINETUNE_DIR, "finetune_pairs_hard.jsonl")

PHASE_A_MODEL = os.path.join(FINETUNE_DIR, "output", "kure-v1-finetuned-inbatch")
PHASE_A_CACHE = os.path.join(
    BASE_DIR, "cache_embeddings",
    "eval_._output_kure-v1-finetuned-inbatch_chunk300_overlap100_corpus.npy",
)

CHUNK_SIZE = 300
OVERLAP = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
QUERY_BATCH_SIZE = 32
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


def l2norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    np.maximum(n, 1e-12, out=n)
    return x / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-negatives", type=int, default=1, help="쿼리당 붙일 hard negative 개수")
    parser.add_argument("--search-pool", type=int, default=100, help="정답 문서 걸러내기 전 넓게 뽑을 후보 수")
    parser.add_argument("--skip-top", type=int, default=5,
                         help="유사도 최상위 몇 개는 건너뛸지 — 너무 상위는 실제로 관련 있는 내용일 "
                              "false negative 위험이 커서 그 아래 순위에서 negative를 고름")
    args = parser.parse_args()

    if not os.path.exists(PHASE_A_CACHE):
        raise FileNotFoundError(
            f"{PHASE_A_CACHE} 없음 — 먼저 `eval_model.py --model-path {PHASE_A_MODEL}`를 "
            f"한 번 실행해서 Phase A 모델의 코퍼스 캐시를 만들어야 함")
    if not os.path.exists(PAIRS_PATH):
        raise FileNotFoundError(f"{PAIRS_PATH} 없음 — prepare_data.py를 먼저 실행해야 함")

    print("코퍼스 로딩 중...")
    case_ids, doc_texts = load_corpus()
    chunk_texts, chunk_case_ids = build_chunks(case_ids, doc_texts)
    chunk_case_ids = np.array(chunk_case_ids)
    print(f"chunk 개수: {len(chunk_texts)}")

    print(f"Phase A 코퍼스 임베딩 캐시 로드: {PHASE_A_CACHE}")
    doc_embs = np.load(PHASE_A_CACHE).astype(np.float32)
    assert doc_embs.shape[0] == len(chunk_texts), "캐시 row 수와 chunk 수가 안 맞음"
    doc_embs = l2norm(doc_embs)
    doc_t = torch.from_numpy(doc_embs).to(DEVICE)

    with open(PAIRS_PATH, encoding="utf-8") as f:
        positive_by_query = {json.loads(line)["query"]: json.loads(line)["positive_chunk"]
                              for line in f}

    with open(TRAIN_PATH, encoding="utf-8") as f:
        train_data = json.load(f)
    train_data = [item for item in train_data if item["query"] in positive_by_query]
    print(f"positive가 있는 train 쿼리 수: {len(train_data)}")

    print(f"Phase A 모델 로딩: {PHASE_A_MODEL}")
    model = SentenceTransformer(PHASE_A_MODEL, device=DEVICE,
                                 model_kwargs={"torch_dtype": torch.float16} if DEVICE == "cuda" else None)

    queries = [item["query"] for item in train_data]
    query_embs = l2norm(model.encode(
        queries, batch_size=QUERY_BATCH_SIZE, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=True,
    ).astype(np.float32))
    query_t = torch.from_numpy(query_embs).to(DEVICE)

    pool_k = min(args.search_pool, doc_t.shape[0])
    results = []
    skipped_no_negative = 0

    for start in tqdm(range(0, len(train_data), EVAL_BATCH), desc="hard negative 채굴", ncols=80):
        end = min(start + EVAL_BATCH, len(train_data))
        sims = query_t[start:end] @ doc_t.T
        _, top_idx = torch.topk(sims, k=pool_k, dim=1)
        top_idx = top_idx.cpu().numpy()

        for row, item in zip(top_idx, train_data[start:end]):
            case_no = item["case_ids"][0]
            # row는 유사도 순으로 이미 정렬돼있음 — 정답 문서(case_no)가 아닌 것들만 순서대로 모으고,
            # 그중 상위 skip_top개는 건너뛴 뒤(false negative 위험 완화) 그다음부터 negative로 사용
            eligible = [chunk_texts[j] for j in row if chunk_case_ids[j] != case_no]
            negs = eligible[args.skip_top: args.skip_top + args.num_negatives]
            if not negs:
                skipped_no_negative += 1
                continue
            entry = {
                "query": item["query"],
                "positive_chunk": positive_by_query[item["query"]],
            }
            if args.num_negatives == 1:
                entry["hard_negative_chunk"] = negs[0]
            else:
                entry["hard_negative_chunks"] = negs
            results.append(entry)

    del doc_t, query_t
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    print(f"생성된 쌍: {len(results)}, negative 못 찾아서 건너뜀: {skipped_no_negative}")
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"저장: {OUT_PATH}")


if __name__ == "__main__":
    main()
