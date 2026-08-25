"""
cross-encoder reranker 파인튜닝용 (anchor, positive, negative) triplet을 만든다.

prepare_data.py + mine_hard_negatives.py 두 단계를 합친 것과 비슷한데, 결정적으로 다른 점 하나:
그 둘은 코퍼스 전체(823,763개)를 로컬 .npy 캐시로 GPU에 올려서 브루트포스로 hard negative를
찾았는데(디스크에 캐시 몇 GB 필요, 지금 이 인스턴스는 그럴 여유가 없음 — docs/log.md
2026-08-25 디스크 점검 참고), 여기서는 **pgvector(HNSW 인덱스)에 이미 다 있는 걸 그대로
씀** — 코퍼스를 다시 로컬로 끌어올 필요가 전혀 없어서 디스크 추가 소요가 0.

또 하나 다른 점: hard negative를 Phase A(in-batch만 학습된) 모델이 아니라 **지금 실제로
서빙 중인 최종 dense 모델(kure-v1-finetuned-hard)**로 채굴한다 — reranker는 실제 서빙에서
이 모델이 pgvector로 뽑아온 top-N 후보만 보게 되므로, "이 모델이 지금 헷갈려하는 것"을
반영해야 학습 신호가 의미 있음(mine_hard_negatives.py의 "Phase A로 채굴하는 이유"와 같은
논리를 최종 모델 기준으로 한 단계 더 적용한 것).

사용법:
    python prepare_reranker_data.py
    python prepare_reranker_data.py --num-negatives 4 --search-pool 100 --skip-top 5
"""
import argparse
import json
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import psycopg2
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # server/finetune/
TRAIN_PATH = os.path.join(BASE_DIR, "..", "..", "data", "Train", "train_query.json")
OUT_PATH = os.path.join(BASE_DIR, "reranker_pairs.jsonl")

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:lexchatbot_dev@127.0.0.1:5432/lexchatbot")
TABLE = "chunks_300_overlap100"
DENSE_MODEL = os.path.join(BASE_DIR, "output", "kure-v1-finetuned-hard")  # 최종 서빙 모델
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
QUERY_BATCH_SIZE = 32


def parse_pgvector(s):
    """psycopg2가 vector 컬럼을 '[0.1,0.2,...]' 형태의 문자열로 돌려주는 걸 numpy 배열로 변환."""
    return np.array([float(x) for x in s.strip("[]").split(",")], dtype=np.float32)


def pg_vector_literal(vec):
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-negatives", type=int, default=4, help="쿼리당 붙일 hard negative 개수")
    parser.add_argument("--search-pool", type=int, default=100,
                         help="정답 문서 걸러내기 전 pgvector로 넓게 뽑을 후보 chunk 수")
    parser.add_argument("--skip-top", type=int, default=5,
                         help="유사도 최상위 몇 개는 건너뛸지 — 너무 상위는 실제로 관련 있는 내용일 "
                              "false negative 위험이 커서 그 아래 순위에서 negative를 고름 "
                              "(mine_hard_negatives.py와 동일 근거)")
    parser.add_argument("--ef-search", type=int, default=200, help="HNSW ef_search")
    parser.add_argument("--limit", type=int, default=None, help="빠른 확인용 쿼리 수 제한")
    args = parser.parse_args()

    if not os.path.exists(TRAIN_PATH):
        raise FileNotFoundError(f"{TRAIN_PATH} 없음")

    with open(TRAIN_PATH, encoding="utf-8") as f:
        train_data = json.load(f)
    if args.limit:
        train_data = train_data[:args.limit]
    print(f"train 쿼리 수: {len(train_data)}")

    print(f"dense 모델 로딩: {DENSE_MODEL}")
    model = SentenceTransformer(DENSE_MODEL, device=DEVICE,
                                 model_kwargs={"torch_dtype": torch.float16} if DEVICE == "cuda" else None)

    queries = [item["query"] for item in train_data]
    query_embs = model.encode(
        queries, batch_size=QUERY_BATCH_SIZE, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=True,
    ).astype(np.float32)

    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    cur.execute("SET hnsw.ef_search = %s", (args.ef_search,))

    pairs = []
    skipped_no_positive = 0
    skipped_no_negative = 0

    for item, q_emb in tqdm(list(zip(train_data, query_embs)), desc="triplet 생성", ncols=80):
        case_no = item["case_ids"][0]  # 정답이 여러 개면 첫 번째만 사용 (prepare_data.py와 동일 단순화)
        lit = pg_vector_literal(q_emb)

        # 1) positive: 정답 문서(case_no) 안의 chunk들 중 쿼리와 가장 유사한 것 (prepare_data.py와 동일)
        cur.execute(f"SELECT chunk_text, embedding FROM {TABLE} WHERE case_no = %s", (case_no,))
        pos_rows = cur.fetchall()
        if not pos_rows:
            skipped_no_positive += 1
            continue
        pos_chunk_texts = [r[0] for r in pos_rows]
        pos_chunk_embs = np.stack([parse_pgvector(r[1]) for r in pos_rows])
        positive_chunk = pos_chunk_texts[int(np.argmax(pos_chunk_embs @ q_emb))]

        # 2) hard negative: 전체 코퍼스에서 pgvector(HNSW)로 넓게 뽑은 뒤, 정답 문서가 아닌 것들만
        #    순서대로 모으고 skip_top개는 건너뛰고 그다음부터 num_negatives개 사용
        cur.execute(
            f"SELECT case_no, chunk_text FROM {TABLE} "
            f"ORDER BY embedding <=> %s LIMIT {args.search_pool}",
            (lit,),
        )
        candidates = cur.fetchall()
        eligible = [text for c_no, text in candidates if c_no != case_no]
        negatives = eligible[args.skip_top: args.skip_top + args.num_negatives]
        if not negatives:
            skipped_no_negative += 1
            continue

        pairs.append({
            "query": item["query"],
            "positive_chunk": positive_chunk,
            "negative_chunks": negatives,
            "case_no": case_no,
        })

    conn.close()
    print(f"생성된 triplet: {len(pairs)}, positive 없어서 건너뜀: {skipped_no_positive}, "
          f"negative 없어서 건너뜀: {skipped_no_negative}")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"저장: {OUT_PATH}")


if __name__ == "__main__":
    main()
