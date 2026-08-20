"""
train_query.json(질문-정답 문서 쌍)으로부터 파인튜닝용 (질문, positive chunk) 쌍을 만든다.

train_query.json은 문서(case_id) 단위 정답만 갖고 있고 chunk 단위 정답은 원본 데이터 자체에
없다. 그래서 정답 문서 안의 chunk들 중 현재 KURE-v1로 유사도가 가장 높은 chunk를 positive
근사치(proxy label)로 쓴다 — 완벽한 정답은 아니지만, "82만 개 전체에서 정답 찾기"(어려운
문제, 지금 recall@1 0.57)가 아니라 "이미 정답인 걸 아는 문서 1개 안의 chunk 10~30개 중에서
고르기"(훨씬 쉬운 하위 문제)라 신뢰도가 상대적으로 높다고 판단함 (docs/log.md 2026-08-20 참고).

chunks_300_overlap100 테이블(이미 임베딩 계산 완료)에서 case_no로 후보 chunk를 조회하므로
재인코딩은 정답 문서 chunk 수만큼만 필요 — corpus 전체 재인코딩 없음.

사용법:
    python prepare_data.py
"""
import json
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import psycopg2
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # server/
TRAIN_PATH = os.path.join(BASE_DIR, "..", "data", "Train", "train_query.json")
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "finetune_pairs.jsonl")

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:lexchatbot_dev@127.0.0.1:5432/lexchatbot")
TABLE = "chunks_300_overlap100"
REPO = "nlpai-lab/KURE-v1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
QUERY_BATCH_SIZE = 32


def parse_pgvector(s):
    """psycopg2가 vector 컬럼을 '[0.1,0.2,...]' 형태의 문자열로 돌려주는 걸 numpy 배열로 변환."""
    return np.array([float(x) for x in s.strip("[]").split(",")], dtype=np.float32)


def main():
    with open(TRAIN_PATH, encoding="utf-8") as f:
        train_data = json.load(f)
    print(f"train 쿼리 수: {len(train_data)}")

    model = SentenceTransformer(REPO, device=DEVICE,
                                 model_kwargs={"torch_dtype": torch.float16} if DEVICE == "cuda" else None)

    queries = [item["query"] for item in train_data]
    query_embs = model.encode(
        queries, batch_size=QUERY_BATCH_SIZE, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=True,
    ).astype(np.float32)

    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()

    pairs = []
    skipped_no_chunks = 0
    for item, q_emb in tqdm(list(zip(train_data, query_embs)), desc="positive chunk 선택", ncols=80):
        case_no = item["case_ids"][0]  # 정답이 여러 개면 첫 번째만 사용 (단순화, 대부분 1개)
        cur.execute(f"SELECT chunk_text, embedding FROM {TABLE} WHERE case_no = %s", (case_no,))
        rows = cur.fetchall()
        if not rows:
            skipped_no_chunks += 1
            continue

        chunk_texts = [r[0] for r in rows]
        chunk_embs = np.stack([parse_pgvector(r[1]) for r in rows])
        sims = chunk_embs @ q_emb
        best_idx = int(np.argmax(sims))

        pairs.append({
            "query": item["query"],
            "positive_chunk": chunk_texts[best_idx],
            "case_no": case_no,
            "sim": float(sims[best_idx]),
        })

    conn.close()
    print(f"생성된 쌍: {len(pairs)}, chunks_300_overlap100에 case_no 없어서 건너뜀: {skipped_no_chunks}")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"저장: {OUT_PATH}")


if __name__ == "__main__":
    main()
