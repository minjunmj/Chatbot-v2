"""
DB_data(원본 JSON) + 이미 계산된 chunk 임베딩 캐시(server/cache_embeddings/)를
pgvector 테이블에 적재한다. chunk_size + overlap 조합을 임의로 받는다.

재임베딩은 하지 않는다 — kure_chunk.py/kure_chunk_overlap.py가 저장한 .npy는 chunk 텍스트/
사건번호 배열을 같이 저장하지 않았기 때문에(벡터만 저장됨), 이 스크립트는 그 스크립트들과
완전히 동일한 로직(load_corpus의 sorted glob 순서, chunk_document의 분할 방식)으로 chunk
텍스트/사건번호를 재생성해서 캐시된 벡터와 순서를 맞춘다. 문서 슬라이싱만 다시 하는 거라
GPU 불필요, 수 초면 끝남.

npy 파일명 규칙:
    overlap=0  → kure-v1_chunk{size}_corpus.npy            (kure_chunk.py 산출물)
    overlap>0  → kure-v1_chunk{size}_overlap{overlap}_corpus.npy  (kure_chunk_overlap.py 산출물)
테이블명 규칙:
    overlap=0  → chunks_{size}
    overlap>0  → chunks_{size}_overlap{overlap}

사용법:
    python build_vector_db.py --chunk-size 200                    # overlap 없음
    python build_vector_db.py --chunk-size 200 --overlap 50
    python build_vector_db.py --chunk-size 300 --overlap 100
"""
import argparse
import glob
import io
import json
import os
import threading
import time

import numpy as np
import psycopg2
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # server/
DB_DATA_DIR = os.path.join(BASE_DIR, "..", "data", "DB_data")
CACHE_DIR = os.path.join(BASE_DIR, "cache_embeddings")
SCHEMA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "schema.sql")

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:lexchatbot_dev@127.0.0.1:5432/lexchatbot")
COPY_BATCH = 50_000


def chunk_document(text, chunk_size, overlap):
    """overlap만큼 겹치게 자름 — kure_chunk_overlap.py와 동일 로직.
    chunk 길이는 항상 chunk_size로 고정, step(=chunk_size-overlap)만 줄어듦."""
    step = chunk_size - overlap
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), step)]
    return [c for c in chunks if c.strip()]


def load_corpus():
    """kure_chunk.py의 load_corpus()와 동일한 순서/로직 + metadata까지 같이 반환."""
    files = sorted(glob.glob(os.path.join(DB_DATA_DIR, "*.json")))
    files = [f for f in files if not os.path.basename(f).startswith("._")]

    cases = []
    for f in tqdm(files, desc="corpus 로딩", ncols=80):
        with open(f, encoding="utf-8") as fh:
            d = json.load(fh)
        content = (d.get("판례내용") or "").strip()
        if not content:
            continue
        case_no = d.get("사건번호") or os.path.splitext(os.path.basename(f))[0]
        cases.append({
            "case_no": case_no,               # 사건번호
            "case_name": d.get("사건명"),        # 사건명
            "court_name": d.get("법원명"),       # 법원명
            "judgment_date": d.get("선고일자"),   # 선고일자
            "case_type": d.get("사건종류명"),     # 사건종류명
            "content": content,               # 판례내용 (chunk_text/embedding으로 변환됨)
        })
    return cases


def pg_vector_literal(vec):
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"


def copy_rows(conn, table, rows):
    """rows: (case_no, case_name, court_name, judgment_date, case_type, chunk_index, chunk_text,
    embedding_literal) 튜플 리스트 = (사건번호, 사건명, 법원명, 선고일자, 사건종류명, chunk 순서, chunk 원문, 임베딩)"""
    buf = io.StringIO()
    for r in rows:
        buf.write("\t".join(
            "\\N" if v is None else str(v).replace("\\", "\\\\").replace("\t", " ").replace("\n", " ")
            for v in r
        ) + "\n")
    buf.seek(0)
    with conn.cursor() as cur:
        cur.copy_expert(
            # 사건번호, 사건명, 법원명, 선고일자, 사건종류명, chunk 순서, chunk 원문, 임베딩
            f"COPY {table} (case_no, case_name, court_name, judgment_date, case_type, "
            f"chunk_index, chunk_text, embedding) FROM STDIN WITH (FORMAT text)",
            buf,
        )
    conn.commit()


def create_hnsw_index_with_progress(table):
    """CREATE INDEX는 단일 SQL문이라 자체 진행률이 없음 — 별도 스레드에서 실행하는 동안
    pg_stat_progress_create_index 뷰를 폴링해서 현재 phase/처리 tuple 수를 보여준다."""
    def worker():
        conn2 = psycopg2.connect(DATABASE_URL)
        with conn2.cursor() as cur:
            cur.execute(f"CREATE INDEX IF NOT EXISTS {table}_embedding_hnsw ON {table} "
                        f"USING hnsw (embedding vector_cosine_ops)")
        conn2.commit()
        conn2.close()

    t = threading.Thread(target=worker)
    t.start()

    poll_conn = psycopg2.connect(DATABASE_URL)
    pbar = tqdm(desc=f"{table} HNSW 인덱스 빌드", unit="s", ncols=80, bar_format="{desc}: {elapsed} 경과 | {postfix}")
    while t.is_alive():
        with poll_conn.cursor() as cur:
            cur.execute("SELECT phase, tuples_done, tuples_total FROM pg_stat_progress_create_index")
            row = cur.fetchone()
        if row:
            phase, tuples_done, tuples_total = row
            pbar.set_postfix_str(f"{phase} ({tuples_done}/{tuples_total} tuples)")
        pbar.update(0)
        time.sleep(2)
    t.join()
    pbar.set_postfix_str("완료")
    pbar.close()
    poll_conn.close()


def build_for_config(chunk_size, overlap, cases):
    table = f"chunks_{chunk_size}" if overlap == 0 else f"chunks_{chunk_size}_overlap{overlap}"
    table_suffix = str(chunk_size) if overlap == 0 else f"{chunk_size}_overlap{overlap}"
    npy_name = (f"kure-v1_chunk{chunk_size}_corpus.npy" if overlap == 0
                else f"kure-v1_chunk{chunk_size}_overlap{overlap}_corpus.npy")
    npy_path = os.path.join(CACHE_DIR, npy_name)
    if not os.path.exists(npy_path):
        raise FileNotFoundError(
            f"{npy_path} 없음 — kure_chunk.py 또는 kure_chunk_overlap.py를 먼저 돌려서 캐시를 만들어야 함")

    print(f"\n===== chunk_size={chunk_size}, overlap={overlap} (table={table}) =====")
    embs = np.load(npy_path)
    print(f"임베딩 로드: {embs.shape}")

    conn = psycopg2.connect(DATABASE_URL)
    with conn.cursor() as cur:
        cur.execute("SELECT make_chunk_table(%s)", (table_suffix,))
        cur.execute(f"TRUNCATE {table} RESTART IDENTITY")
    conn.commit()

    row_idx = 0
    batch = []
    t0 = time.time()
    total_rows = 0
    for case in tqdm(cases, desc=f"{table} 적재", ncols=80):
        chunks = chunk_document(case["content"], chunk_size, overlap)
        for chunk_index, text in enumerate(chunks):
            vec = embs[row_idx]
            batch.append((
                # 사건번호, 사건명, 법원명, 선고일자, 사건종류명, chunk 순서, chunk 원문, 임베딩
                case["case_no"], case["case_name"], case["court_name"],
                case["judgment_date"], case["case_type"],
                chunk_index, text, pg_vector_literal(vec),
            ))
            row_idx += 1
            if len(batch) >= COPY_BATCH:
                copy_rows(conn, table, batch)
                total_rows += len(batch)
                batch = []
    if batch:
        copy_rows(conn, table, batch)
        total_rows += len(batch)

    assert row_idx == embs.shape[0], f"row 수 불일치: 재생성된 chunk {row_idx}개 vs 캐시된 벡터 {embs.shape[0]}개"
    load_time = time.time() - t0
    print(f"적재 완료: {total_rows}행, {load_time:.1f}초")

    t0 = time.time()
    create_hnsw_index_with_progress(table)
    with conn.cursor() as cur:
        cur.execute(f"CREATE INDEX IF NOT EXISTS {table}_case_no_idx ON {table} (case_no)")
        cur.execute(f"CREATE INDEX IF NOT EXISTS {table}_case_type_idx ON {table} (case_type)")
    conn.commit()
    index_time = time.time() - t0
    print(f"인덱스 빌드 완료: {index_time:.1f}초")

    with conn.cursor() as cur:
        cur.execute(f"ANALYZE {table}")
        cur.execute(f"SELECT pg_size_pretty(pg_total_relation_size('{table}'))")
        size = cur.fetchone()[0]
    conn.commit()
    print(f"테이블+인덱스 총 용량: {size}")
    conn.close()

    return {
        "table": table, "chunk_size": chunk_size, "overlap": overlap, "rows": total_rows,
        "load_time_sec": round(load_time, 1), "index_build_time_sec": round(index_time, 1),
        "total_size": size,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-size", required=True, type=int)
    parser.add_argument("--overlap", type=int, default=0)
    args = parser.parse_args()

    conn = psycopg2.connect(DATABASE_URL)
    with conn.cursor() as cur, open(SCHEMA_PATH, encoding="utf-8") as f:
        cur.execute(f.read())
    conn.commit()
    conn.close()
    print("스키마(make_chunk_table 함수) 준비 완료")

    print("원본 코퍼스 로딩 중...")
    cases = load_corpus()
    print(f"코퍼스 문서 수: {len(cases)}")

    result = build_for_config(args.chunk_size, args.overlap, cases)

    print("\n===== 요약 =====")
    print(result)


if __name__ == "__main__":
    main()
