"""
DB_data(원본 JSON) + 이미 계산된 chunk 임베딩 캐시(server/cache_embeddings/)를
pgvector 테이블에 적재한다. chunk_size + overlap 조합을 임의로 받는다.
dense 임베딩 외에, Kiwi로 형태소 분석한 sparse/BM25 검색용 컬럼(chunk_text_kiwi, content_tsv)
도 같이 채운다 (docs/log.md 2026-08-21 논의 — Postgres 내장 전문검색+Kiwi, pgvector 0.6.0이라
sparsevec 미지원이라 BGE-M3 sparse 대신 이 방식 채택).

재임베딩은 하지 않는다 — kure_chunk.py/kure_chunk_overlap.py/eval_model.py류가 저장한 .npy는
chunk 텍스트/사건번호 배열을 같이 저장하지 않았기 때문에(벡터만 저장됨), 이 스크립트는 그
스크립트들과 완전히 동일한 로직(load_corpus의 sorted glob 순서, chunk_document의 분할 방식)
으로 chunk 텍스트/사건번호를 재생성해서 캐시된 벡터와 순서를 맞춘다. 문서 슬라이싱만 다시
하는 거라 GPU 불필요, 수 초면 끝남.

npy 파일명 규칙(기본, --npy-path로 임의 경로 지정 가능):
    overlap=0  → kure-v1_chunk{size}_corpus.npy            (kure_chunk.py 산출물)
    overlap>0  → kure-v1_chunk{size}_overlap{overlap}_corpus.npy  (kure_chunk_overlap.py 산출물)
테이블명 규칙:
    overlap=0  → chunks_{size}
    overlap>0  → chunks_{size}_overlap{overlap}

사용법:
    python build_vector_db.py --chunk-size 300 --overlap 100
    # 파인튜닝 모델 등 다른 npy 캐시를 쓰고 싶을 때
    python build_vector_db.py --chunk-size 300 --overlap 100 \
        --npy-path ../cache_embeddings/eval_._output_kure-v1-finetuned-hard_chunk300_overlap100_corpus.npy
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
from kiwipiepy import Kiwi
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # server/
DB_DATA_DIR = os.path.join(BASE_DIR, "..", "data", "DB_data")
CACHE_DIR = os.path.join(BASE_DIR, "cache_embeddings")
SCHEMA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "schema.sql")

# postgresql://{유저명}:{비밀번호}@{호스트}:{포트}/{DB이름}
# 환경변수 DATABASE_URL이 있으면 그걸 쓰고(.env에서 로드됨), 없으면 이 기본값을 씀
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:lexchatbot_dev@127.0.0.1:5432/lexchatbot")
COPY_BATCH = 50_000

# 조사(J*)/어미(E*)/접미사(XS*)/문장부호(S[FPS])는 제외 — 명사/동사/부사/숫자/한자/외국어 등
# 실질형태소만 남겨서 검색 노이즈를 줄임 (Kiwi 태그 기준)
# NN: 명사(NNG 일반명사/NNP 고유명사/NNB 의존명사 등 NN으로 시작하는 전부)
# NP: 대명사(나/너/우리 등)       NR: 수사(하나/둘/일/이 등 숫자를 나타내는 말)
# VV: 동사(가다/먹다)             VA: 형용사(예쁘다/크다)        VX: 보조용언(있다/않다 등 본동사 뒤에 붙는 것)
# MAG: 일반부사(빨리/매우)        MAJ: 접속부사(그러나/그리고)
# SL: 외국어(alphabet)           SH: 한자                      SN: 숫자(750 같은 아라비아 숫자)
# XR: 어근(반짝반짝의 '반짝'처럼 단독으론 안 쓰이지만 의미를 가진 뿌리 부분)
KIWI_KEEP_PREFIXES = ("NN", "NP", "NR", "VV", "VA", "VX", "MAG", "MAJ", "SL", "SH", "SN", "XR")


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


def build_chunk_rows(cases, chunk_size, overlap):
    """cases를 chunk 단위로 펼쳐서, embs와 순서가 맞는 병렬 리스트들을 반환."""
    chunk_texts = []
    row_meta = []  # (case_no, case_name, court_name, judgment_date, case_type, chunk_index)
    for case in cases:
        chunks = chunk_document(case["content"], chunk_size, overlap)
        for chunk_index, text in enumerate(chunks):
            chunk_texts.append(text)
            row_meta.append((
                case["case_no"], case["case_name"], case["court_name"],
                case["judgment_date"], case["case_type"], chunk_index,
            ))
    return chunk_texts, row_meta


def kiwi_tokenize_all(chunk_texts):
    """전체 chunk_text를 형태소 분석해서 실질형태소만 공백으로 이어붙인 문자열 리스트로 반환.
    num_workers=-1(전체 코어) 배치 처리 — 82만 개 기준 약 7분(이 인스턴스 실측)."""
    kiwi = Kiwi(num_workers=-1)
    # kiwi.tokenize(chunk_texts)는 문장(chunk)마다 하나씩 토큰 리스트를 돌려줌 — 2단계 구조:
    #   [ [Token(form='손해',tag='NNG'), Token(form='배상',tag='NNG'), Token(form='을',tag='JKO'), ...],  <- chunk_texts[0]의 토큰들
    #     [Token(form='임대차',tag='NNG'), ...],                                                            <- chunk_texts[1]의 토큰들
    #     ... ]
    # 바깥 for는 이 리스트를 chunk 하나씩(=tokens) 꺼내고, 안쪽 for는 그 chunk 안의 토큰(t)을 하나씩 훑음.
    # t.form=형태소 글자, t.tag=품사(NNG/JKO 등, KIWI_KEEP_PREFIXES 주석 참고)
    results = []
    for tokens in tqdm(kiwi.tokenize(chunk_texts), total=len(chunk_texts), desc="Kiwi 형태소 분석", ncols=80):
        kept = [t.form for t in tokens if t.tag.startswith(KIWI_KEEP_PREFIXES)]
        results.append(" ".join(kept))
    return results


def pg_vector_literal(vec):
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"


def copy_rows(conn, table, rows):
    """많은 row를 한 줄씩 INSERT하지 않고, Postgres의 COPY(대량 적재 전용 명령)로 한 번에
    밀어넣는 함수 — INSERT를 수만 번 반복하는 것보다 훨씬 빠름.
    rows: (case_no, case_name, court_name, judgment_date, case_type, chunk_index, chunk_text,
    chunk_text_kiwi, embedding_literal) 튜플 리스트"""
    buf = io.StringIO()  # 디스크에 진짜 파일을 안 만들고, 메모리 안에서 파일처럼 동작하는 버퍼
    for r in rows:
        buf.write("\t".join(
            # COPY FORMAT text는 컬럼을 탭(\t)으로 구분함 — None은 Postgres가 NULL로 알아듣는
            # 특수 표기 \N으로, 값 안에 진짜 탭/줄바꿈이 섞여있으면 구분자로 오해되지 않게
            # 공백으로 치환(\\는 이스케이프 처리)
            "\\N" if v is None else str(v).replace("\\", "\\\\").replace("\t", " ").replace("\n", " ")
            for v in r
        ) + "\n")  # row 하나 끝 = 줄바꿈
    buf.seek(0)  # 다 쓴 뒤 buf의 "쓰기 위치"가 맨 끝에 가있어서, 읽으려면 처음으로 되감아야 함
    with conn.cursor() as cur:
        cur.copy_expert(
            # 사건번호, 사건명, 법원명, 선고일자, 사건종류명, chunk 순서, chunk 원문, kiwi 토큰, 임베딩
            # COPY ... FROM STDIN: buf(메모리 파일)의 내용을 이 컬럼 순서 그대로 테이블에 흘려넣음
            f"COPY {table} (case_no, case_name, court_name, judgment_date, case_type, "
            f"chunk_index, chunk_text, chunk_text_kiwi, embedding) FROM STDIN WITH (FORMAT text)",
            buf,
        )
    conn.commit()  # 확정 저장


def create_hnsw_index_with_progress(table):
    """CREATE INDEX는 단일 SQL문이라 자체 진행률이 없음 — 별도 스레드에서 실행하는 동안
    pg_stat_progress_create_index 뷰를 폴링해서 현재 phase/처리 tuple 수를 보여준다."""
    def worker():
        conn2 = psycopg2.connect(DATABASE_URL) # psycopg2로 Postgres에 실제 접속 -> 파이썬이 PostgreSQL이랑 대화할 수 있게 해주는 통역사 역할
        with conn2.cursor() as cur:
            cur.execute(f"CREATE INDEX IF NOT EXISTS {table}_embedding_hnsw ON {table} "
                        f"USING hnsw (embedding vector_cosine_ops)")
        conn2.commit()
        conn2.close()

    t = threading.Thread(target=worker) # 스레드 생성 worker를 실행하라고 지정 백그라운드에서 실행, 스레드가 없으면 메인 스레드가 블로킹되어 진행률을 표시할 수 없음
    t.start()

    poll_conn = psycopg2.connect(DATABASE_URL)
    pbar = tqdm(desc=f"{table} HNSW 인덱스 빌드", unit="s", ncols=80, bar_format="{desc}: {elapsed} 경과 | {postfix}")
    while t.is_alive():
        with poll_conn.cursor() as cur:
            cur.execute("SELECT phase, tuples_done, tuples_total FROM pg_stat_progress_create_index") # pg_ 로 실행하는건 내장 시스템 정보용 테이블
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


def build_for_config(chunk_size, overlap, cases, npy_path=None):
    table = f"chunks_{chunk_size}" if overlap == 0 else f"chunks_{chunk_size}_overlap{overlap}"
    table_suffix = str(chunk_size) if overlap == 0 else f"{chunk_size}_overlap{overlap}"

    if npy_path is None:
        npy_name = (f"kure-v1_chunk{chunk_size}_corpus.npy" if overlap == 0
                    else f"kure-v1_chunk{chunk_size}_overlap{overlap}_corpus.npy")
        npy_path = os.path.join(CACHE_DIR, npy_name)
    if not os.path.exists(npy_path):
        raise FileNotFoundError(f"{npy_path} 없음 — 임베딩 캐시를 먼저 만들어야 함")

    print(f"\n===== chunk_size={chunk_size}, overlap={overlap} (table={table}) =====")
    print(f"임베딩 캐시: {npy_path}")
    embs = np.load(npy_path)
    print(f"임베딩 로드: {embs.shape}")

    print("chunk 텍스트 재생성 중...")
    chunk_texts, row_meta = build_chunk_rows(cases, chunk_size, overlap)
    assert len(chunk_texts) == embs.shape[0], (
        f"row 수 불일치: 재생성된 chunk {len(chunk_texts)}개 vs 캐시된 벡터 {embs.shape[0]}개")
    print(f"chunk 개수: {len(chunk_texts)}")

    chunk_texts_kiwi = kiwi_tokenize_all(chunk_texts)
    
    conn = psycopg2.connect(DATABASE_URL) #접속
    with conn.cursor() as cur:
        cur.execute("SELECT make_chunk_table(%s)", (table_suffix,))
        # 이전 빌드(예: base 모델용)에서 이미 인덱스가 붙어있는 테이블을 재구축하는 경우,
        # TRUNCATE는 데이터만 비우고 인덱스는 그대로 둠 — 그 상태로 COPY하면 매 row마다
        # HNSW 인덱스를 실시간으로 갱신해야 해서 극도로 느려짐(수십~수백 배). 그래서 데이터
        # 넣기 전에 기존 인덱스를 먼저 지우고, 다 넣은 뒤(아래) 한 번에 다시 빌드한다.
        cur.execute(f"DROP INDEX IF EXISTS {table}_embedding_hnsw")
        cur.execute(f"DROP INDEX IF EXISTS {table}_case_no_idx")
        cur.execute(f"DROP INDEX IF EXISTS {table}_case_type_idx")
        cur.execute(f"DROP INDEX IF EXISTS {table}_content_tsv_gin")
        cur.execute(f"TRUNCATE {table} RESTART IDENTITY")
    conn.commit()

    t0 = time.time()
    batch = []
    total_rows = 0
    for meta, text, text_kiwi, vec in tqdm(
            zip(row_meta, chunk_texts, chunk_texts_kiwi, embs), total=len(chunk_texts),
            desc=f"{table} 적재", ncols=80):
        batch.append((*meta, text, text_kiwi, pg_vector_literal(vec)))
        if len(batch) >= COPY_BATCH:
            copy_rows(conn, table, batch)
            total_rows += len(batch)
            batch = []
    if batch:
        copy_rows(conn, table, batch)
        total_rows += len(batch)

    load_time = time.time() - t0
    print(f"적재 완료: {total_rows}행, {load_time:.1f}초")

    t0 = time.time()
    with conn.cursor() as cur:
        cur.execute(f"UPDATE {table} SET content_tsv = to_tsvector('simple', coalesce(chunk_text_kiwi, ''))")
    conn.commit()
    tsv_time = time.time() - t0
    print(f"content_tsv 채우기 완료: {tsv_time:.1f}초")

    t0 = time.time()
    create_hnsw_index_with_progress(table)
    with conn.cursor() as cur:
        cur.execute(f"CREATE INDEX IF NOT EXISTS {table}_case_no_idx ON {table} (case_no)")
        cur.execute(f"CREATE INDEX IF NOT EXISTS {table}_case_type_idx ON {table} (case_type)")
        cur.execute(f"CREATE INDEX IF NOT EXISTS {table}_content_tsv_gin ON {table} USING gin (content_tsv)")
    conn.commit()
    index_time = time.time() - t0
    print(f"인덱스 빌드 완료(HNSW+case_no+case_type+GIN): {index_time:.1f}초")

    with conn.cursor() as cur:
        cur.execute(f"ANALYZE {table}")
        cur.execute(f"SELECT pg_size_pretty(pg_total_relation_size('{table}'))")
        size = cur.fetchone()[0] # fetchone 튜플을 들고옴 값을 꺼내려면 [0]으로 접근
    conn.commit()
    print(f"테이블+인덱스 총 용량: {size}")
    conn.close()

    return {
        "table": table, "chunk_size": chunk_size, "overlap": overlap, "rows": total_rows,
        "load_time_sec": round(load_time, 1), "tsv_time_sec": round(tsv_time, 1),
        "index_build_time_sec": round(index_time, 1), "total_size": size,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-size", required=True, type=int)
    parser.add_argument("--overlap", type=int, default=0)
    parser.add_argument("--npy-path", default=None,
                         help="임베딩 캐시 경로 직접 지정(기본: cache_embeddings/kure-v1_chunk{size}[_overlap{overlap}]_corpus.npy)")
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

    result = build_for_config(args.chunk_size, args.overlap, cases, npy_path=args.npy_path)

    print("\n===== 요약 =====")
    print(result)


if __name__ == "__main__":
    main()
