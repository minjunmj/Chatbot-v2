-- pgvector 기반 판례 chunk 테이블. chunk_size/overlap 조합별로 별도 테이블을 둬서
-- 정확도뿐 아니라 인덱스 빌드 시간/검색 latency/저장 용량까지 나란히 비교할 수 있게 한다.
-- (chunk 크기 최종 결정 전 단계 — docs/PROJECT_STATE.md 5절 참고)
--
-- make_chunk_table(suffix)만 정의하고 테이블은 미리 만들지 않음 — build_vector_db.py가
-- 필요한 조합(예: '200_overlap50', '300_overlap100')만 그때그때 SELECT make_chunk_table(...)로 생성.
--
-- 포함 필드: 사건명/사건번호/선고일자/법원명/사건종류명 (metadata) + 판례내용(chunk_text/embedding)
-- 그 외 원본 필드(판례일련번호/법원종류코드/사건종류코드/판결유형/선고/판례상세링크/판시사항/판결요지/참조조문/참조판례)는 미포함.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE OR REPLACE FUNCTION make_chunk_table(suffix text) RETURNS void AS $$
BEGIN
    EXECUTE format('
        CREATE TABLE IF NOT EXISTS chunks_%1$s (
            id             BIGSERIAL PRIMARY KEY,        -- 내부 PK (원본 데이터에는 없음)
            case_no        TEXT NOT NULL,                 -- 사건번호
            case_name      TEXT,                          -- 사건명
            court_name     TEXT,                          -- 법원명
            judgment_date  DATE,                          -- 선고일자
            case_type      TEXT,                          -- 사건종류명
            chunk_index    INT NOT NULL,                  -- 판례내용 내 chunk 순서 (원본에는 없음, 재구성용)
            chunk_text     TEXT NOT NULL,                 -- 판례내용 (chunk 단위로 분할된 조각)
            embedding      VECTOR(1024) NOT NULL           -- 판례내용 chunk의 임베딩 벡터
        )', suffix);
END;
$$ LANGUAGE plpgsql;
