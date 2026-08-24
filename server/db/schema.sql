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

-- $$ ... $$ : Postgres의 "달러 인용(dollar-quoting)" 문법 — 함수 본문처럼 긴 문자열을
-- 작은따옴표 이스케이프 없이 감싸는 용도. $$는 가장 기본형 태그일 뿐, $아무이름$...$아무이름$
-- 처럼 태그를 직접 지어도 됨(중첩해서 써야 할 때만 서로 다른 태그로 구분).
CREATE OR REPLACE FUNCTION make_chunk_table(suffix text) RETURNS void AS $$
BEGIN
    -- format('...%1$s...', suffix) : %1$s는 format()의 1번째 인자(suffix)를 문자열(s)로
    -- 끼워넣으라는 자리표시자. %=시작표시, 1=몇 번째 인자, $=구분자, s=타입(string).
    -- 인자가 suffix 하나뿐이라 %s로 써도 결과는 같지만, 인자 번호를 명시해두면 나중에
    -- 인자가 늘어나도 이 자리는 항상 1번째 값으로 고정돼서 더 명확함.
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
            embedding      VECTOR(1024) NOT NULL,          -- 판례내용 chunk의 임베딩 벡터 (dense)
            chunk_text_kiwi TEXT,                         -- Kiwi로 형태소 분석한 chunk_text (조사/어미 제거, 공백 조인)
            content_tsv    tsvector                       -- chunk_text_kiwi로부터 생성 (sparse/BM25 검색용, GIN 인덱스 필요)
        )', suffix);

    -- 이미 만들어져 있던(구버전 스키마) 테이블에도 새 컬럼을 안전하게 추가
    EXECUTE format('ALTER TABLE chunks_%1$s ADD COLUMN IF NOT EXISTS chunk_text_kiwi TEXT', suffix);
    EXECUTE format('ALTER TABLE chunks_%1$s ADD COLUMN IF NOT EXISTS content_tsv tsvector', suffix);
END;
$$ LANGUAGE plpgsql;
