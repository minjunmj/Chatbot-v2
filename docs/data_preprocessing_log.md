# 판례 데이터 전처리 작업 기록

작성일: 2026-08-16
관련 스크립트: [scripts/build_train_val_dataset.py](../scripts/build_train_val_dataset.py)

이 문서는 AIHub 판례 데이터(`115.법률-규정_텍스트_분석_데이터_고도화_상황에_따른_판례_데이터`)를
RAG 챗봇용으로 전처리하기까지 진행한 분석과 의사결정 과정을 정리한 기록입니다.

---

## 1. 데이터 구조 파악

데이터셋은 `Training`/`Validation` 두 split 아래에 각각 `01.원천데이터`(원문 전체)와
`02.라벨링데이터`(정제/요약본)가 있는 구조입니다.

```
data/115.법률-규정_텍스트_분석_데이터_고도화_상황에_따른_판례_데이터/3.개방데이터/1.데이터/
├── Training/
│   ├── 01.원천데이터/   (판례 원문 json)
│   └── 02.라벨링데이터/ (요약/Q&A/키워드 등 라벨링 json)
├── Validation/
│   ├── 01.원천데이터/
│   └── 02.라벨링데이터/
├── Other/QA데이터/
└── Sublabel/ (판례, 심결례)
```

### 1-1. 원천데이터 필드

| 필드 | 의미 |
|---|---|
| `판시사항` | 판결이 다룬 법적 쟁점을 한 줄로 요약한 제목(headline) |
| `판결요지` | 판시사항에 대한 법원의 결론·핵심 법리를 요약한 문단 |
| `참조조문` | 이 판결이 적용/해석한 법령 조항 목록 (예: "민법 제812조,제815조") |
| `참조판례` | 이 판결이 근거로 삼거나 인용한 선행 판례 목록 |
| `판례내용` | 판결문 전문(全文) — 당사자, 원심판결, 주문, 청구취지, 이유 전체 포함 |

그 외 식별/분류용 필드: `판례일련번호`, `사건명`, `사건번호`, `선고일자`, `법원명`,
`사건종류명`(민사/형사/가사/일반행정/세무/특허 등 22개 카테고리), `판결유형` 등.

> **판례내용이 없는 문서(15,155건)** 는 전부 `사건종류명`이 `None`인 **행정심판재결례**로,
> 스키마 자체가 달라(`이유`/`주문`/`청구취지` 필드 사용) `판례내용` 필드가 존재하지 않음.
> → 판례(法) 문서와 재결례(행정) 문서가 같은 폴더에 섞여있다는 점에 주의.

### 1-2. 라벨링데이터 필드

| 필드 | 의미 |
|---|---|
| `info.caseNo` / `caseNoID` | 사건번호 (원천데이터의 `사건번호`와 매칭되는 키) |
| `info.courtNm`, `courtType`, `judmnAdjuDe` | 법원명, 판례/결정 구분, 선고일자 |
| `info.caseNm`, `caseTitle` | 사건명, 정식 인용 표기 |
| `jdgmn` | 판시사항 제목 요약 (원천데이터 `판시사항`과 유사한 한 줄 headline) |
| `jdgmnInfo[].question` / `.answer` | 판례 쟁점을 Q&A 형태로 재구성한 것 |
| `Summary[].summ_contxt` | **원천데이터의 `판결요지`와 사실상 동일** (직접 확인함, 94도852 사례 등에서 100% 일치) |
| `Summary[].summ_pass` | `summ_contxt`에서 핵심 문장만 더 추출한 축약본 (부분집합) |
| `keyword_tagg[].keyword` | 판례 키워드 태그 |
| `Reference_info.reference_rules/court_case` | 참조조문/참조판례 (원천데이터와 대응) |
| `Class_info.class_name`, `instance_name` | 사건 분류/사건명 |

---

## 2. 분석 결과

### 2-1. `판결요지` vs `summ_contxt` 관계 검증

94도852 사례로 직접 비교한 결과:
```
판결요지 (원천데이터)  ≈  summ_contxt (라벨링, 전체 요지)   ← 거의 동일
                              ⊃
                         summ_pass (라벨링, 핵심 문장만 추출한 더 짧은 요약)
```
→ 라벨링데이터의 `summ_contxt`는 원천데이터 `판결요지`를 그대로 옮겨 담은 것에 가까움.
   임베딩 소스로는 더 짧은 `summ_pass` + `jdgmn`이 노이즈가 적어 유리할 것으로 판단.
   (단, 이 결론은 1건 샘플 기준이므로 여러 건 검증 필요)

### 2-2. `판례내용` 토큰 수 통계 (cl100k_base 기준, 원천데이터 44,705건)

| 지표 | 값 |
|---|---|
| 평균 | 약 3,491 토큰 |
| 중앙값 | 2,465 토큰 |
| 최소 / 최대 | 4 / 212,558 토큰 |
| 90th percentile | 6,446 토큰 |

분포가 오른쪽으로 심하게 치우쳐 있어(평균 > 중앙값), 극단적으로 긴 문서 소수가 평균을 끌어올림.
→ `판례내용` 원문을 통째로 임베딩하면 대부분의 임베딩 모델 context window를 초과하거나
   청킹이 필요함을 시사. 청킹 전략은 평균보다 중앙값/90th percentile 기준으로 설계 권장.

### 2-3. `사건종류명` 분포 (판례내용 있는 문서 기준)

민사(18,029) > 형사(12,918) > 일반행정(5,499) > 세무(3,516) > 특허(2,391) > 가사(836) 등
총 22개 카테고리, 최소 9건(제조물_책임_민사)까지 존재. 카테고리별 건수 편차가 크므로
train/val을 무작위로 통째로 나누면 소수 카테고리가 한쪽에 쏠릴 위험 있음
→ 3-3절의 stratified split 설계로 이어짐.

---

## 3. context / metadata 설계 논의 (보류 중)

라벨링데이터의 각 필드를 RAG의 **검색 대상(context/page_content)** 과
**필터·인용용 부가정보(metadata)** 중 어디에 쓸지에 대한 1차 제안:

- **metadata 후보**: `사건번호`(PK), `법원명`, `선고일자`, `사건명`/`caseTitle`,
  `Class_info`(카테고리), `keyword_tagg`(태그)
- **context(임베딩 대상) 후보**: `jdgmn`, `Summary`(summ_contxt/summ_pass), `jdgmnInfo`(Q&A)
- **참조조문/참조판례**: metadata와 context 둘 다 필요할 수 있음 (구조화 필터 + 텍스트 검색 양쪽 대응)

> **현재 상태: 이 결정은 보류**. 사용자 요청에 따라 지금 단계에서는 필드 가공 없이
> "판례내용이 있는 원본 데이터를 그대로 추출"하는 것까지만 진행하고,
> context/metadata 분리는 이후 별도로 결정하기로 함.

---

## 4. `scripts/build_train_val_dataset.py` 개발 이력

### v1 — page_content/metadata 분리 + eval 쿼리셋
- `01.원천데이터`에서 `판례내용` 있는 문서만 골라 `page_content`(판시사항/판결요지/판례내용 결합)와
  `metadata`(사건번호 등)로 가공 → `data/processed/train_corpus.jsonl`
- `02.라벨링데이터`의 `jdgmnInfo.question`을 query, `info.caseNoID`를 정답 라벨로 사용해
  `server/acc_test.py`가 읽는 `data/evaluate/evaldata.json` 포맷으로 저장
- 실행 결과: train corpus 44,700건, eval 쿼리 48,599건 생성 후 **검증 목적으로 실제 실행**
- 이후 사용자 요청으로 생성된 산출물 파일은 삭제 (스크립트만 유지)

### v2 — 단순 추출로 축소
- "context/metadata 분리는 나중에 결정" → `build_page_content()` 제거,
  원천데이터를 가공 없이 원본 dict 그대로 추출하도록 단순화

### v3 (현재) — DB_data 물리 복사 + 사건종류명 기준 85:15 stratified split
사용자가 데이터 구조를 다음과 같이 재정의:
1. `data/DB_data/` : `판례내용`이 있는 원천데이터 json **파일 자체를 그대로 복사**
   (RAG 검색 대상이 될 원본 코퍼스)
2. `data/DB_data`의 사건들을 **`사건종류명`별로 그룹지어 각각 85:15로 무작위 분할**
   (카테고리 쏠림 방지)
3. `02.라벨링데이터`를 순회하며 `question`/`caseNoID`를 위 분할 결과에 따라
   `data/Train/train_query.json`, `data/Val/val_query.json`으로 분리 저장

#### 함수 구성

| 함수 | 입력 | 반환 | 역할 |
|---|---|---|---|
| `build_db_data(data_root, db_data_dir)` | 데이터 루트 경로, 복사 대상 경로 | `{사건번호: 사건종류명}` | 판례내용 있는 파일을 `DB_data`로 복사 |
| `stratified_split(case_type_map, train_ratio, seed)` | 위 매핑, 비율, 시드 | `(train 사건번호 set, val 사건번호 set)` | 사건종류명별 85:15 분할 |
| `build_query_datasets(data_root, train_case_nos, val_case_nos)` | 데이터 루트, 두 집합 | `(train rows, val rows)` | 라벨링데이터의 질문을 분할 결과에 따라 배분 |
| `main()` | CLI 인자 (`--data-root`, `--db-data-dir`, `--train-dir`, `--val-dir`, `--train-ratio`, `--seed`) | 없음(파일 저장) | 위 세 함수를 순서대로 실행 |

#### 출력 경로 (기본값)
- `data/DB_data/*.json` — 원본 판례 json 파일 복사본
- `data/Train/train_query.json` — `[{"query": ..., "case_ids": [...]}, ...]`
- `data/Val/val_query.json` — 동일 포맷

재현성을 위해 `sorted()`로 파일 순서를 고정한 뒤 `random.Random(seed)`로 셔플 → 같은 시드에서
항상 같은 분할 결과가 나오도록 설계.

---

## 5. 남은 작업 (TODO)

- [ ] `page_content`/`metadata` 필드 구성 최종 결정 (2절 논의 이어서 진행)
- [ ] `DB_data`를 실제 FAISS 인덱스로 빌드하는 스크립트 작성
- [ ] `server/acc_test.py`의 평가 데이터 경로(`./data/evaluate/evaldata.json`)를
      새 구조(`data/Val/val_query.json`)에 맞게 갱신할지 결정
- [ ] `summ_contxt` ≈ `판결요지` 동일성 여부, 여러 건으로 재검증