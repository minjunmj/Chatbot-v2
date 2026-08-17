"""
AIHub 판례 데이터 전처리 스크립트.

1) DB_data 구축 : 01.원천데이터(Training + Validation) 중 `판례내용`이 있는 json 파일만
   원본 그대로 data/DB_data/ 로 복사한다. (RAG 검색 대상이 될 원본 코퍼스)
2) 사건종류명 기준 stratified 85:15 분할 : DB_data에 포함된 사건들을 `사건종류명`별로
   그룹지어 각각 85% / 15% 비율로 무작위 분할한다. (특정 사건종류가 한쪽에 몰리지 않도록)
3) Train/Val 쿼리셋 구축 : 02.라벨링데이터(Training + Validation)를 순회하며
   jdgmnInfo.question을 query로, info.caseNoID를 정답 라벨(case_ids)로 매핑하되,
   1)에서 만든 분할 결과에 따라 data/Train/, data/Val/ 에 각각 저장한다.

사용법:
  python scripts/build_train_val_dataset.py
  python scripts/build_train_val_dataset.py --train-ratio 0.85 --seed 42
"""
import argparse
import json
import random
import shutil
from pathlib import Path

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_ROOT = (
    PROJECT_ROOT
    / "data"
    / "115.법률-규정_텍스트_분석_데이터_고도화_상황에_따른_판례_데이터"
    / "3.개방데이터"
    / "1.데이터"
)
SPLITS = ["Training", "Validation"]  # 원본 데이터셋의 폴더 구분 (우리가 새로 나눌 train/val과는 무관)

DEFAULT_DB_DATA_DIR = PROJECT_ROOT / "data" / "DB_data"
DEFAULT_TRAIN_DIR = PROJECT_ROOT / "data" / "Train"
DEFAULT_VAL_DIR = PROJECT_ROOT / "data" / "Val"
DEFAULT_TRAIN_RATIO = 0.85
DEFAULT_SEED = 42


def build_db_data(data_root: Path, db_data_dir: Path) -> dict[str, str]:
    """
    판례내용이 있는 원천데이터 json 파일을 db_data_dir로 복사한다.
    반환: {사건번호: 사건종류명} (stratified split에 사용)
    """
    db_data_dir.mkdir(parents=True, exist_ok=True)
    case_type_map: dict[str, str] = {}
    skipped_no_content = 0
    skipped_dup = 0

    for split in SPLITS:
        src_dir = data_root / split / "01.원천데이터"
        if not src_dir.is_dir():
            continue
        files = sorted(src_dir.glob("*.json"))
        for fp in tqdm(files, desc=f"[DB_data] {split}/01.원천데이터", ncols=90):
            try:
                raw = json.loads(fp.read_text(encoding="utf-8"))
            except Exception as e:
                print(f"⚠️  JSON 로드 실패: {fp} → {e}")
                continue

            content = (raw.get("판례내용") or "").strip()
            if not content:
                skipped_no_content += 1
                continue

            case_no = raw.get("사건번호")
            if not case_no:
                continue
            if case_no in case_type_map:
                skipped_dup += 1
                continue

            shutil.copy2(fp, db_data_dir / fp.name)
            case_type_map[case_no] = raw.get("사건종류명") or "기타"

    print(
        f"✅ DB_data: {len(case_type_map)}건 복사 완료 → {db_data_dir} "
        f"(판례내용 없음 제외 {skipped_no_content}건, 중복 사건번호 제외 {skipped_dup}건)"
    )
    return case_type_map


def stratified_split(
    case_type_map: dict[str, str], train_ratio: float, seed: int
) -> tuple[set[str], set[str]]:
    """사건종류명별로 train_ratio : (1-train_ratio) 비율로 무작위 분할"""
    by_type: dict[str, list[str]] = {}
    for case_no, case_type in case_type_map.items():
        by_type.setdefault(case_type, []).append(case_no)

    rng = random.Random(seed)
    train_case_nos: set[str] = set()
    val_case_nos: set[str] = set()

    print("=== 사건종류명별 분할 결과 ===")
    for case_type, case_nos in sorted(by_type.items(), key=lambda kv: -len(kv[1])):
        case_nos = sorted(case_nos)  # 셔플 전 정렬로 순서 고정 → 재현성 보장
        rng.shuffle(case_nos)
        n_train = round(len(case_nos) * train_ratio)
        train_case_nos.update(case_nos[:n_train])
        val_case_nos.update(case_nos[n_train:])
        print(f"  {case_type}: 전체 {len(case_nos)}건 → train {n_train} / val {len(case_nos) - n_train}")

    print(f"✅ 전체 train {len(train_case_nos)}건 / val {len(val_case_nos)}건")
    return train_case_nos, val_case_nos


def build_query_datasets(
    data_root: Path, train_case_nos: set[str], val_case_nos: set[str]
) -> tuple[list[dict], list[dict]]:
    """라벨링데이터의 question(query) + caseNoID(case_ids)를 train/val로 분배"""
    train_rows: list[dict] = []
    val_rows: list[dict] = []
    skipped_no_match = 0
    skipped_empty_q = 0

    for split in SPLITS:
        label_dir = data_root / split / "02.라벨링데이터"
        if not label_dir.is_dir():
            continue
        files = sorted(label_dir.glob("*.json"))
        for fp in tqdm(files, desc=f"[query]  {split}/02.라벨링데이터", ncols=90):
            try:
                label = json.loads(fp.read_text(encoding="utf-8"))
            except Exception as e:
                print(f"⚠️  JSON 로드 실패: {fp} → {e}")
                continue

            case_no = (label.get("info") or {}).get("caseNoID")
            if not case_no:
                continue

            if case_no in train_case_nos:
                bucket = train_rows
            elif case_no in val_case_nos:
                bucket = val_rows
            else:
                # DB_data(판례내용 있는 사건)에 없는 사건 → 검색 대상 자체가 없으므로 제외
                skipped_no_match += 1
                continue

            for qa in label.get("jdgmnInfo") or []:
                question = (qa.get("question") or "").strip()
                if not question:
                    skipped_empty_q += 1
                    continue
                bucket.append({"query": question, "case_ids": [case_no]})

    print(
        f"✅ query 데이터셋: train {len(train_rows)}건 / val {len(val_rows)}건 "
        f"(DB_data에 없는 사건 제외 {skipped_no_match}건, 빈 질문 제외 {skipped_empty_q}건)"
    )
    return train_rows, val_rows


def main():
    parser = argparse.ArgumentParser(
        description="원천데이터/라벨링데이터 → DB_data 구축 + 사건종류명 기준 85:15 Train/Val 쿼리셋 생성"
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--db-data-dir", type=Path, default=DEFAULT_DB_DATA_DIR)
    parser.add_argument("--train-dir", type=Path, default=DEFAULT_TRAIN_DIR)
    parser.add_argument("--val-dir", type=Path, default=DEFAULT_VAL_DIR)
    parser.add_argument("--train-ratio", type=float, default=DEFAULT_TRAIN_RATIO)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="사건종류명별 분할 시 사용할 랜덤 시드")
    args = parser.parse_args()

    case_type_map = build_db_data(args.data_root, args.db_data_dir)
    train_case_nos, val_case_nos = stratified_split(case_type_map, args.train_ratio, args.seed)
    train_rows, val_rows = build_query_datasets(args.data_root, train_case_nos, val_case_nos)

    args.train_dir.mkdir(parents=True, exist_ok=True)
    train_out = args.train_dir / "train_query.json"
    with train_out.open("w", encoding="utf-8") as f:
        json.dump(train_rows, f, ensure_ascii=False, indent=2)
    print(f"💾 train 쿼리셋 저장: {train_out}")

    args.val_dir.mkdir(parents=True, exist_ok=True)
    val_out = args.val_dir / "val_query.json"
    with val_out.open("w", encoding="utf-8") as f:
        json.dump(val_rows, f, ensure_ascii=False, indent=2)
    print(f"💾 val 쿼리셋 저장: {val_out}")


if __name__ == "__main__":
    main()