"""
cross-encoder reranker(BAAI/bge-reranker-v2-m3)를 도메인 데이터로 파인튜닝한다.

prepare_reranker_data.py가 만든 reranker_pairs.jsonl(anchor=query, positive=정답 문서 내
최유사 chunk, negative_chunks=최종 dense 모델 기준 hard negative 여러 개)로, CrossEncoder용
MultipleNegativesRankingLoss(dense 모델 Phase A/B 학습 때와 같은 InfoNCE 계열, train.py 참고)를
써서 base 위에 이어서 학습한다. bge-reranker-v2-m3(범용, 한국어 법률 도메인 미세조정 안 됨)로
평가했을 때 dense 단독보다 오히려 recall이 낮게 나온 문제(docs/log.md 2026-08-25)를
"도메인 데이터로 이어서 학습"하는 방식으로 개선해보려는 시도 — KURE-v1 dense 모델을
파인튜닝해서 +10%p 개선했던 것과 같은 전략을 reranker에도 적용.

⚠️ 디스크 안전장치가 핵심: base 모델(HF 캐시, ~2.2GB)을 로드한 직후 그 캐시 폴더를 지운다.
가중치는 이미 메모리에 다 올라가 있어서 이후 학습/저장엔 디스크상의 원본 파일이 더 필요
없다 — 안 지우면 학습 완료 후 파인튜닝 모델(~2.2GB)을 저장할 때 base 캐시(2.2GB)와 동시에
존재해야 해서 도합 4.4GB가 필요한데, 이 인스턴스는 그럴 디스크 여유가 없음(docs/log.md
2026-08-25 디스크 점검 결과 여유 <2GB).

사용법:
    python prepare_reranker_data.py   # 먼저 실행해서 reranker_pairs.jsonl 생성
    python train_reranker.py
    python train_reranker.py --epochs 1 --batch-size 16
"""
import argparse
import json
import os
import shutil

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from datasets import Dataset
from sentence_transformers.cross_encoder import CrossEncoder, CrossEncoderTrainer, CrossEncoderTrainingArguments
from sentence_transformers.cross_encoder.losses import MultipleNegativesRankingLoss

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # server/finetune/
PAIRS_PATH = os.path.join(BASE_DIR, "reranker_pairs.jsonl")
OUTPUT_DIR = os.path.join(BASE_DIR, "output", "bge-reranker-v2-m3-finetuned")

BASE_MODEL = "BAAI/bge-reranker-v2-m3"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_dataset(path):
    rows = [json.loads(line) for line in open(path, encoding="utf-8")]
    # 행마다 negative_chunks 개수가 다를 수 있어(--num-negatives는 상한일 뿐, 코퍼스 사정에
    # 따라 더 적게 나올 수 있음) 파일 전체에서 가장 적은 개수로 맞춰서 열(column) 수를 고정함
    num_negatives = min(len(r["negative_chunks"]) for r in rows)
    print(f"쿼리당 사용할 negative 수: {num_negatives} (파일 내 최소값 기준)")

    data = {
        "anchor": [r["query"] for r in rows],
        "positive": [r["positive_chunk"] for r in rows],
    }
    for i in range(num_negatives):
        col = "negative" if num_negatives == 1 else f"negative_{i + 1}"
        data[col] = [r["negative_chunks"][i] for r in rows]
    return Dataset.from_dict(data)


def free_base_model_cache(model_name):
    """모델을 이미 메모리로 로드한 뒤 호출 — 상단 docstring의 "디스크 안전장치" 참고.
    학습 재개(resume) 없이 이번 프로세스 안에서 저장까지 끝내므로 캐시를 지워도 안전함."""
    from huggingface_hub import scan_cache_dir
    info = scan_cache_dir()
    for repo in info.repos:
        if repo.repo_id == model_name:
            size_gb = repo.size_on_disk / 1e9
            shutil.rmtree(repo.repo_path, ignore_errors=True)
            print(f"base 모델 캐시 삭제로 디스크 {size_gb:.2f}GB 회수: {repo.repo_path}")
            return
    print(f"경고: {model_name} 캐시를 못 찾음 — 디스크 회수 안 됨")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1,
                         help="이미 학습된 범용 reranker 위에 이어서 학습이라 과적합 위험 — "
                              "dense Phase B와 같은 이유로 기본값을 낮게 잡음(train.py 참고)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    args = parser.parse_args()

    if not os.path.exists(PAIRS_PATH):
        raise FileNotFoundError(f"{PAIRS_PATH} 없음 — prepare_reranker_data.py를 먼저 실행해야 함")

    print("학습 데이터 로딩 중...")
    train_dataset = load_dataset(PAIRS_PATH)
    print(f"학습 triplet 수: {len(train_dataset)}")

    print(f"base 모델 로딩: {BASE_MODEL}")
    # dtype 지정 안 함(train.py의 dense 학습과 동일) — 학습은 fp32로 올려서 TrainingArguments의
    # fp16=True가 알아서 mixed precision을 처리하게 둠. 추론 전용(run_eval.py)에서만 로딩
    # 시점에 바로 fp16으로 캐스팅해서 속도를 얻는 것과는 다른 상황(그쪽은 학습이 아니라서 gradient
    # 정밀도를 신경 안 써도 됨).
    model = CrossEncoder(BASE_MODEL, num_labels=1, device=DEVICE, max_length=512)

    # 로드 직후 바로 캐시 삭제 — 상단 docstring "디스크 안전장치" 참고
    free_base_model_cache(BASE_MODEL)

    loss = MultipleNegativesRankingLoss(model)

    training_args = CrossEncoderTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_checkpointing=True,
        learning_rate=args.lr,
        fp16=(DEVICE == "cuda"),
        warmup_steps=0.1,  # transformers v5+에서 warmup_ratio가 deprecated, warmup_steps에 float(비율)로 대체
        dataloader_drop_last=True,
        save_strategy="no",  # dense train.py와 동일 이유: 중간 체크포인트는 디스크만 잡아먹음
        logging_steps=50,
        report_to="none",
    )

    trainer = CrossEncoderTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        loss=loss,
    )
    trainer.train()

    model.save(args.output_dir)
    print(f"파인튜닝 완료, 저장 위치: {args.output_dir}")


if __name__ == "__main__":
    main()
