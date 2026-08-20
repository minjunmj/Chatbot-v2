"""
KURE-v1을 finetune_pairs.jsonl(prepare_data.py 산출물)로 파인튜닝한다 — 1단계: in-batch
negative만 사용 (hard negative는 2단계에서 별도 스크립트로 추가 예정, docs/log.md 2026-08-20 참고).

loss는 MultipleNegativesRankingLoss(InfoNCE 계열) — BGE-M3/KURE-v1 원래 학습 방식과 같은
계열. 배치 안 (query, positive) 쌍들에서 자기 자신 외 나머지가 전부 negative로 쓰이므로,
batch_size가 클수록 negative가 많아져 학습 신호가 좋아짐 — VRAM이 허용하는 한 크게 잡을 것
(gradient accumulation은 유효 batch는 키우지만 physical batch당 negative pool은 안 늘어남).

train_query.json이 문서 단위로 순회 생성돼 있어 안 섞으면 인접 쿼리가 같은 case_id를 가리키는
비율이 8.15%로 실측됨(false negative 위험) — 그래서 매 epoch 셔플이 필수. HuggingFace
Trainer는 기본적으로 매 epoch 셔플하지만, 명시적으로 사용 중임을 확인해뒀다(아래 참고).

사용법:
    python train.py                                  # 기본 설정으로 학습
    python train.py --epochs 3 --batch-size 32 --output-dir ./output/kure-v1-finetuned-inbatch
"""
import argparse
import json
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PAIRS_PATH = os.path.join(BASE_DIR, "finetune_pairs.jsonl")
DEFAULT_OUTPUT_DIR = os.path.join(BASE_DIR, "output", "kure-v1-finetuned-inbatch")

REPO = "nlpai-lab/KURE-v1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_dataset(path):
    queries, positives = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            queries.append(row["query"])
            positives.append(row["positive_chunk"])
    return Dataset.from_dict({"anchor": queries, "positive": positives})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32,
                         help="in-batch negative 개수(batch_size-1)에 직결 — VRAM 허용 한도까지 키울 것")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    if not os.path.exists(PAIRS_PATH):
        raise FileNotFoundError(f"{PAIRS_PATH} 없음 — prepare_data.py를 먼저 실행해야 함")

    print("학습 데이터 로딩 중...")
    train_dataset = load_dataset(PAIRS_PATH)
    print(f"학습 쌍 수: {len(train_dataset)}")

    print(f"베이스 모델 로딩: {REPO}")
    model = SentenceTransformer(REPO, device=DEVICE)

    loss = MultipleNegativesRankingLoss(model)

    training_args = SentenceTransformerTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        fp16=(DEVICE == "cuda"),
        # in-batch negative 방식이라 마지막에 크기 안 맞는 배치가 negative 수를 흔들지 않게 버림
        dataloader_drop_last=True,
        # HuggingFace Trainer 기본값 — 매 epoch 셔플 (false negative 완화에 필수, log.md 참고)
        # 명시적으로 남겨서 나중에 실수로 안 꺼지게 함
        save_strategy="epoch",
        logging_steps=50,
    )

    trainer = SentenceTransformerTrainer(
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
