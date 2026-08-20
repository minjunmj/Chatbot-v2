"""
KURE-v1을 파인튜닝한다. 두 단계를 이 스크립트 하나로 처리:
- 1단계(in-batch negative만): --pairs-file finetune_pairs.jsonl (prepare_data.py 산출물)
- 2단계(hard negative 추가): --pairs-file finetune_pairs_hard.jsonl (mine_hard_negatives.py 산출물)
  → 파일에 hard_negative_chunk 필드가 있으면 자동으로 (anchor, positive, negative) 3열
  데이터셋을 만들고, 기본 시작점도 1단계 결과 모델로 바뀐다 (처음부터 다시 학습하는 게 아니라
  1단계에서 이미 배운 것 위에 이어서 학습 — 근거: docs/log.md 2026-08-20)

loss는 MultipleNegativesRankingLoss(InfoNCE 계열) — BGE-M3/KURE-v1 원래 학습 방식과 같은
계열. (anchor, positive) 2열이면 배치 안 나머지가 전부 negative(in-batch), 3열(negative 포함)
이면 명시적 negative가 배치 전체에 추가로 더해짐. batch_size가 클수록 negative가 많아져
학습 신호가 좋아짐 — VRAM이 허용하는 한 크게 잡을 것(gradient accumulation은 유효 batch는
키우지만 physical batch당 negative pool은 안 늘어남).

train_query.json이 문서 단위로 순회 생성돼 있어 안 섞으면 인접 쿼리가 같은 case_id를 가리키는
비율이 8.15%로 실측됨(false negative 위험) — 그래서 매 epoch 셔플이 필수. HuggingFace
Trainer는 기본적으로 매 epoch 셔플하지만, 명시적으로 사용 중임을 확인해뒀다(아래 참고).

사용법:
    python train.py                                                    # 1단계 (기본)
    python train.py --pairs-file finetune_pairs_hard.jsonl             # 2단계 (hard negative)
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
INBATCH_PAIRS_PATH = os.path.join(BASE_DIR, "finetune_pairs.jsonl")
HARD_PAIRS_PATH = os.path.join(BASE_DIR, "finetune_pairs_hard.jsonl")

REPO = "nlpai-lab/KURE-v1"
PHASE_A_OUTPUT_DIR = os.path.join(BASE_DIR, "output", "kure-v1-finetuned-inbatch")
PHASE_B_OUTPUT_DIR = os.path.join(BASE_DIR, "output", "kure-v1-finetuned-hard")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_dataset(path):
    rows = [json.loads(line) for line in open(path, encoding="utf-8")]
    has_hard_negative = "hard_negative_chunk" in rows[0]

    data = {
        "anchor": [r["query"] for r in rows],
        "positive": [r["positive_chunk"] for r in rows],
    }
    if has_hard_negative:
        data["negative"] = [r["hard_negative_chunk"] for r in rows]
    return Dataset.from_dict(data), has_hard_negative


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs-file", default=None,
                         help="기본: finetune_pairs.jsonl (1단계). finetune_pairs_hard.jsonl 등 지정 가능")
    parser.add_argument("--base-model", default=None,
                         help="기본: 1단계는 base KURE-v1, 2단계(hard negative 파일)는 1단계 결과 모델")
    parser.add_argument("--epochs", type=int, default=None,
                         help="기본값: 1단계 3, 2단계(이미 적응된 모델 위에 이어서 학습이라 과적합 위험) 1")
    parser.add_argument("--batch-size", type=int, default=None,
                         help="in-batch negative 개수(batch_size-1)에 직결 — VRAM 허용 한도까지 키울 것. "
                              "기본값: 1단계 32, 2단계(hard negative, 텍스트 3열이라 더 무거움) 16")
    parser.add_argument("--lr", type=float, default=None,
                         help="기본값: 1단계 2e-5, 2단계(계속 학습이라 더 조심스럽게) 5e-6")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    pairs_path = args.pairs_file or INBATCH_PAIRS_PATH
    if not os.path.isabs(pairs_path):
        pairs_path = os.path.join(BASE_DIR, pairs_path)
    if not os.path.exists(pairs_path):
        raise FileNotFoundError(
            f"{pairs_path} 없음 — prepare_data.py(1단계) 또는 mine_hard_negatives.py(2단계)를 먼저 실행해야 함")

    print("학습 데이터 로딩 중...")
    train_dataset, has_hard_negative = load_dataset(pairs_path)
    print(f"학습 쌍 수: {len(train_dataset)}, hard negative 포함: {has_hard_negative}")

    base_model = args.base_model
    if base_model is None:
        base_model = PHASE_A_OUTPUT_DIR if has_hard_negative else REPO
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = PHASE_B_OUTPUT_DIR if has_hard_negative else PHASE_A_OUTPUT_DIR
    batch_size = args.batch_size
    if batch_size is None:
        # hard negative 있으면 배치당 텍스트 3개(anchor/positive/negative) 인코딩해야 해서
        # 2열(32에서 OOM 안 났음)보다 훨씬 무거움 — 기본을 절반으로 낮춤
        batch_size = 16 if has_hard_negative else 32
    epochs = args.epochs
    if epochs is None:
        # 2단계는 이미 적응된 모델 위에 이어서 학습 — epoch을 많이 돌리면 과적합/드리프트 위험
        epochs = 1 if has_hard_negative else 3
    lr = args.lr
    if lr is None:
        # 2단계는 처음부터 학습이 아니라 이어서 학습이라 더 작은 학습률로 조심스럽게
        lr = 5e-6 if has_hard_negative else 2e-5

    print(f"베이스 모델 로딩: {base_model} (epochs={epochs}, batch_size={batch_size}, lr={lr})")
    model = SentenceTransformer(base_model, device=DEVICE)

    loss = MultipleNegativesRankingLoss(model)

    training_args = SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_checkpointing=True,  # 활성화 메모리를 재계산으로 아껴서 OOM 여유 확보 (속도는 약간 느려짐)
        learning_rate=lr,
        warmup_ratio=0.1,
        fp16=(DEVICE == "cuda"),
        # in-batch negative 방식이라 마지막에 크기 안 맞는 배치가 negative 수를 흔들지 않게 버림
        dataloader_drop_last=True,
        # epoch마다 체크포인트(모델+optimizer 상태, 수GB씩) 저장하면 디스크 금방 찬다 —
        # 재개(resume) 안 할 거라 중간 체크포인트는 끄고 아래 model.save()로 최종본만 남김
        save_strategy="no",
        logging_steps=50,
        report_to="none",  # wandb/tensorboard 등 외부 로깅 비활성화 (설치 안 돼있어 원래도 안 붙지만 명시)
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        loss=loss,
    )
    trainer.train()

    model.save(output_dir)
    print(f"파인튜닝 완료, 저장 위치: {output_dir}")


if __name__ == "__main__":
    main()
