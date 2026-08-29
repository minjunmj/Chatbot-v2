"""
"top-5 판례 원문 전체" vs "top-5 문서당 매칭 chunk+앞뒤" 두 컨텍스트 방식으로 각각 답변을
생성한 뒤, GPT-5-mini를 judge로 써서 비교한다 (docs/log.md 2026-08-25 논의 참고).

judge는 답변만 보고 채점하지 않고 판례 원문 전체(레퍼런스)를 같이 줘서, chunk 버전 답변이
실제로 원문에 근거했는지도 판단할 수 있게 한다. 위치 편향(순서 선호) 방지를 위해 A/B
순서를 매 쿼리마다 무작위로 섞는다 — 길이 편향(길고 자세하면 무조건 우세 판정하는 경향)에
대한 주의사항도 judge 프롬프트에 명시한다.

사용법:
    python judge_compare.py                # val_query.json 0번째 쿼리
    python judge_compare.py --index 5
    OPENAI_API_KEY는 환경변수 또는 server/../.env에서 읽음 (OPENAI_MODEL로 judge 모델 교체 가능)
"""
import argparse
import json
import os
import random

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from generate import (
    VAL_PATH, TOP_K,
    load_models, retrieve_top_k, load_full_doc,
    build_context_full, build_context_chunks, generate_answer,
)

JUDGE_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5-mini")

JUDGE_SYSTEM = (
    "당신은 한국 법률 RAG 챗봇의 답변 품질을 비교 평가하는 심사위원입니다. "
    "동일한 질문에 대해 서로 다른 방식으로 생성된 두 답변(A, B)을 아래 기준으로 비교하세요.\n\n"
    "기준:\n"
    "1. 정확성: 제공된 판례 원문(레퍼런스)과 비교했을 때 내용이 사실과 부합하는가\n"
    "2. Groundedness: 레퍼런스에 없는 내용을 지어내지 않았는가(할루시네이션 없음)\n"
    "3. Citation 정확성: 인용한 [사건번호]가 실제로 그 주장을 뒷받침하는 판례가 맞는가\n\n"
    "주의: 답변이 더 길거나 자세하다는 이유만으로 우세하다고 판단하지 마세요 — 근거 없이 "
    "장황하기만 한 답변은 오히려 감점해야 합니다. 반드시 아래 JSON 형식으로만 답하세요:\n"
    '{"more_accurate": "A|B|tie", "more_grounded": "A|B|tie", '
    '"more_correct_citation": "A|B|tie", "overall_better": "A|B|tie", "reasoning": "..."}'
)


def build_judge_user_message(query, reference_docs, answer_a, answer_b):
    ref_text = "\n\n---\n\n".join(
        f"[사건번호: {d['사건번호']}] {d['사건명']} ({d['법원명']}, {d['선고일자']})\n{d['판례내용']}"
        for d in reference_docs
    )
    return (
        f"=== 질문 ===\n{query}\n\n"
        f"=== 레퍼런스 판례 원문(전체) ===\n{ref_text}\n\n"
        f"=== 답변 A ===\n{answer_a}\n\n"
        f"=== 답변 B ===\n{answer_b}"
    )


def call_judge(query, reference_docs, answer_a, answer_b):
    from openai import OpenAI
    client = OpenAI()  # OPENAI_API_KEY 환경변수 사용
    resp = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": build_judge_user_message(query, reference_docs, answer_a, answer_b)},
        ],
    )
    raw = resp.choices[0].message.content
    try:
        return json.loads(raw), raw
    except json.JSONDecodeError:
        return None, raw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, default=0, help="val_query.json에서 몇 번째 쿼리를 쓸지")
    parser.add_argument("--query", type=str, default=None, help="직접 쿼리 입력(지정 시 --index 무시)")
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--seed", type=int, default=None, help="A/B 순서 섞을 때 쓸 시드(재현용)")
    args = parser.parse_args()

    if args.query:
        query = args.query
    else:
        with open(VAL_PATH, encoding="utf-8") as f:
            val_data = json.load(f)
        query = val_data[args.index]["query"]
        print(f"[val_query.json #{args.index}] 정답 case_ids: {val_data[args.index]['case_ids']}")

    print(f"쿼리: {query}")

    dense_model, retriever, tokenizer, llm, conn = load_models()

    print("검색+재정렬 중...")
    ranked = retrieve_top_k(retriever, query, args.top_k)
    print(f"top-{args.top_k} 판례: {ranked}")
    reference_docs = [load_full_doc(cn) for cn in ranked]

    print("컨텍스트 구성 중 (full)...")
    context_full = build_context_full(ranked)
    print("컨텍스트 구성 중 (chunks)...")
    context_chunks = build_context_chunks(dense_model, conn, ranked, query)

    print("답변 생성 중 (full)...")
    answer_full = generate_answer(tokenizer, llm, query, context_full)
    print("답변 생성 중 (chunks)...")
    answer_chunks = generate_answer(tokenizer, llm, query, context_chunks)

    # 위치 편향 방지: full/chunks 중 뭐가 A가 될지 매번 무작위로 섞음
    rng = random.Random(args.seed)
    if rng.random() < 0.5:
        label_map = {"A": "full", "B": "chunks"}
        answer_a, answer_b = answer_full, answer_chunks
    else:
        label_map = {"A": "chunks", "B": "full"}
        answer_a, answer_b = answer_chunks, answer_full
    print(f"(A={label_map['A']}, B={label_map['B']} — judge에겐 안 알려줌)")

    print("\n===== 답변(full) =====")
    print(answer_full)
    print("\n===== 답변(chunks) =====")
    print(answer_chunks)

    print("\nGPT judge 평가 중...")
    verdict, raw = call_judge(query, reference_docs, answer_a, answer_b)

    print("\n===== judge 원본 응답 =====")
    print(raw)

    if verdict:
        print("\n===== 최종 판정(A/B → full/chunks로 해석) =====")
        for key in ("more_accurate", "more_grounded", "more_correct_citation", "overall_better"):
            v = verdict.get(key)
            resolved = label_map.get(v, v)
            print(f"{key}: {v} -> {resolved}")
        print(f"reasoning: {verdict.get('reasoning')}")


if __name__ == "__main__":
    main()
