"""
End-to-end eval: vector retrieval -> cited generation -> faithfulness and correctness judges.

Runs in parts so it stays under the OpenRouter limit of 20 requests per minute:
  - requests are spaced at least MIN_INTERVAL_S apart
  - each run processes at most --batch-size questions, appending each row to
    evals/results/faithfulness.jsonl as it finishes; each row records the retrieved
    chunk IDs and whether retrieval hit the expected chunks
  - questions already in that file are skipped, so rerun until all 30 are done

Each answered question costs three calls (generation, faithfulness judge, correctness
judge), plus one embedding batch when the question embeddings are first computed.
When everything is done, a seeded sample of 10 answered rows is written to
evals/results/spot_check_<run-name>.jsonl for manual review of the judge.
Use --run-name to start a new run; the default name resumes the existing run.
"""

import argparse
import json
import random
import time
from pathlib import Path

from policy_answer_service.config import get_settings
from policy_answer_service.generation import (
    build_client,
    generate_answer,
    score_correctness,
    score_faithfulness,
)
from policy_answer_service.retrieval import VectorRetriever, load_chunks

MIN_INTERVAL_S = 3.5  # 60 / 3.5 ~= 17 requests per minute, leaving headroom under the 20 limit
SPOT_CHECK_SIZE = 10
SPOT_CHECK_SEED = 0

_last_request_at = 0.0


def pace() -> None:
    """Sleep so consecutive API requests are at least MIN_INTERVAL_S apart."""
    global _last_request_at
    wait = MIN_INTERVAL_S - (time.monotonic() - _last_request_at)
    if wait > 0:
        time.sleep(wait)
    _last_request_at = time.monotonic()


def load_done(results_path: Path) -> dict[str, dict]:
    if not results_path.exists():
        return {}
    with open(results_path) as f:
        return {row["id"]: row for row in map(json.loads, f)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--batch-size", type=int, default=4, help="questions to process this run")
    parser.add_argument(
        "--run-name",
        default="faithfulness",
        help="name for this run's output files; use a new name to start a fresh run",
    )
    args = parser.parse_args()

    settings = get_settings()
    chunks = load_chunks(settings.data_dir / "chunks.jsonl")
    golden_set = [json.loads(line) for line in open(settings.golden_set_path)]
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    results_path = results_dir / f"{args.run_name}.jsonl"
    spot_check_path = results_dir / f"spot_check_{args.run_name}.jsonl"

    done = load_done(results_path)
    pending = [item for item in golden_set if item["id"] not in done][: args.batch_size]
    print(f"done: {len(done)}/{len(golden_set)}  this run: {len(pending)}")

    if pending:
        pace()
        vector = VectorRetriever(chunks, settings.embedding_model_api)
        vector.embed_queries([item["question"] for item in pending])
        client = build_client()

        with open(results_path, "a") as out:
            for item in pending:
                retrieved = vector.retrieve(item["question"], top_k=settings.retrieval_top_k)
                pace()
                response = generate_answer(client, settings.llm_model, item["question"], retrieved)
                cited_ids = {c.chunk_id for c in response.citations}
                cited_chunks = [c for c in retrieved if c["chunk_id"] in cited_ids]
                retrieved_ids = [c["chunk_id"] for c in retrieved]

                row = {
                    "id": item["id"],
                    "category": item["category"],
                    "question": item["question"],
                    "answer": response.answer,
                    "answerable": response.answerable,
                    "confidence": response.confidence,
                    "citations": sorted(cited_ids),
                    "expected_chunks": item["expected_chunks"],
                    "retrieved": retrieved_ids,
                    "retrieval_hit": bool(set(item["expected_chunks"]) & set(retrieved_ids)),
                    "excerpts": [
                        {"chunk_id": c["chunk_id"], "text": c["text"]} for c in cited_chunks
                    ],
                    "faithfulness": None,
                    "correctness": None,
                }
                if response.answerable:
                    pace()
                    row["faithfulness"] = score_faithfulness(
                        client, settings.llm_model, item["question"], response.answer, cited_chunks
                    )
                    pace()
                    row["correctness"] = score_correctness(
                        client, settings.llm_model, item["question"], response.answer, item["notes"]
                    )
                out.write(json.dumps(row) + "\n")
                out.flush()
                done[item["id"]] = row
                print(
                    f"  {item['id']} answerable={row['answerable']} "
                    f"faithfulness={row['faithfulness']} correctness={row['correctness']}"
                )

    if len(done) < len(golden_set):
        print(f"{len(golden_set) - len(done)} questions remaining; rerun to continue.")
        return

    rows = [done[item["id"]] for item in golden_set]
    answered = [r for r in rows if r["answerable"]]
    unanswerable = [r for r in rows if r["category"] == "unanswerable"]
    spot_check = random.Random(SPOT_CHECK_SEED).sample(
        answered, min(SPOT_CHECK_SIZE, len(answered))
    )
    with open(spot_check_path, "w") as f:
        for row in spot_check:
            f.write(json.dumps(row) + "\n")

    refusals_correct = sum(not r["answerable"] for r in unanswerable)
    refused_answerable = [
        r for r in rows if r["category"] != "unanswerable" and not r["answerable"]
    ]
    refused_with_hit = sum(r["retrieval_hit"] for r in refused_answerable)
    mean_faithfulness = (
        sum(r["faithfulness"] for r in answered) / len(answered) if answered else 0.0
    )
    mean_correctness = sum(r["correctness"] for r in answered) / len(answered) if answered else 0.0
    print(f"refusal accuracy on unanswerable: {refusals_correct}/{len(unanswerable)}")
    refused_missed = len(refused_answerable) - refused_with_hit
    print(
        f"answerable questions refused: {len(refused_answerable)} "
        f"(retrieval hit on {refused_with_hit}; missed on {refused_missed})"
    )
    print(f"answered: {len(answered)}/{len(rows)}")
    print(f"mean faithfulness: {mean_faithfulness:.2f}  mean correctness: {mean_correctness:.2f}")
    print(f"spot-check sample ({len(spot_check)} rows): {spot_check_path}")


if __name__ == "__main__":
    main()
