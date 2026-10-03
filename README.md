# Policy Answer Service

A fictional e-commerce company, **Northwind Goods**, gets 2,000 support tickets a month.
Agents waste time searching 20 policy documents by hand. This service
answers policy questions with cited sources, so a human can trust and verify
every answer in seconds.

## Problem

- 20 policy documents (returns, shipping, damaged items, warranty, loyalty,
  payments), including two outdated/conflicting versions and one doc dense
  with product codes.
- Support agents need fast, cited, honest answers — including an explicit
  "not in the policies" response when nothing relevant is found.

## Architecture

Question → Retrieval ladder (BM25 → vector → hybrid+RRF → +reranker)
         → top-k chunks (with metadata: title, section, effective_date, version)
         → Generation (cited answer + confidence flag)
         → FastAPI /ask endpoint
<!-- TODO: replace with an actual diagram once the pipeline is built. -->

## Results

<!-- TODO: fill in after the retrieval ladder and generation are built. -->

| Method | recall@5 | MRR | Latency (p50) | Cost / query |
| --- | --- | --- | --- | --- |
| BM25 only | | | | |
| Dense vector | | | | |
| Hybrid (RRF) | | | | |
| Hybrid + reranker | | | | |

**Faithfulness on 30-question golden set:** TODO

## How to run

```bash
uv sync
uv run python -m pytest
uv run python -m uvicorn policy_answer_service.api:app --reload
```

Then: `curl -X POST localhost:8000/ask -H "Content-Type: application/json" -d '{"question": "..."}'`

## Project layout

## Project layout

```text
src/policy_answer_service/   # application code
tests/                       # unit tests (mirror the src/ modules)
data/                        # synthetic policy docs, embedding cache
evals/                       # golden set, eval scripts, results
DECISIONS.md                 # what was chosen, what was rejected, why
```

## Decisions

See [DECISIONS.md](DECISIONS.md) for the reasoning behind the retrieval
stopping point, the chunking strategy, and how version conflicts are
handled.

## What I'd do next

<!-- TODO: fill in during packaging -->