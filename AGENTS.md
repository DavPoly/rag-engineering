# Project instructions

## What this is

Week 1 of a 5-week AI Engineering interview prep project: a policy Q&A
service with cited, honest answers over Northwind Goods' policy documents.
See README.md for architecture and DECISIONS.md for the reasoning trail.

## Stack

- Python 3.11+, managed with `uv`
- FastAPI + Pydantic v2 for the service layer
- pytest for tests (mocked LLM/embedding calls in unit tests)
- rank-bm25 for keyword retrieval, sentence-transformers for dense retrieval
- OpenRouter (via the `openai` client, pointed at a different base_url) for LLM calls
- ruff for lint/format

## Conventions

- Small, typed functions. Type hints everywhere in `src/`.
- Config only through `policy_answer_service.config.get_settings()` —
  never read `os.environ` directly in application code.
- Secrets only via environment variables / `.env` (git-ignored). Never put
  API keys in code, prompts, or commits.
- One module per pipeline stage: `ingestion.py`, `retrieval.py`,
  `generation.py`, `api.py`, `models.py`. Keep them that way rather than
  merging things together as the project grows.
- Mock LLM and embedding-model calls in unit tests (`tests/`). Real calls
  belong in `evals/` scripts, run deliberately, with cost tracked.
- Request/response models are `UserRequest` / `ModelResponse` in `models.py`.

## Environment notes

- Windows with an Application Control policy that blocks executables in
  `.venv\Scripts\`. **Always invoke tools as `uv run python -m <tool>`**,
  never `uv run <tool>` directly (e.g. `uv run python -m pytest`, not
  `uv run pytest`).

## Test commands

```text
uv run python -m pytest                 # all unit tests
uv run python -m pytest -k ingestion    # one module
uv run python -m pytest --cov=src       # with coverage
uv run python -m ruff check .           # lint
uv run python -m ruff format .          # format
uv run python -m uvicorn policy_answer_service.api:app --reload   # run the service
```

## Working agreement

- Ask before adding a new dependency — justify it over what's already here.
- Keep changes small and reviewable: one function or module per change,
  not sweeping diffs.
- Run the tests after every change and show me the output.
- Simplest version first. Don't add a vector DB, reranker, or framework
  until a measured eval result on the golden set says the simpler version
  isn't enough — and then log that in DECISIONS.md.
- When something in this pipeline could plausibly go wrong (empty
  retrieval, conflicting policy versions, malformed input), write the
  failure case as a test before writing the fix.