## Test commands

```bash
uv run python -m pytest                 # all unit tests
uv run python -m pytest -k ingestion    # one module
uv run python -m pytest --cov=src       # with coverage
uv run python -m ruff check .           # lint
uv run python -m ruff format .          # format
```