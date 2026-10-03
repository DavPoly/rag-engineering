"""
FastAPI service layer. Wires ingestion -> retrieval -> generation behind POST /ask.
"""

from fastapi import FastAPI

app = FastAPI(title="Policy Answer Service")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}