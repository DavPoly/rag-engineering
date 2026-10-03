"""Smoke test — confirms the FastAPI app boots and the health check responds.
Keep this passing at all times; it's your fastest signal that nothing is broken."""

from fastapi.testclient import TestClient

from policy_answer_service.api import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}