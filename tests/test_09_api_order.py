import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.tools import STATE, reset_state

client = TestClient(app)


@pytest.fixture(autouse=True)
def fresh_state():
    reset_state()


def test_order_ok():
    r = client.post("/order", json={"request": "PH002 here. Order 5 boxes of Sleepora 5."})
    assert r.status_code == 200
    body = r.json()
    assert body["provider"] == "mock" and body["aborted"] is False
    assert "confirmed" in body["reply"].lower()
    assert body["tool_calls"][-1]["tool"] == "create_order"
    assert len(STATE["orders"]) == 1


def test_order_refused_by_the_rules():
    r = client.post("/order", json={"request": "PH001: order 5 boxes of Sleepora 5."})
    assert r.status_code == 200
    assert "authorization" in r.json()["reply"].lower()
    assert STATE["orders"] == []


def test_order_validation_errors_422():
    assert client.post("/order", json={"request": "ab"}).status_code == 422
    assert client.post("/order", json={}).status_code == 422


def test_order_provider_down_503(monkeypatch):
    class Down:
        name, model = "down", "down"

        def chat(self, *args, **kwargs):
            raise RuntimeError("simulated outage")

    monkeypatch.setattr("app.agent.get_provider", lambda mode: Down())
    r = client.post("/order", json={"request": "PH001: how much does Gastrovel 20 cost?"})
    assert r.status_code == 503
    assert r.json()["detail"] == "Inference service unavailable"
