from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)
GOOD = {"question": "How many Cefalix 500 tablets can an adult take in one day?"}


def test_identify_ok():
    r = client.post("/identify", json=GOOD)
    assert r.status_code == 200
    body = r.json()
    assert body["provider"] == "mock" and body["sufficient"] is True
    assert body["sources"] and body["candidates"]


def test_identify_unknown_topic_is_refused_not_an_error():
    r = client.post("/identify", json={"question": "What is the best restaurant near the pharmacy?"})
    assert r.status_code == 200
    assert r.json()["sufficient"] is False and r.json()["requires_review"] is True


def test_identify_validation_errors_422():
    assert client.post("/identify", json={"question": "ab"}).status_code == 422
    assert client.post("/identify", json={}).status_code == 422


def test_identify_provider_down_503(monkeypatch):
    class Down:
        name, model = "down", "down"

        def chat(self, *args, **kwargs):
            raise RuntimeError("simulated outage")

    monkeypatch.setattr("app.generation.get_provider", lambda mode: Down())
    r = client.post("/identify", json=GOOD)
    assert r.status_code == 503
    assert r.json()["detail"] == "Inference service unavailable"
