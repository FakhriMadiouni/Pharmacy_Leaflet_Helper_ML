import json
from types import SimpleNamespace

from fastapi.testclient import TestClient

from app.main import app
from app.providers import GroqProvider, get_provider
from app.mock_llm import MockProvider

client = TestClient(app)


def test_health():
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["mode"] == "mock"
    assert body["provider"] == "mock"


def test_get_provider_mock_and_fallback():
    assert isinstance(get_provider("mock"), MockProvider)
    # real mode without a key falls back to the mock instead of crashing
    assert isinstance(get_provider("real"), MockProvider)


# ---- GroqProvider, tested with a fake client (no network, no key)
class FakeClient:
    def __init__(self, message, tokens=0):
        self.calls = []
        self.message, self.tokens = message, tokens
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=self.message)],
                               usage=SimpleNamespace(total_tokens=self.tokens))


def make_call(arguments):
    return SimpleNamespace(id="call_1", function=SimpleNamespace(name="check_stock", arguments=arguments))


def test_groq_provider_text_reply():
    fake = FakeClient(SimpleNamespace(content="hello", tool_calls=None))
    reply = GroqProvider(client=fake, model="m").chat([{"role": "user", "content": "hi"}])
    assert reply.content == "hello" and reply.tool_calls == [] and reply.model == "m"
    assert fake.calls[0]["temperature"] == 0.0
    assert "tools" not in fake.calls[0] and "response_format" not in fake.calls[0]


def test_groq_provider_tool_calls_and_json_mode():
    message = SimpleNamespace(content=None, tool_calls=[make_call(json.dumps({"medicine_id": "M01"}))])
    fake = FakeClient(message)
    reply = GroqProvider(client=fake, model="m").chat([], tools=[{"x": 1}], json_mode=True)
    assert reply.tool_calls[0].name == "check_stock"
    assert reply.tool_calls[0].arguments == {"medicine_id": "M01"}
    assert fake.calls[0]["tools"] == [{"x": 1}]
    assert fake.calls[0]["response_format"] == {"type": "json_object"}


def test_groq_provider_broken_arguments_become_empty_dict():
    fake = FakeClient(SimpleNamespace(content=None, tool_calls=[make_call("{not json")]))
    reply = GroqProvider(client=fake, model="m").chat([], tools=[{"x": 1}])
    assert reply.tool_calls[0].arguments == {}
