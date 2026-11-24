import json

from app.generation import REFUSAL, answer_question
from app.schemas import LLMReply

GOOD_QUESTION = "How many Cefalix 500 tablets can an adult take in one day?"


class Scripted:
    """A fake model that returns a fixed text and counts how often it is called."""
    name, model = "fake", "fake-1"

    def __init__(self, content):
        self.content, self.calls = content, 0

    def chat(self, messages, tools=None, json_mode=False):
        self.calls += 1
        return LLMReply(content=self.content, model=self.model)


def reply(**data):
    return json.dumps(data)


def test_below_threshold_refuses_without_calling_the_model():
    fake = Scripted(reply(answer="x", sources=[], sufficient=True))
    r = answer_question("What is the best restaurant near the pharmacy?", provider=fake, threshold=0.2)
    assert fake.calls == 0
    assert r.sufficient is False and r.requires_review is True and r.answer == REFUSAL


def test_good_answer_with_valid_source():
    fake = Scripted(reply(answer="Max 4 tablets a day.", sources=["M01-dosage"], sufficient=True))
    r = answer_question(GOOD_QUESTION, provider=fake, threshold=0.2)
    assert fake.calls == 1
    assert r.sufficient and r.sources == ["M01-dosage"] and r.requires_review is False
    assert r.candidates[0].medicine_id == "M01"
    assert r.provider == "fake" and r.latency_ms >= 0


def test_invented_source_is_dropped_and_flagged():
    fake = Scripted(reply(answer="ok", sources=["M01-dosage", "M99-uses"], sufficient=True))
    r = answer_question(GOOD_QUESTION, provider=fake, threshold=0.2)
    assert r.sources == ["M01-dosage"]
    assert r.requires_review is True


def test_model_says_insufficient():
    fake = Scripted(reply(answer="?", sources=[], sufficient=False))
    r = answer_question(GOOD_QUESTION, provider=fake, threshold=0.2)
    assert r.sufficient is False and r.answer == REFUSAL and r.requires_review is True


def test_broken_json_is_a_refusal_not_a_crash():
    r = answer_question(GOOD_QUESTION, provider=Scripted("not json at all"), threshold=0.2)
    assert r.sufficient is False and r.requires_review is True


def test_json_that_is_not_an_object():
    r = answer_question(GOOD_QUESTION, provider=Scripted("[1, 2]"), threshold=0.2)
    assert r.sufficient is False


def test_answer_without_any_source_needs_review():
    fake = Scripted(reply(answer="ok", sources=[], sufficient=True))
    r = answer_question(GOOD_QUESTION, provider=fake, threshold=0.2)
    assert r.requires_review is True
