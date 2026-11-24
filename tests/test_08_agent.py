import copy
import json

import pytest

from app.agent import run_agent
from app.schemas import LLMReply, ToolCall
from app.tools import SCHEMAS, STATE, reset_state


@pytest.fixture(autouse=True)
def fresh_state():
    reset_state()


class Scripted:
    """Returns the given replies one by one and keeps a copy of the messages it received."""
    name, model = "fake", "fake-1"

    def __init__(self, replies):
        self.replies, self.seen, self.tools_seen = list(replies), [], []

    def chat(self, messages, tools=None, json_mode=False):
        self.seen.append(copy.deepcopy(messages))
        self.tools_seen.append(tools)
        return self.replies.pop(0)


def ask(*calls):
    return LLMReply(model="fake-1", tool_calls=[ToolCall(id=i, name=n, arguments=a) for i, n, a in calls])


def say(text):
    return LLMReply(model="fake-1", content=text)


def test_text_only_reply_ends_in_one_turn():
    r = run_agent("hello there", provider=Scripted([say("hi")]))
    assert r.reply == "hi" and r.turns == 1 and r.aborted is False and r.tool_calls == []


def test_tool_result_goes_back_to_the_model_with_its_id():
    fake = Scripted([ask(("c1", "check_stock", {"medicine_id": "M05", "quantity": 3})), say("done")])
    r = run_agent("stock of M05?", provider=fake)
    assert r.turns == 2 and r.reply == "done"
    assert r.tool_calls[0].tool == "check_stock" and r.tool_calls[0].result["stock"] == 8
    second_call = fake.seen[1]
    assistant = second_call[-2]
    tool = second_call[-1]
    assert assistant["role"] == "assistant" and assistant["tool_calls"][0]["id"] == "c1"   # request first
    assert tool["role"] == "tool" and tool["tool_call_id"] == "c1"                          # then its result
    assert json.loads(tool["content"])["stock"] == 8


def test_two_tools_in_one_turn():
    fake = Scripted([ask(("a", "get_pharmacy", {"pharmacy_id": "PH001"}),
                         ("b", "search_medicine", {"name": "Cefalix"})), say("ok")])
    r = run_agent("x y z", provider=fake)
    assert [t.tool for t in r.tool_calls] == ["get_pharmacy", "search_medicine"]
    assert [m["tool_call_id"] for m in fake.seen[1] if m["role"] == "tool"] == ["a", "b"]


def test_endless_loop_is_stopped_by_max_turns():
    endless = Scripted([ask((f"c{i}", "search_medicine", {"name": "x"})) for i in range(10)])
    r = run_agent("loop forever", provider=endless, max_turns=3)
    assert r.aborted is True and r.turns == 3 and len(r.tool_calls) == 3 and r.reply == ""
    assert len(endless.seen) == 3  # the model was called 3 times, not 10


def test_model_mistakes_do_not_crash_the_loop():
    fake = Scripted([ask(("c1", "create_order", {"pharmacy_id": "PH001"}), ("c2", "unknown_tool", {})),
                     say("sorry")])
    r = run_agent("order please", provider=fake)
    assert "error" in r.tool_calls[0].result and "error" in r.tool_calls[1].result
    assert STATE["orders"] == []


def test_the_tool_list_is_given_to_the_model():
    fake = Scripted([say("hi")])
    run_agent("hello there", provider=fake)
    assert fake.tools_seen[0] == SCHEMAS


# ---- with the mock model: the whole behaviour
def test_order_creates_exactly_what_was_asked():
    r = run_agent("Pharmacy PH001 here. Please order 20 boxes of Cefalix 500.")
    assert "confirmed" in r.reply.lower()
    assert len(STATE["orders"]) == 1 and STATE["orders"][0]["quantity"] == 20


def test_questions_never_create_orders():
    for text in ("PH001: how much does Gastrovel 20 cost?",
                 "PH004: do you have Allerzin 10 in stock?",
                 "PH001: we are thinking about ordering Cefalix 500 next week. Do you have it in stock?"):
        reset_state()
        r = run_agent(text)
        assert "create_order" not in [t.tool for t in r.tool_calls]
        assert STATE["orders"] == []


def test_no_tool_when_nothing_should_be_called():
    for text in ("What is the weather like in Tunis today?",
                 "I want to order 10 boxes of Cefalix 500.",
                 "Hello, PH001 here. Is PH002 blocked or not?"):
        assert run_agent(text).tool_calls == []


def test_rules_in_code_protect_even_when_the_model_asks_for_a_bad_order():
    bad = Scripted([ask(("c1", "create_order",
                         {"pharmacy_id": "PH003", "medicine_id": "M01", "quantity": 5})), say("ok")])
    r = run_agent("PH003: order 5 boxes of Cefalix 500", provider=bad)
    assert r.tool_calls[0].result["status"] == "rejected"
    assert STATE["orders"] == []
