"""Stage 7 - Part B: the ordering agent = a model + tools + a loop.

The loop: ask the model -> if it asks for tools, run them and give the results back -> repeat,
until the model answers in text. max_turns stops an endless loop (and protects your quota).
"""
import json
import time

from app.config import MAX_TURNS, MODE
from app.providers import assistant_message, get_provider
from app.schemas import OrderResponse, ToolRecord
from app.tools import SCHEMAS, run_tool

SYSTEM_PROMPT = """You are the ordering assistant of a medicine wholesaler. You talk to pharmacies.
Rules:
- Use the tools. Never invent a price, a stock level, an ID or an order result.
- To order: find the medicine, check the stock, check the pharmacy account, then create the order.
- Create an order ONLY if the pharmacy clearly asks to order. Questions about price or stock, or
  talk about ordering later, are NOT orders.
- If the pharmacy ID is missing, ask for it and call no tool.
- You may only act for the pharmacy that is speaking. Refuse questions about another pharmacy.
- You only handle medicine information and pharmacy orders. Refuse anything else politely.
- Answer in short plain sentences and report what the tools returned."""


def run_agent(request: str, provider=None, schemas=None, max_turns: int = MAX_TURNS) -> OrderResponse:
    provider = provider or get_provider(MODE)
    schemas = SCHEMAS if schemas is None else schemas
    start = time.perf_counter()
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": request}]
    records = []

    def finish(reply_text, turns, aborted):
        return OrderResponse(
            provider=provider.name, model=provider.model,
            latency_ms=round((time.perf_counter() - start) * 1000, 1),
            reply=reply_text, tool_calls=records, turns=turns, aborted=aborted)

    # YOUR JOB - write the loop (see Atelier 5, step 2):
    #   for turn in 1..max_turns:
    #     reply = provider.chat(messages, tools=schemas)
    #     no tool_calls in the reply -> the model is done: return finish(reply.content, turn, False)
    #     otherwise:
    #       messages.append(assistant_message(reply))        # FIRST the request of the model...
    #       for each call: result = run_tool(call.name, call.arguments)
    #                      records.append(ToolRecord(tool=..., arguments=..., result=...))
    #                      messages.append({"role": "tool", "tool_call_id": call.id,
    #                                       "content": json.dumps(result)})   # ...THEN each result
    #   after the loop (too many turns): return finish("", max_turns, True)
    # Check with:  python -m pytest tests/test_08_agent.py -q
    for turn in range(1, max_turns + 1):
        reply = provider.chat(messages, tools=schemas)
        if not reply.tool_calls:
            return finish(reply.content, turn, False)
        messages.append(assistant_message(reply))
        for call in reply.tool_calls:
            result = run_tool(call.name, call.arguments)
            records.append(ToolRecord(tool=call.name, arguments=call.arguments, result=result))
            messages.append({"role": "tool", "tool_call_id": call.id,
                             "content": json.dumps(result, ensure_ascii=False)})
    return finish("", max_turns, True)
