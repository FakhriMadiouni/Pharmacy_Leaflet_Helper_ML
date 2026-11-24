"""Stage 3 - the model behind our API, in two versions with the SAME interface.

    provider.chat(messages, tools=None, json_mode=False)  ->  LLMReply

  - MockProvider (app/mock_llm.py, PROVIDED): simple rules, no network, no key.
  - GroqProvider (YOU write it): calls Groq like in Atelier 2 and 5.

The rest of the app never knows which one it uses.
"""
import json
import time

from app.config import GROQ_API_KEY, LLM_MODEL
from app.mock_llm import MockProvider
from app.schemas import LLMReply, ToolCall

TOKENS_PER_MINUTE = 8000  # free Groq limit: we pause after each call to stay under it


def assistant_message(reply: LLMReply) -> dict:
    """PROVIDED. Puts the model's reply back into the history, in the format Groq expects.
    Needed by the agent loop: the tool results must come AFTER the request that asked for them."""
    message = {"role": "assistant", "content": reply.content}
    if reply.tool_calls:
        message["tool_calls"] = [
            {"id": c.id, "type": "function",
             "function": {"name": c.name, "arguments": json.dumps(c.arguments)}}
            for c in reply.tool_calls
        ]
    return message


class GroqProvider:
    name = "groq"

    def __init__(self, client=None, model=None):
        self.model = model or LLM_MODEL
        if client is None:
            if not GROQ_API_KEY:
                raise RuntimeError("GROQ_API_KEY is missing")
            from groq import Groq

            client = Groq(api_key=GROQ_API_KEY)
        self.client = client

    def chat(self, messages, tools=None, json_mode=False) -> LLMReply:
        """Send the messages to the model and return an LLMReply.

        YOUR JOB:
          1. call self.client.chat.completions.create(model=..., messages=..., temperature=0.0)
             and add tools=tools if tools, response_format={"type": "json_object"} if json_mode
          2. if Groq says RateLimitError, wait 20 s and try again (max 5 attempts, then raise RuntimeError)
          3. msg = response.choices[0].message ; its tool requests are in msg.tool_calls
             (each one: call.id, call.function.name, call.function.arguments = a JSON STRING)
             -> convert them to ToolCall objects; if the JSON string is broken use {} as arguments
          4. tokens = response.usage.total_tokens ; pause tokens / TOKENS_PER_MINUTE * 60 seconds
          5. return LLMReply(content=msg.content or "", tool_calls=..., model=self.model, tokens=tokens)

        Check with:  python -m pytest tests/test_02_api_health.py -q   (uses a fake client)
        """
        from groq import BadRequestError, RateLimitError

        kwargs = {"model": self.model, "messages": messages, "temperature": 0.0}
        if tools:
            kwargs["tools"] = tools
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        for _ in range(5):
            try:
                response = self.client.chat.completions.create(**kwargs)
                break
            except RateLimitError:
                time.sleep(20)
            except BadRequestError as exc:
                if "tool_use_failed" not in str(exc):
                    raise
                # the model wrote a broken tool call (typo in a parameter name...): ask again
        else:
            raise RuntimeError("Groq: too many failed attempts (rate limit or broken tool calls)")

        msg = response.choices[0].message
        calls = []
        for call in msg.tool_calls or []:
            try:
                arguments = json.loads(call.function.arguments)
            except (json.JSONDecodeError, TypeError):
                arguments = {}
            calls.append(ToolCall(id=call.id, name=call.function.name, arguments=arguments))

        tokens = response.usage.total_tokens if response.usage else 0
        time.sleep(tokens / TOKENS_PER_MINUTE * 60)
        return LLMReply(content=msg.content or "", tool_calls=calls, model=self.model, tokens=tokens)


def get_provider(mode: str, model: str | None = None):
    """mode "real" -> GroqProvider, anything else -> MockProvider.
    If the real provider cannot start (no key, no groq package) we fall back to the mock,
    exactly like get_provider in Atelier 1."""
    if mode == "real":
        try:
            return GroqProvider(model=model)
        except Exception:
            return MockProvider()
    return MockProvider()
