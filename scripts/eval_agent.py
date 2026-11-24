"""Stage 7 and 8 - measure the ordering agent on the 24 tasks of data/tasks_orders.csv.

Run:  python scripts/eval_agent.py                          default model, precise descriptions
      python scripts/eval_agent.py openai/gpt-oss-20b       another model (the ONE variable changed)
      python scripts/eval_agent.py qwen/qwen3.8-27b vague   same model, vague tool descriptions
In real mode this takes several minutes (pauses to respect the free quota): do not run two at once.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.agent import run_agent  # noqa: E402
from app.config import LLM_MODEL, MODE  # noqa: E402
from app.providers import get_provider  # noqa: E402
from app.tools import SCHEMAS, make_vague, reset_state  # noqa: E402
from scripts.scoring import facts_ok, read_csv, save_results, tools_ok, unrequested_orders  # noqa: E402

args = [a for a in sys.argv[1:] if a != "vague"]
vague = "vague" in sys.argv[1:]
model = args[0] if args else LLM_MODEL
provider = get_provider(MODE, model)
schemas = make_vague(SCHEMAS) if vague else SCHEMAS

tasks = read_csv("tasks_orders.csv")
items = []
for task in tasks:
    reset_state()  # every task starts with the same stock and an empty order book
    order_requested = task["order_requested"] == "1"
    try:
        r = run_agent(task["request"], provider=provider, schemas=schemas)
        called = [t.tool for t in r.tool_calls]
        item = {"reply": r.reply, "called": called, "turns": r.turns, "aborted": r.aborted,
                "latency_ms": r.latency_ms, "error": ""}
    except Exception as exc:  # a crash counts as a failed task, the run goes on
        called = []
        item = {"reply": "", "called": [], "turns": 0, "aborted": False, "latency_ms": 0, "error": str(exc)}
    item.update({"id": task["id"], "required": task["required_tools"],
                 "tools_ok": tools_ok(task["required_tools"], called, order_requested),
                 "facts_ok": facts_ok(item["reply"], task["expected_facts"]),
                 "unrequested_orders": unrequested_orders(called, order_requested)})
    item["success"] = item["tools_ok"] and item["facts_ok"]
    items.append(item)

n = len(items)
summary = {"provider": provider.name, "model": provider.model, "descriptions": "vague" if vague else "precise",
           "tasks": n, "success": sum(i["success"] for i in items),
           "tools_ok": sum(i["tools_ok"] for i in items), "facts_ok": sum(i["facts_ok"] for i in items),
           "unrequested_orders": sum(i["unrequested_orders"] for i in items),
           "aborted": sum(i["aborted"] for i in items),
           "calls_per_task": round(sum(len(i["called"]) for i in items) / n, 1)}
label = f"{provider.model.split('/')[-1]}__{summary['descriptions']}"
print(f"=== agent {label} ===")
print(f"tasks succeeded     : {summary['success']}/{n}")
print(f"right tools         : {summary['tools_ok']}/{n}")
print(f"right answers       : {summary['facts_ok']}/{n}")
print(f"tool calls per task : {summary['calls_per_task']}")
print(f"unrequested orders  : {summary['unrequested_orders']}     aborted (too many turns): {summary['aborted']}")
print("Failed tasks:")
for task, i in zip(tasks, items):
    if not i["success"]:
        print(f"  {i['id']} tools_ok={i['tools_ok']} facts_ok={i['facts_ok']} required=[{i['required']}] "
              f"called={i['called']} {i['error']}")
save_results(label, "agent", summary, items)
