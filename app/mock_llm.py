"""The simulated model (PROVIDED, nothing to write here).

Like the MockProvider of Atelier 1, this is NOT a real model. It follows simple rules so that the
whole app can be built and tested offline. It gives the same type of reply as Groq:
  - with json_mode  -> a JSON text {"answer", "sources", "sufficient"}   (Part A)
  - with tools      -> tool requests, then a final text                  (Part B)
Real quality (and real mistakes) only appear with the real model: that is what you measure.
"""
import json
import re

from app.config import DATA
from app.embeddings import content_words
from app.schemas import LLMReply, ToolCall

PH_ID = re.compile(r"\bPH\d{3}\b")
ORDER_ITEM = re.compile(r"(\d+)\s+(?:boxes|box|tubes|bottles|packs|units|vials)\s+of\s+([A-Za-z]+)", re.I)
QTY = re.compile(r"(\d+)\s+(?:boxes|box|tubes|bottles|packs|units|vials)", re.I)
ORDER_WORDS = re.compile(r"\b(order|need|send|buy|wants?)\b", re.I)
HYPOTHETICAL = ("thinking about", "would it", "what if", "do you have", "how much")
LEAFLET_WORDS = ("side effect", "how many tablets", "dosage", "how often", "can i take",
                 "used for", "look like", "pregnan", "tell me about")


def _catalogue_names() -> list[str]:
    medicines = json.loads((DATA / "medicines.json").read_text(encoding="utf-8"))
    return [m["name"].split()[0].lower() for m in medicines]


class MockProvider:
    name = "mock"
    model = "mock-rules"

    def chat(self, messages, tools=None, json_mode=False) -> LLMReply:
        if tools:
            return self._agent(messages)
        return self._rag(messages)

    # ---------------------------------------------------------------- Part A
    def _rag(self, messages) -> LLMReply:
        user = messages[-1]["content"]
        passages = re.findall(r"^\[([A-Z0-9]+-[a-z_]+)\] (.+)$", user, re.M)
        question = user.split("Question:")[-1]
        words = set(content_words(question))
        known = set(content_words(" ".join(text for _, text in passages)))
        covered = len(words & known) / len(words) if words else 0
        if not passages or covered < 0.5:
            data = {"answer": "Insufficient information.", "sources": [], "sufficient": False}
        else:
            first_id, first_text = passages[0]
            data = {"answer": first_text, "sources": [first_id], "sufficient": True}
        return LLMReply(content=json.dumps(data), model=self.model)

    # ---------------------------------------------------------------- Part B
    def _agent(self, messages) -> LLMReply:
        request = next(m["content"] for m in messages if m["role"] == "user")
        history = self._history(messages)

        def result(tool_name, /, **args):
            return history.get((tool_name, json.dumps(args, sort_keys=True)))

        n_calls = sum(len(m.get("tool_calls", [])) for m in messages if m["role"] == "assistant")
        lower = request.lower()
        pharmacies = sorted(set(PH_ID.findall(request)))
        items = [(int(q), n) for q, n in ORDER_ITEM.findall(request)]
        known = [n for n in _catalogue_names() if n in lower]

        def ask(*calls):
            tool_calls = [ToolCall(id=f"call_{n_calls + i + 1}", name=n, arguments=a)
                          for i, (n, a) in enumerate(calls)]
            return LLMReply(tool_calls=tool_calls, model=self.model)

        def say(text):
            return LLMReply(content=text, model=self.model)

        if len(pharmacies) > 1:
            return say("I can only share information about your own pharmacy account, "
                       "so I cannot check another pharmacy.")

        is_order = bool(items) and ORDER_WORDS.search(request) and not any(h in lower for h in HYPOTHETICAL)
        if is_order:
            return self._order(items, pharmacies, result, ask, say)
        return self._info(request, lower, known, pharmacies, result, ask, say)

    def _order(self, items, pharmacies, result, ask, say):
        if not pharmacies:
            return say("Please tell me your pharmacy ID (for example PH001) so I can place the order.")
        ph = pharmacies[0]

        # turn 1: look up every medicine and the pharmacy
        todo = [("search_medicine", {"name": n}) for _, n in items if result("search_medicine", name=n) is None]
        if result("get_pharmacy", pharmacy_id=ph) is None:
            todo.append(("get_pharmacy", {"pharmacy_id": ph}))
        if todo:
            return ask(*todo)

        pharmacy = result("get_pharmacy", pharmacy_id=ph)
        if "error" in pharmacy:
            return say(f"Pharmacy {ph} was not found.")
        if pharmacy["status"] == "blocked":
            return say(f"Pharmacy {ph} is blocked, so I cannot place orders for it.")

        found, problems = [], []
        for qty, name in items:
            matches = result("search_medicine", name=name)["matches"]
            if not matches:
                problems.append(f"Medicine '{name}' was not found in our catalogue.")
                continue
            med = matches[0]
            if med["controlled"] and not pharmacy["special_authorization"]:
                problems.append(f"{med['name']} is a controlled medicine: this pharmacy has no special authorization.")
                continue
            found.append((qty, med))
        if problems:
            return say(" ".join(problems))

        # turn 2: stock
        todo = [("check_stock", {"medicine_id": m["id"], "quantity": q}) for q, m in found
                if result("check_stock", medicine_id=m["id"], quantity=q) is None]
        if todo:
            return ask(*todo)
        for qty, med in found:
            stock = result("check_stock", medicine_id=med["id"], quantity=qty)
            if "error" in stock:
                problems.append(stock["error"])
            elif stock["stock"] == 0:
                problems.append(f"{med['name']} is out of stock.")
            elif not stock["available"]:
                problems.append(f"Not enough stock for {med['name']}: only {stock['stock']} boxes available.")
        if problems:
            return say(" ".join(problems))

        # turn 3: create the orders (the pharmacy asked for them)
        todo = [("create_order", {"pharmacy_id": ph, "medicine_id": m["id"], "quantity": q}) for q, m in found
                if result("create_order", pharmacy_id=ph, medicine_id=m["id"], quantity=q) is None]
        if todo:
            return ask(*todo)
        lines = []
        for qty, med in found:
            order = result("create_order", pharmacy_id=ph, medicine_id=med["id"], quantity=qty)
            if order["status"] == "confirmed":
                lines.append(f"Order confirmed: {qty} boxes of {med['name']} (order {order['order_id']}).")
            elif order["status"] == "pending_review":
                lines.append(f"The order of {qty} boxes of {med['name']} needs review by our team before confirmation.")
            else:
                lines.append(f"Order rejected for {med['name']}: {order['reason']}")
        return say(" ".join(lines))

    def _info(self, request, lower, known, pharmacies, result, ask, say):
        wants_price = any(w in lower for w in ("cost", "price", "how much"))
        wants_stock = "stock" in lower or "available" in lower
        wants_controlled = "controlled" in lower
        wants_account = bool(pharmacies) and "account" in lower
        wants_leaflet = any(w in lower for w in LEAFLET_WORDS)
        if known and not (wants_price or wants_stock or wants_controlled or wants_leaflet):
            wants_leaflet = True
        wants_medicine = bool(known) and (wants_price or wants_stock or wants_controlled)

        if not (wants_leaflet or wants_medicine or wants_account):
            return say("I can only help with medicine information and pharmacy orders.")

        todo = []
        # like a real model, the mock rewrites the question: first part only, without the pharmacy ID
        leaflet_question = PH_ID.sub("", re.split(r",\s*and\b", request)[0]).strip(" :,")
        if wants_leaflet and result("search_leaflet", question=leaflet_question) is None:
            todo.append(("search_leaflet", {"question": leaflet_question}))
        if wants_medicine and result("search_medicine", name=known[0]) is None:
            todo.append(("search_medicine", {"name": known[0]}))
        if wants_account and result("get_pharmacy", pharmacy_id=pharmacies[0]) is None:
            todo.append(("get_pharmacy", {"pharmacy_id": pharmacies[0]}))
        if todo:
            return ask(*todo)

        qty_match = QTY.search(request)
        qty = int(qty_match.group(1)) if qty_match else 1
        med = None
        if wants_medicine:
            matches = result("search_medicine", name=known[0])["matches"]
            if not matches:
                return say("This medicine was not found in our catalogue.")
            med = matches[0]
            if wants_stock and result("check_stock", medicine_id=med["id"], quantity=qty) is None:
                return ask(("check_stock", {"medicine_id": med["id"], "quantity": qty}))

        parts = []
        if wants_leaflet:
            found = result("search_leaflet", question=leaflet_question).get("passages", [])
            parts.append("From the leaflets: " + " ".join(p["text"] for p in found[:3]) if found
                         else "I found nothing about this in the leaflets, please ask a pharmacist.")
        if med and wants_price:
            parts.append(f"{med['name']}: price {med['price']:.2f} per box.")
        if med and wants_controlled:
            parts.append(f"{med['name']} is {'a' if med['controlled'] else 'not a'} controlled medicine.")
        if med and wants_stock:
            stock = result("check_stock", medicine_id=med["id"], quantity=qty)
            if "error" in stock:
                parts.append(stock["error"])
            elif stock["available"]:
                parts.append(f"{med['name']}: {stock['stock']} boxes in stock.")
            else:
                parts.append(f"Not enough stock for {med['name']}: only {stock['stock']} boxes available.")
        if wants_account:
            pharmacy = result("get_pharmacy", pharmacy_id=pharmacies[0])
            parts.append(f"Pharmacy {pharmacies[0]} account status: {pharmacy.get('status', 'not found')}.")
        return say(" ".join(parts))

    @staticmethod
    def _history(messages) -> dict:
        """{(tool name, arguments as sorted json): result} rebuilt from the messages."""
        asked = {}
        for m in messages:
            if m["role"] == "assistant":
                for c in m.get("tool_calls", []):
                    asked[c["id"]] = (c["function"]["name"],
                                      json.dumps(json.loads(c["function"]["arguments"]), sort_keys=True))
        history = {}
        for m in messages:
            if m["role"] == "tool" and m["tool_call_id"] in asked:
                history[asked[m["tool_call_id"]]] = json.loads(m["content"])
        return history
