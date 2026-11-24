"""Stage 6 - the tools of the ordering agent and the business rules.

The model never touches the data. It asks for a tool (name + arguments); OUR code runs it.
Five tools, only ONE has an effect (create_order):

  search_medicine(name)                              read only
  check_stock(medicine_id, quantity)                 read only
  get_pharmacy(pharmacy_id)                          read only
  search_leaflet(question)                           read only (reuses Part A's search)
  create_order(pharmacy_id, medicine_id, quantity)   WRITES an order

YOUR JOB (2 things, marked below): SCHEMA_CREATE_ORDER and validate_order.
Check with:  python -m pytest tests/test_07_tools.py -q
"""
import json
from difflib import SequenceMatcher

from app.config import DATA, MAX_QTY, MIN_QTY, REVIEW_QTY, SEARCH_THRESHOLD
from app.schemas import OrderResult, OrderStatus
from app.search import search_leaflets

# ------------------------------------------------------------------ data (in memory)
STATE = {}


def reset_state():
    """Reload medicines, pharmacies and an empty list of orders. Call it between two test tasks."""
    medicines = json.loads((DATA / "medicines.json").read_text(encoding="utf-8"))
    pharmacies = json.loads((DATA / "pharmacies.json").read_text(encoding="utf-8"))
    STATE["medicines"] = {m["id"]: m for m in medicines}
    STATE["pharmacies"] = {p["id"]: p for p in pharmacies}
    STATE["orders"] = []


reset_state()


# ------------------------------------------------------------------ read-only tools (PROVIDED)
def search_medicine(name: str) -> dict:
    wanted = name.lower().strip()
    scored = []
    for med in STATE["medicines"].values():
        full = med["name"].lower()
        first = full.split()[0]
        score = max(SequenceMatcher(None, wanted, full).ratio(), SequenceMatcher(None, wanted, first).ratio())
        if score >= 0.7:
            scored.append((score, med))
    scored.sort(key=lambda pair: -pair[0])
    matches = [{"id": m["id"], "name": m["name"], "price": m["price"], "controlled": m["controlled"],
                "prescription_only": m["prescription_only"]} for _, m in scored[:3]]
    return {"matches": matches} if matches else {"matches": [], "message": "medicine not found"}


def check_stock(medicine_id: str, quantity: int = 1) -> dict:
    if not isinstance(quantity, int) or not MIN_QTY <= quantity <= MAX_QTY:
        return {"error": f"quantity must be between {MIN_QTY} and {MAX_QTY} (maximum per order: {MAX_QTY} boxes)"}
    med = STATE["medicines"].get(medicine_id)
    if med is None:
        return {"error": "medicine not found"}
    return {"medicine_id": medicine_id, "name": med["name"], "stock": med["stock"],
            "available": med["stock"] >= quantity}


def get_pharmacy(pharmacy_id: str) -> dict:
    ph = STATE["pharmacies"].get(pharmacy_id)
    if ph is None:
        return {"error": "pharmacy not found"}
    return {"id": ph["id"], "name": ph["name"], "status": ph["status"],
            "special_authorization": ph["special_authorization"]}


def search_leaflet(question: str) -> dict:
    found = search_leaflets(question, k=3)
    if not found or found[0]["score"] < SEARCH_THRESHOLD:
        return {"passages": [], "message": "nothing relevant in the leaflets"}
    return {"passages": [{"id": p["id"], "text": p["text"], "score": p["score"]} for p in found]}


# ------------------------------------------------------------------ the business rules (YOU)
def validate_order(pharmacy: dict, medicine: dict, quantity: int) -> tuple[OrderStatus, str]:
    """Decide what happens to an order. THE CODE DECIDES, NOT THE MODEL.

    Apply these rules in this order, and return (status, reason) at the first one that applies:
      1. quantity is not an integer between MIN_QTY and MAX_QTY
            -> rejected, reason mentions "maximum per order is 100 boxes"
      2. pharmacy["status"] == "blocked"
            -> rejected, reason mentions "blocked"
      3. medicine["controlled"] and not pharmacy["special_authorization"]
            -> rejected, reason mentions "special authorization"
      4. medicine["stock"] == 0                -> rejected, reason mentions "out of stock"
         medicine["stock"] < quantity          -> rejected, reason like "only 8 boxes available"
      5. quantity > REVIEW_QTY (but not above MAX_QTY)
            -> pending_review, reason mentions "needs review"
      6. otherwise -> confirmed
    """
    if not isinstance(quantity, int) or not MIN_QTY <= quantity <= MAX_QTY:
        return OrderStatus.rejected, f"the maximum per order is {MAX_QTY} boxes (minimum {MIN_QTY})"
    if pharmacy["status"] == "blocked":
        return OrderStatus.rejected, "the pharmacy account is blocked"
    if medicine["controlled"] and not pharmacy["special_authorization"]:
        return OrderStatus.rejected, "controlled medicine: special authorization required"
    if medicine["stock"] == 0:
        return OrderStatus.rejected, "out of stock"
    if medicine["stock"] < quantity:
        return OrderStatus.rejected, f"not enough stock: only {medicine['stock']} boxes available"
    if quantity > REVIEW_QTY:
        return OrderStatus.pending_review, "large order: needs review by our team"
    return OrderStatus.confirmed, "order accepted"


def create_order(pharmacy_id: str, medicine_id: str, quantity: int) -> dict:
    """PROVIDED. The only tool with an effect. It always goes through validate_order."""
    pharmacy = STATE["pharmacies"].get(pharmacy_id)
    medicine = STATE["medicines"].get(medicine_id)
    if pharmacy is None:
        return {"error": "pharmacy not found"}
    if medicine is None:
        return {"error": "medicine not found"}
    status, reason = validate_order(pharmacy, medicine, quantity)
    order_id = None
    if status != OrderStatus.rejected:
        order_id = f"ORD-{len(STATE['orders']) + 1:04d}"
        STATE["orders"].append({"id": order_id, "pharmacy_id": pharmacy_id,
                                "medicine_id": medicine_id, "quantity": quantity, "status": status.value})
        if status == OrderStatus.confirmed:
            medicine["stock"] -= quantity
    result = OrderResult(status=status, reason=reason, order_id=order_id,
                         pharmacy_id=pharmacy_id, medicine_id=medicine_id, quantity=quantity)
    return result.model_dump(mode="json")


# ------------------------------------------------------------------ tool descriptions (JSON Schema)
def schema(name, description, properties, required):
    """PROVIDED. Builds the description the model reads. The description IS the model's instruction."""
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties, "required": required}}}


PHARMACY_ID = {"type": "string", "description": "Pharmacy ID: PH followed by 3 digits, e.g. PH001"}
MEDICINE_ID = {"type": "string", "description": "Medicine ID such as M01, taken from search_medicine"}

SCHEMA_SEARCH_MEDICINE = schema(
    "search_medicine",
    "Finds a medicine in the catalogue by name (spelling mistakes allowed) and returns its ID, price "
    "and whether it is a controlled medicine. Use it first whenever a medicine is mentioned.",
    {"name": {"type": "string", "description": "Medicine name, e.g. Cefalix 500"}}, ["name"])

SCHEMA_CHECK_STOCK = schema(
    "check_stock",
    "Tells how many boxes of a medicine are in stock and whether the wanted quantity is available. "
    "Use it for any question about availability, or before ordering. It does not order anything.",
    {"medicine_id": MEDICINE_ID,
     "quantity": {"type": "integer", "minimum": 1, "description": "Number of boxes wanted (1 if not stated)"}},
    ["medicine_id"])

SCHEMA_GET_PHARMACY = schema(
    "get_pharmacy",
    "Returns the account of ONE pharmacy: whether it is active or blocked and whether it may order "
    "controlled medicines. Only for the pharmacy that is speaking, never for another one.",
    {"pharmacy_id": PHARMACY_ID}, ["pharmacy_id"])

SCHEMA_SEARCH_LEAFLET = schema(
    "search_leaflet",
    "Searches the medicine leaflets (appearance, use, dosage, warnings, side effects). Use it for any "
    "medical-information question. Never answer such a question from memory.",
    {"question": {"type": "string", "description": "The question, in the user's words"}}, ["question"])

# YOUR JOB: describe create_order. It is the ONLY tool with an effect, so the description must say
# clearly WHEN to use it and when NOT to (questions about price or stock, "thinking about ordering",
# a missing pharmacy ID...). Use schema(...) like the four above.
# Parameters: pharmacy_id (PHARMACY_ID), medicine_id (MEDICINE_ID),
#             quantity (integer, minimum 1, "Number of boxes"). All three are required.
SCHEMA_CREATE_ORDER = schema(
    "create_order",
    "Places a REAL order for a pharmacy. Use it ONLY when the pharmacy clearly and explicitly asks to "
    "order or buy something and has given its pharmacy ID. Never use it for questions about price or "
    "stock, for hypothetical or future orders, or when the pharmacy ID or quantity is missing.",
    {"pharmacy_id": PHARMACY_ID, "medicine_id": MEDICINE_ID,
     "quantity": {"type": "integer", "minimum": 1, "description": "Number of boxes"}},
    ["pharmacy_id", "medicine_id", "quantity"])

SCHEMAS = [SCHEMA_SEARCH_MEDICINE, SCHEMA_CHECK_STOCK, SCHEMA_GET_PHARMACY,
           SCHEMA_SEARCH_LEAFLET, SCHEMA_CREATE_ORDER]


def make_vague(schemas):
    """PROVIDED. Same tools, but each description is reduced to the tool name and parameter
    descriptions are removed. Used in stage 8 to change ONE variable: the quality of the descriptions."""
    vague = []
    for s in schemas:
        fn = s["function"]
        properties = {k: {"type": v["type"]} for k, v in fn["parameters"]["properties"].items()}
        vague.append(schema(fn["name"], fn["name"].replace("_", " "), properties,
                            fn["parameters"]["required"]))
    return vague


# ------------------------------------------------------------------ running a tool
TOOLS = {"search_medicine": search_medicine, "check_stock": check_stock, "get_pharmacy": get_pharmacy,
         "search_leaflet": search_leaflet, "create_order": create_order}
READ_ONLY = {"search_medicine", "check_stock", "get_pharmacy", "search_leaflet"}


def run_tool(name: str, arguments: dict) -> dict:
    """PROVIDED. Runs a tool. A mistake of the model (unknown tool, wrong arguments) becomes a
    readable {"error": ...} that the model can see, instead of crashing the program."""
    if name not in TOOLS:
        return {"error": f"unknown tool: {name}"}
    try:
        return TOOLS[name](**arguments)
    except TypeError as exc:
        return {"error": f"invalid arguments: {exc}"}
