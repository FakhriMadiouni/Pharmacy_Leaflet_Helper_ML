import pytest

from app.schemas import OrderStatus
from app.tools import (SCHEMAS, STATE, create_order, check_stock, get_pharmacy, make_vague,
                       reset_state, run_tool, search_medicine, validate_order)


@pytest.fixture(autouse=True)
def fresh_state():
    reset_state()


def rule(pharmacy_id, medicine_id, quantity):
    return validate_order(STATE["pharmacies"][pharmacy_id], STATE["medicines"][medicine_id], quantity)


# ---- business rules (validate_order)
def test_normal_order_confirmed():
    assert rule("PH001", "M01", 20)[0] == OrderStatus.confirmed


def test_quantity_bounds():
    for bad in (0, -3, 101, 500):
        status, reason = rule("PH001", "M01", bad)
        assert status == OrderStatus.rejected and "maximum" in reason
    assert rule("PH001", "M01", 100)[0] == OrderStatus.pending_review  # 100 itself is inside the bounds


def test_blocked_pharmacy_rejected():
    status, reason = rule("PH003", "M01", 5)
    assert status == OrderStatus.rejected and "blocked" in reason


def test_controlled_medicine_needs_authorization():
    status, reason = rule("PH001", "M08", 5)
    assert status == OrderStatus.rejected and "special authorization" in reason
    assert rule("PH002", "M08", 5)[0] == OrderStatus.confirmed


def test_stock_rules():
    status, reason = rule("PH004", "M09", 5)
    assert status == OrderStatus.rejected and "out of stock" in reason
    status, reason = rule("PH004", "M05", 20)
    assert status == OrderStatus.rejected and "8" in reason


def test_large_order_goes_to_review():
    status, reason = rule("PH001", "M01", 80)
    assert status == OrderStatus.pending_review and "review" in reason
    assert rule("PH001", "M01", 50)[0] == OrderStatus.confirmed


def test_rule_order_blocked_before_stock():
    assert "blocked" in rule("PH003", "M09", 5)[1]


# ---- create_order (uses the rules, records the order)
def test_create_order_confirmed_updates_stock_and_book():
    result = create_order("PH001", "M01", 20)
    assert result["status"] == "confirmed" and result["order_id"] == "ORD-0001"
    assert STATE["medicines"]["M01"]["stock"] == 100
    assert len(STATE["orders"]) == 1


def test_create_order_rejected_leaves_no_trace():
    result = create_order("PH003", "M01", 20)
    assert result["status"] == "rejected" and result["order_id"] is None
    assert STATE["orders"] == [] and STATE["medicines"]["M01"]["stock"] == 120


def test_create_order_pending_review_is_recorded_but_stock_untouched():
    result = create_order("PH001", "M01", 80)
    assert result["status"] == "pending_review"
    assert len(STATE["orders"]) == 1 and STATE["medicines"]["M01"]["stock"] == 120


def test_create_order_unknown_ids():
    assert create_order("PH999", "M01", 1) == {"error": "pharmacy not found"}
    assert create_order("PH001", "M99", 1) == {"error": "medicine not found"}


# ---- read-only tools and run_tool
def test_read_only_tools():
    assert search_medicine("tussimel")["matches"][0]["id"] == "M05"
    assert search_medicine("Zyloprex")["matches"] == []
    assert check_stock("M05", 3)["available"] is True
    assert check_stock("M05", 20)["available"] is False
    assert "maximum" in check_stock("M01", 500)["error"]
    assert get_pharmacy("PH003")["status"] == "blocked"
    assert get_pharmacy("PH999")["error"]


def test_run_tool_turns_mistakes_into_readable_errors():
    assert "unknown tool" in run_tool("delete_everything", {})["error"]
    assert "invalid arguments" in run_tool("create_order", {"pharmacy_id": "PH001"})["error"]
    assert STATE["orders"] == []


# ---- the descriptions
def test_five_tools_described_in_json_schema():
    assert len(SCHEMAS) == 5
    names = {s["function"]["name"] for s in SCHEMAS}
    assert names == {"search_medicine", "check_stock", "get_pharmacy", "search_leaflet", "create_order"}
    for s in SCHEMAS:
        fn = s["function"]
        assert s["type"] == "function" and len(fn["description"]) > 40
        assert set(fn["parameters"]["required"]) <= set(fn["parameters"]["properties"])


def test_create_order_schema_is_strict_about_when_to_use_it():
    fn = next(s["function"] for s in SCHEMAS if s["function"]["name"] == "create_order")
    assert set(fn["parameters"]["required"]) == {"pharmacy_id", "medicine_id", "quantity"}
    assert fn["parameters"]["properties"]["quantity"]["type"] == "integer"
    text = fn["description"].lower()
    assert "only" in text and "explicit" in text


def test_vague_schemas_keep_names_and_parameters():
    vague = make_vague(SCHEMAS)
    assert [s["function"]["name"] for s in vague] == [s["function"]["name"] for s in SCHEMAS]
    assert all(len(s["function"]["description"]) < 30 for s in vague)
