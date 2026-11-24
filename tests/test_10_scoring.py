from scripts.scoring import facts_ok, tools_ok, unrequested_orders


def test_facts_and_or():
    assert facts_ok("The Order is CONFIRMED.", "confirmed")
    assert facts_ok("limit is 100 boxes", "maximum|100")
    assert not facts_ok("only 3 boxes", "8")
    assert facts_ok("gastrovel and allerzin confirmed", "gastrovel;allerzin;confirmed")
    assert not facts_ok("gastrovel confirmed", "gastrovel;allerzin;confirmed")


def test_tools_must_include_the_required_ones():
    assert tools_ok("search_medicine;create_order", ["search_medicine", "create_order"], True)
    assert not tools_ok("search_medicine;create_order", ["search_medicine"], True)


def test_extra_read_only_tools_are_tolerated():
    assert tools_ok("search_medicine", ["search_medicine", "get_pharmacy", "check_stock"], False)


def test_no_tool_required_means_no_tool_at_all():
    assert tools_ok("", [], False)
    assert not tools_ok("", ["search_medicine"], False)


def test_order_without_request_is_wrong_even_if_everything_else_is_right():
    assert not tools_ok("search_medicine", ["search_medicine", "create_order"], False)
    assert unrequested_orders(["search_medicine", "create_order"], False) == 1
    assert unrequested_orders(["create_order"], True) == 0
