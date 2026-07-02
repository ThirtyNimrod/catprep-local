from core.graph_utils import normalize_entity_name

def test_normalize_entity_name():
    assert normalize_entity_name("Time and Work") == "time and work"
    assert normalize_entity_name("Time & Work") == "time and work"
    assert normalize_entity_name("  Time  and  Work  ") == "time and work"
    assert normalize_entity_name("Time, and Work!") == "time and work"
