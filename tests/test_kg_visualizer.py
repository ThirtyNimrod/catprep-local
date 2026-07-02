from utils.kg_visualizer import (
    build_pyvis_html,
    filter_triples,
    get_graph_stats,
    get_unique_values,
)

TRIPLES = [
    ("Time and Work", "is_a", "Topic"),
    ("Q5", "tests", "Time and Work"),
    ("Q5", "has_answer", "A1"),
]


def test_get_graph_stats_counts_nodes_edges_and_predicates():
    stats = get_graph_stats(TRIPLES)
    assert stats["node_count"] == 4  # Time and Work, Topic, Q5, A1
    assert stats["edge_count"] == 3
    assert stats["predicates"] == {"is_a", "tests", "has_answer"}


def test_get_graph_stats_empty_triples():
    stats = get_graph_stats([])
    assert stats == {"node_count": 0, "edge_count": 0, "isolated_count": 0, "predicates": set()}


def test_filter_triples_by_subject():
    result = filter_triples(TRIPLES, subject="Q5")
    assert len(result) == 2
    assert all(t[0] == "Q5" for t in result)


def test_filter_triples_by_predicate():
    assert filter_triples(TRIPLES, predicate="is_a") == [("Time and Work", "is_a", "Topic")]


def test_filter_triples_by_object():
    assert filter_triples(TRIPLES, object_="A1") == [("Q5", "has_answer", "A1")]


def test_filter_triples_no_filters_returns_all():
    assert filter_triples(TRIPLES) == TRIPLES


def test_filter_triples_combined_filters_can_return_empty():
    assert filter_triples(TRIPLES, subject="Q5", predicate="is_a") == []


def test_get_unique_values_sorted():
    subjects, predicates, objects = get_unique_values(TRIPLES)
    assert subjects == ["Q5", "Time and Work"]
    assert predicates == ["has_answer", "is_a", "tests"]
    assert objects == ["A1", "Time and Work", "Topic"]


def test_build_pyvis_html_returns_html_containing_node_labels():
    html = build_pyvis_html(TRIPLES, height="300px")
    assert isinstance(html, str)
    assert "<html" in html.lower()
    assert "Q5" in html


def test_build_pyvis_html_empty_triples_still_returns_html():
    html = build_pyvis_html([])
    assert isinstance(html, str)
    assert "<html" in html.lower()
