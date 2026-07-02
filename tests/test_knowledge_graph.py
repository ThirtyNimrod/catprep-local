from core.knowledge_graph import _extract_keywords, _node_matches_keyword, retrieve_graph_context

def test_extract_keywords():
    assert "q5" in _extract_keywords("What is Q5?")
    assert "a1" in _extract_keywords("a1")
    assert "the" not in _extract_keywords("the cat")
    assert "cat" not in _extract_keywords("the cat") # 'cat' is in stop words!

def test_node_matches_keyword():
    assert _node_matches_keyword("Corporate Finance", ["corporate"])
    # Test the fix: "rate" shouldn't match "Corporate"
    assert not _node_matches_keyword("Corporate Finance", ["rate"])
    assert _node_matches_keyword("Interest Rate", ["rate"])

def test_retrieve_graph_context(sample_graph):
    # Test with embeddings disabled to isolate keyword logic
    context, triples = retrieve_graph_context(
        query="Corporate Finance", 
        graph=sample_graph, 
        use_embeddings=False
    )
    assert "Corporate Finance" in context
    assert "Interest Rate" in context
    
    context2, triples2 = retrieve_graph_context(
        query="Q5",
        graph=sample_graph,
        use_embeddings=False
    )
    assert "Q5" in context2
    assert "A1" in context2
    
    # Fallback to centrality
    context3, triples3 = retrieve_graph_context(
        query="unknownquery",
        graph=sample_graph,
        use_embeddings=False
    )
    assert len(triples3) > 0 # Should pick something based on degree centrality
