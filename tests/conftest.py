"""Shared fixtures for pytest"""
import pytest
import networkx as nx

@pytest.fixture
def sample_graph():
    G = nx.DiGraph()
    G.add_node("Corporate Finance", label="Corporate Finance")
    G.add_node("Interest Rate", label="Interest Rate")
    G.add_node("Q5", label="Q5")
    G.add_node("A1", label="A1")
    G.add_edge("Corporate Finance", "Interest Rate", relation="includes", source_text="CF includes IR.")
    G.add_edge("Q5", "A1", relation="has_answer", source_text="Q5 -> A1")
    return G
