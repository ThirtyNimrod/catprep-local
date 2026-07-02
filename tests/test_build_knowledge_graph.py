import asyncio

import networkx as nx

import scripts.build_knowledge_graph as bkg
from scripts.build_knowledge_graph import (
    Relationship,
    add_relationships_to_graph,
    chunk_text,
    is_worth_extracting,
)


def test_chunk_text():
    text = "Hello world! This is a test string."
    chunks = chunk_text(text, chunk_size=15, chunk_overlap=5)
    assert isinstance(chunks, list)
    assert len(chunks) > 0


def test_add_relationships_to_graph():
    G = nx.DiGraph()
    canonical_map = {}
    seen_edges = set()

    rels1 = [
        Relationship(source="Time & Work", target="Formulas", relation="has"),
        Relationship(source="Math", target="Time and Work", relation="includes")
    ]

    add_relationships_to_graph(G, rels1, canonical_map, seen_edges, "doc1.pdf", "text 1")

    nodes = list(G.nodes)
    assert "Time & Work" in nodes
    assert "Time and Work" not in nodes
    assert "Formulas" in nodes
    assert "Math" in nodes

    rels2 = [
        Relationship(source="time and work", target="formulas", relation="HAS")
    ]
    add_relationships_to_graph(G, rels2, canonical_map, seen_edges, "doc2.pdf", "text 2")

    edges = list(G.edges(data=True))
    assert len(edges) == 2


def test_is_worth_extracting_rejects_short_chunks():
    assert not is_worth_extracting("too short")


def test_is_worth_extracting_rejects_pure_instruction_boilerplate():
    text = (
        "Read the following instructions carefully. Time allowed is 60 minutes. "
        "General instructions: do not open this booklet until instructed."
    )
    assert not is_worth_extracting(text)


def test_is_worth_extracting_accepts_real_content():
    text = (
        "Quantitative Aptitude covers Number Systems, Algebra, Geometry, and "
        "Time-Speed-Distance problems, each carrying different weightage in the exam."
    )
    assert is_worth_extracting(text)


def test_create_batches_respects_batch_size():
    chunks = [{"page_content": "x" * 10} for _ in range(10)]
    batches = bkg._create_batches(chunks, batch_size=4, fallback_size=2, max_chars=10000)
    assert [len(b) for b in batches] == [4, 4, 2]


def test_create_batches_falls_back_when_too_many_chars():
    chunks = [{"page_content": "x" * 100} for _ in range(4)]
    batches = bkg._create_batches(chunks, batch_size=4, fallback_size=2, max_chars=250)
    # A full batch of 4 would be 400 chars (> 250), so it falls back to 2 chunks/batch.
    assert [len(b) for b in batches] == [2, 2]


class _FakeStructuredLLM:
    """Stands in for `llm.with_structured_output(...)` — returns the same
    canned relationships for every batch, regardless of the chunk text."""

    def __init__(self, relationships):
        self.relationships = relationships

    async def ainvoke(self, messages):
        class _Result:
            pass

        result = _Result()
        result.relationships = self.relationships
        return result


def test_build_graph_async_dedupes_relationships_across_separate_batches(monkeypatch):
    # Force one chunk per batch so the two chunks below are processed as two
    # independent async batches that both "extract" the same relationship —
    # exercising the cross-batch dedup path, not just the pure function.
    monkeypatch.setattr(bkg, "BATCH_SIZE", 1)
    monkeypatch.setattr(bkg, "BATCH_FALLBACK", 1)
    monkeypatch.setattr(bkg, "MAX_BATCH_CHARS", 10_000)

    fake_llm = _FakeStructuredLLM(
        [Relationship(source="Time and Work", target="Formulas", relation="has")]
    )
    chunks = [
        {"page_content": "chunk one", "metadata": {"source_file": "doc1.pdf", "chunk_index": 0}},
        {"page_content": "chunk two", "metadata": {"source_file": "doc1.pdf", "chunk_index": 1}},
    ]

    G = asyncio.run(bkg.build_graph_async(chunks, fake_llm))

    assert G.number_of_nodes() == 2
    assert G.number_of_edges() == 1
