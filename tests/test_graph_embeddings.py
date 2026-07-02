import math
import networkx as nx
from core.graph_embeddings import cosine_similarity, get_node_embeddings, embed_query

class FakeEmbedder:
    def embed_documents(self, texts):
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text):
        return [1.0, 0.0]

def test_cosine_similarity():
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0
    assert math.isclose(cosine_similarity([1.0, 0.0], [1.0, 1.0]), 0.7071, rel_tol=1e-3)
    assert cosine_similarity([], []) == 0.0

def test_get_node_embeddings_and_query(tmp_path):
    G = nx.DiGraph()
    G.add_node("A")
    G.add_node("B")

    embedder = FakeEmbedder()
    # Route both the graphml mtime check and the cache write through tmp_path so
    # this test can never touch the real data/knowledge_graph_embeddings.json.
    embs = get_node_embeddings(
        G,
        embedder=embedder,
        graphml_path=tmp_path / "dummy.graphml",
        cache_path=tmp_path / "cache.json",
    )

    assert len(embs) == 2
    assert "A" in embs
    assert embs["A"] == [1.0, 0.0]
    assert (tmp_path / "cache.json").exists()

    q_emb = embed_query("test", embedder=embedder)
    assert q_emb == [1.0, 0.0]
