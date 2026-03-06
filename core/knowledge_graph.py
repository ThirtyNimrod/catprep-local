"""
knowledge_graph.py
------------------
Provides graph-enhanced context retrieval from a pre-built .graphml file.

Strategy
--------
1. Load the GraphML once (module-level singleton, thread-safe for read).
2. For a user query, extract candidate keywords using a simple tokenised
   approach (avoids an extra LLM call).
3. Walk 1–2 hops from every matched node and collect the resulting
   (source, relation, target) triples.
4. Return a compact, formatted string that can be appended to FAISS context.
"""

import threading
import re
from pathlib import Path

import networkx as nx

from core.config import CONTEXT_DIR
from core.logger import get_logger

logger = get_logger("knowledge_graph")

# ---------------------------------------------------------------------------
# Singleton graph loader
# ---------------------------------------------------------------------------
_graph: nx.DiGraph | None = None
_graph_lock = threading.Lock()

# Graph lives in data/ (separate from raw PDFs in context/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
GRAPH_PATH = _PROJECT_ROOT / "data" / "knowledge_graph.graphml"

# CAT-specific stop words that are too generic to be useful graph seeds
_STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "and", "or", "but", "if", "in", "on", "at", "to", "for", "of", "by",
    "with", "from", "this", "that", "these", "those", "i", "me", "my",
    "you", "your", "he", "she", "it", "we", "they", "what", "which",
    "who", "whom", "how", "when", "where", "why", "not", "no", "yes",
    "give", "get", "tell", "show", "make", "want", "need", "like", "use",
    "please", "help", "more", "some", "all", "any", "each", "few",
    "question", "questions", "answer", "answers", "practice", "test",
    "mock", "exam", "cat", "about", "me", "my", "plan", "study"
}


def _load_graph() -> nx.DiGraph | None:
    """Load and cache the knowledge graph from disk."""
    global _graph
    if _graph is not None:
        return _graph
    with _graph_lock:
        if _graph is not None:   # double-check after acquiring lock
            return _graph
        if not GRAPH_PATH.exists():
            logger.warning(
                f"Knowledge graph not found at {GRAPH_PATH}. "
                "Run scripts/build_knowledge_graph.py first."
            )
            return None
        try:
            logger.info(f"Loading knowledge graph from {GRAPH_PATH} ...")
            _graph = nx.read_graphml(str(GRAPH_PATH))
            logger.info(
                f"Knowledge graph loaded: "
                f"{_graph.number_of_nodes()} nodes, "
                f"{_graph.number_of_edges()} edges."
            )
        except Exception as exc:
            logger.error(f"Failed to load knowledge graph: {exc}")
            return None
    return _graph


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_keywords(query: str) -> list[str]:
    """Return cleaned, lower-cased tokens from the query (≥3 chars, not stop words)."""
    tokens = re.findall(r"[a-zA-Z0-9/]+", query.lower())
    return [t for t in tokens if len(t) >= 3 and t not in _STOP_WORDS]


def _node_matches_keyword(node_label: str, keywords: list[str]) -> bool:
    """True if any keyword is a substring of the node label (case-insensitive)."""
    label_lower = node_label.lower()
    return any(kw in label_lower for kw in keywords)


def _triples_from_node(G: nx.DiGraph, node: str, hops: int = 2) -> list[tuple]:
    """BFS up to `hops` away from `node`, collecting (src, rel, tgt) triples."""
    visited_nodes = {node}
    frontier = {node}
    triples = []

    for _ in range(hops):
        next_frontier = set()
        for n in frontier:
            # Outgoing edges
            for _, tgt, data in G.out_edges(n, data=True):
                rel = data.get("relation", "related_to")
                text = data.get("source_text", "")
                triples.append((n, rel, tgt, text))
                if tgt not in visited_nodes:
                    next_frontier.add(tgt)
                    visited_nodes.add(tgt)
            # Incoming edges (reverse direction)
            for src, _, data in G.in_edges(n, data=True):
                rel = data.get("relation", "related_to")
                text = data.get("source_text", "")
                triples.append((src, rel, n, text))
                if src not in visited_nodes:
                    next_frontier.add(src)
                    visited_nodes.add(src)
        frontier = next_frontier
        if not frontier:
            break

    return triples


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def retrieve_graph_context(
    query: str,
    max_seed_nodes: int = 8,
    max_triples: int = 25,
    hops: int = 2,
) -> tuple[str, list[tuple]]:
    """
    Return a formatted string of knowledge-graph triples relevant to `query`,
    along with the raw list of triples for UI visualization.

    Parameters
    ----------
    query          : The user's question / search text.
    max_seed_nodes : Maximum number of matching graph nodes to expand from.
    max_triples    : Hard cap on total triples returned (keeps context short).
    hops           : Number of hops to walk from each seed node.

    Returns
    -------
    A tuple (context_string, raw_triples).
    """
    G = _load_graph()
    if G is None:
        return "", []

    keywords = _extract_keywords(query)
    seed_nodes = []

    if keywords:
        # Find seed nodes whose labels match any keyword
        for node in G.nodes:
            label = G.nodes[node].get("label", node)
            if _node_matches_keyword(str(label), keywords):
                seed_nodes.append(node)
            if len(seed_nodes) >= max_seed_nodes:
                break

    # FALLBACK: If query is generic (no keywords matched), pick central nodes
    if not seed_nodes:
        logger.debug(f"No graph nodes matched query: '{query}'. Falling back to central nodes.")
        try:
            deg_cent = nx.degree_centrality(G)
            sorted_nodes = sorted(deg_cent.items(), key=lambda x: x[1], reverse=True)
            seed_nodes = [node for node, _ in sorted_nodes[:max_seed_nodes]]
        except Exception as e:
            logger.error(f"Fallback centrality failed: {e}")
            seed_nodes = list(G.nodes)[:max_seed_nodes]

    if not seed_nodes:
        return "", []

    # Collect triples via BFS from each seed
    all_triples: list[tuple] = []   # UI triplets: (src, rel, tgt)
    seen_triples: set[tuple] = set()
    source_texts: set[str] = set()  # Raw text snippets attached to edges

    for node in seed_nodes:
        for src, rel, tgt, text in _triples_from_node(G, node, hops=hops):
            trip = (src, rel, tgt)
            if trip not in seen_triples:
                seen_triples.add(trip)
                all_triples.append(trip)
            
            if text and len(text) > 20: # skip empty/tiny texts
                source_texts.add(text)
                
            if len(all_triples) >= max_triples:
                break
        if len(all_triples) >= max_triples:
            break

    if not all_triples:
        return "", []

    lines = ["[Knowledge Graph Extracted Relations]"]
    for src, rel, tgt in all_triples:
        lines.append(f"  {src} --[{rel}]--> {tgt}")

    if source_texts:
        lines.append("\n[Original Source Text Snippets]")
        # Limit snippets to avoid blowing up context window
        for text in list(source_texts)[:5]:
            lines.append(f"---\n{text.strip()}\n---")

    return "\n".join(lines), all_triples
