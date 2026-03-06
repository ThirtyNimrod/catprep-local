"""
Knowledge Graph Visualizer — PyVis + NetworkX helper for Streamlit.

Provides functions to build interactive graph HTML, compute stats,
and filter triples for the Knowledge Graph panel.
"""

from __future__ import annotations

import tempfile
from typing import Optional

import networkx as nx
from pyvis.network import Network


# ---------------------------------------------------------------------------
# Color palette for node types (auto-assigned by hash when type is unknown)
# ---------------------------------------------------------------------------
_PALETTE = [
    "#4F46E5", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6",
    "#EC4899", "#06B6D4", "#84CC16", "#F97316", "#14B8A6",
]


def _color_for_type(node_type: str) -> str:
    """Deterministic color from the palette based on node string."""
    return _PALETTE[hash(node_type) % len(_PALETTE)]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_pyvis_html(
    triples: list[tuple[str, str, str]],
    highlight_node: Optional[str] = None,
    height: str = "560px",
    dark_mode: bool = False,
) -> str:
    """Build an interactive PyVis graph and return the raw HTML string.

    Args:
        triples: List of (subject, predicate, object) tuples.
        highlight_node: Optional node ID to highlight (larger + ring).
        height: CSS height for the canvas.
        dark_mode: If True, use a dark background.

    Returns:
        HTML string that can be rendered via ``st.components.v1.html()``.
    """
    bg_color = "#0E1117" if dark_mode else "#ffffff"
    font_color = "#ffffff" if dark_mode else "#333333"

    net = Network(
        height=height,
        width="100%",
        bgcolor=bg_color,
        font_color=font_color,
        directed=True,
        notebook=False,
        cdn_resources="remote",
    )

    # Barnes-Hut physics for nice force-directed layout
    net.barnes_hut(
        gravity=-3000,
        central_gravity=0.3,
        spring_length=120,
        spring_strength=0.05,
        damping=0.09,
    )

    # --- Build a NetworkX graph for centrality calculation ---
    G = nx.DiGraph()
    for src, rel, tgt in triples:
        G.add_edge(src, tgt, label=rel)

    degree_cent = nx.degree_centrality(G) if len(G) > 0 else {}

    # --- Add nodes ---
    added_nodes: set[str] = set()
    for src, _rel, tgt in triples:
        for node_id in (src, tgt):
            if node_id in added_nodes:
                continue
            added_nodes.add(node_id)

            centrality = degree_cent.get(node_id, 0.1)
            size = max(12, int(centrality * 80) + 12)
            color = _color_for_type(node_id)

            is_highlight = highlight_node and node_id == highlight_node
            border_width = 4 if is_highlight else 1
            border_color = "#FFD700" if is_highlight else color

            net.add_node(
                node_id,
                label=node_id,
                size=size,
                color={
                    "background": color,
                    "border": border_color,
                    "highlight": {"background": "#FFD700", "border": "#FFD700"},
                },
                borderWidth=border_width,
                title=f"<b>{node_id}</b><br>Connections: {G.degree(node_id)}",
                font={"size": 11, "color": font_color},
            )

    # --- Add edges ---
    for src, rel, tgt in triples:
        net.add_edge(
            src,
            tgt,
            title=rel,
            label=rel,
            color="#888888",
            font={"size": 9, "color": "#aaaaaa", "align": "middle"},
            arrows="to",
            smooth={"type": "curvedCW", "roundness": 0.15},
        )

    # Generate HTML string
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".html", mode="w", encoding="utf-8")
    net.save_graph(tmp.name)
    tmp.close()
    with open(tmp.name, "r", encoding="utf-8") as f:
        html = f.read()
    return html


def get_graph_stats(triples: list[tuple[str, str, str]]) -> dict:
    """Compute basic stats for the triple set.

    Returns:
        Dict with keys: node_count, edge_count, isolated_count, predicates.
    """
    if not triples:
        return {"node_count": 0, "edge_count": 0, "isolated_count": 0, "predicates": set()}

    G = nx.DiGraph()
    for src, rel, tgt in triples:
        G.add_edge(src, tgt, label=rel)

    isolated = list(nx.isolates(G))
    predicates = {rel for _, rel, _ in triples}

    return {
        "node_count": G.number_of_nodes(),
        "edge_count": G.number_of_edges(),
        "isolated_count": len(isolated),
        "predicates": predicates,
    }


def filter_triples(
    triples: list[tuple[str, str, str]],
    subject: Optional[str] = None,
    predicate: Optional[str] = None,
    object_: Optional[str] = None,
) -> list[tuple[str, str, str]]:
    """Filter triples by subject, predicate, and/or object.

    Pass ``None`` or empty string to skip a filter dimension.
    """
    result = triples
    if subject:
        result = [t for t in result if t[0] == subject]
    if predicate:
        result = [t for t in result if t[1] == predicate]
    if object_:
        result = [t for t in result if t[2] == object_]
    return result


def get_unique_values(
    triples: list[tuple[str, str, str]],
) -> tuple[list[str], list[str], list[str]]:
    """Extract sorted unique subjects, predicates, and objects.

    Returns:
        (subjects, predicates, objects) — each a sorted list of strings.
    """
    subjects = sorted({t[0] for t in triples})
    predicates = sorted({t[1] for t in triples})
    objects = sorted({t[2] for t in triples})
    return subjects, predicates, objects
