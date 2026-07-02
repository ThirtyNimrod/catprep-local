import json
import math
from pathlib import Path
import networkx as nx
from core.embeddings import get_embeddings
from core.logger import get_logger

logger = get_logger("graph_embeddings")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
EMBEDDINGS_CACHE_PATH = _PROJECT_ROOT / "data" / "knowledge_graph_embeddings.json"

def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)

def get_node_embeddings(
    G: nx.DiGraph,
    embedder=None,
    force_refresh: bool = False,
    graphml_path: Path | None = None,
    cache_path: Path | None = None,
) -> dict[str, list[float]] | None:
    if embedder is None:
        try:
            embedder = get_embeddings()
        except Exception as e:
            logger.error(f"Failed to initialize embedder: {e}")
            return None

    if graphml_path is None:
        graphml_path = _PROJECT_ROOT / "data" / "knowledge_graph.graphml"
    if cache_path is None:
        cache_path = EMBEDDINGS_CACHE_PATH

    mtime = 0.0
    if graphml_path.exists():
        mtime = graphml_path.stat().st_mtime

    if not force_refresh and cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cache = json.load(f)
            if cache.get("_generated_from_mtime") == mtime:
                return cache.get("embeddings", {})
        except Exception as e:
            logger.warning(f"Failed to load embeddings cache: {e}")

    nodes = list(G.nodes())
    if not nodes:
        return {}

    logger.info(f"Generating embeddings for {len(nodes)} nodes...")
    try:
        node_embeddings_list = embedder.embed_documents(nodes)
        embeddings = {n: e for n, e in zip(nodes, node_embeddings_list)}

        cache = {
            "_generated_from_mtime": mtime,
            "embeddings": embeddings
        }
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f)

        return embeddings
    except Exception as e:
        logger.error(f"Failed to generate embeddings: {e}")
        return None

def embed_query(query: str, embedder=None) -> list[float] | None:
    if embedder is None:
        try:
            embedder = get_embeddings()
        except Exception as e:
            logger.error(f"Failed to initialize embedder: {e}")
            return None

    try:
        return embedder.embed_query(query)
    except Exception as e:
        logger.error(f"Failed to embed query: {e}")
        return None
