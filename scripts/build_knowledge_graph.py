import re
import asyncio
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
import networkx as nx
import pypdf
import sys

# Add project root to sys path so we can import 'core'
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from core.logger import get_logger, LOGS_DIR
from core.config import CONTEXT_DIR, get_processing_config
from core.llm import get_llm
from core.graph_utils import normalize_entity_name
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tenacity import retry, stop_after_attempt, wait_exponential

logger = get_logger(
    "build_knowledge_graph",
    log_file=LOGS_DIR / "build_knowledge_graph.log",
    mode="a"
)

# --- Configuration ---
CONTEXT_DIR_PATH = Path(CONTEXT_DIR)
DATA_DIR = project_root / "data"
DATA_DIR.mkdir(exist_ok=True)
GRAPH_OUTPUT_FILE = DATA_DIR / "knowledge_graph.graphml"

# Load dynamic processing config based on LLM tier
processing_config = get_processing_config()
CHUNK_SIZE = processing_config["CHUNK_SIZE"]
CHUNK_OVERLAP = processing_config["CHUNK_OVERLAP"]
MAX_CONCURRENT = processing_config["MAX_CONCURRENT"]
BATCH_SIZE = processing_config["BATCH_SIZE"]
BATCH_FALLBACK = processing_config["BATCH_FALLBACK"]
MAX_BATCH_CHARS = processing_config["MAX_BATCH_CHARS"]

# --- System prompt (tight, token-efficient) ---
EXTRACTION_PROMPT = """You are a knowledge graph extractor for exam-preparation material.
Extract (source, target, relation) triples from EVERY chunk below.

Rules:
- Entities: named concepts, formulas, people, topics, sections, question IDs.
- Relations: short verbs or phrases ("is_a", "has", "tests", "requires", "sits_next_to", "scored").
- Skip generic words like "the text", "this passage", "option A/B/C/D" as entities.
- Keep answer-key references (e.g., "Q5 -> Answer -> B").
- Max 10 relationships per chunk. Prefer quality over quantity.
- Return an empty list if no meaningful relationships exist in a chunk."""


# --- Pydantic models ---
class Relationship(BaseModel):
    source: str = Field(description="The source entity name")
    target: str = Field(description="The target entity name")
    relation: str = Field(description="The relationship between source and target")

class ExtractedGraphData(BaseModel):
    model_config = ConfigDict(extra="ignore")

    relationships: list[Relationship] = Field(
        description="List of relationships extracted from the text.",
        default_factory=list
    )


# --- Helpers ---
def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Character-based chunking with overlap using langchain."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_text(text)


# Regex patterns that indicate instruction-only content
_INSTRUCTION_PATTERNS = re.compile(
    r"(read the following (directions|instructions)|"
    r"time allowed|general instructions|"
    r"do not open this (booklet|seal)|"
    r"mark your answer|use of calculator is not permitted|"
    r"there are \d+ questions in this paper|"
    r"each question carries \d+ marks?)",
    re.IGNORECASE
)


def is_worth_extracting(text: str) -> bool:
    """Return False for chunks that are pure exam instructions (no content)."""
    stripped = text.strip()
    if len(stripped) < 50:
        return False
    # If >60 % of the chunk matches instruction boilerplate, skip it
    instruction_hits = _INSTRUCTION_PATTERNS.findall(stripped)
    if len(instruction_hits) >= 3 and len(stripped) < 400:
        return False
    return True


def process_documents(context_dir: Path) -> list[dict]:
    """Parses PDFs in the given directory and chunks them."""
    docs = []
    pdf_files = list(context_dir.rglob("*.pdf"))

    doc_count = len(pdf_files)
    if doc_count == 0:
        logger.warning(f"No PDF files found in {context_dir}")
        return []

    logger.info(f"Found {doc_count} PDF files. Starting parsing...")

    for file_path in pdf_files:
        logger.info(f"Processing: {file_path.name}")
        try:
            reader = pypdf.PdfReader(str(file_path))
            text = ""
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"

            chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
            for i, chunk_content in enumerate(chunks):
                if is_worth_extracting(chunk_content):
                    docs.append({
                        "page_content": chunk_content,
                        "metadata": {"source_file": file_path.name, "chunk_index": i}
                    })
            logger.debug(f"  -> Kept {len([d for d in docs if d['metadata']['source_file'] == file_path.name])}"
                         f"/{len(chunks)} chunks for {file_path.name}")
        except Exception as e:
            logger.error(f"  -> Error processing {file_path}: {e}")

    return docs


# --- Batched + async extraction ---
def _build_batch_prompt(batch_chunks: list[str]) -> str:
    """Combine multiple chunk texts into a single prompt."""
    parts = []
    for i, text in enumerate(batch_chunks, 1):
        parts.append(f"[Chunk {i}]:\n{text}")
    return "\n---\n".join(parts)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    reraise=True,
)
async def _extract_batch_async(llm_with_tools, batch_texts: list[str]) -> list[Relationship]:
    """Call the LLM once for a batch of chunks, with retry on failure."""
    combined = _build_batch_prompt(batch_texts)
    messages = [
        ("system", EXTRACTION_PROMPT),
        ("human", f"Extract relationships from the following chunks:\n\n{combined}")
    ]
    try:
        response = await llm_with_tools.ainvoke(messages)
        return response.relationships
    except Exception as e:
        error_str = str(e)
        # Content-filter blocks are not retryable
        if "content_filter" in error_str or "ResponsibleAIPolicyViolation" in error_str:
            logger.error(f"  -> Content filter block (skipping batch): {error_str[:120]}")
            return []
        logger.warning(f"  -> Extraction error (will retry): {error_str[:120]}")
        raise


def _create_batches(chunks: list[dict], batch_size: int, fallback_size: int, max_chars: int):
    """Split chunks into batches, falling back to smaller size when text is too large."""
    batches = []
    i = 0
    while i < len(chunks):
        # Try full batch_size first
        end = min(i + batch_size, len(chunks))
        batch = chunks[i:end]
        total_chars = sum(len(c["page_content"]) for c in batch)

        if total_chars > max_chars and len(batch) > fallback_size:
            # Reduce to fallback size
            end = min(i + fallback_size, len(chunks))
            batch = chunks[i:end]

        batches.append(batch)
        i = end
    return batches


def add_relationships_to_graph(G, rels, canonical_map, seen_edges, source_file, source_text):
    for rel in rels:
        src_canon = normalize_entity_name(rel.source)
        tgt_canon = normalize_entity_name(rel.target)

        if not src_canon or not tgt_canon:
            continue

        if src_canon not in canonical_map:
            canonical_map[src_canon] = rel.source
        if tgt_canon not in canonical_map:
            canonical_map[tgt_canon] = rel.target

        display_src = canonical_map[src_canon]
        display_tgt = canonical_map[tgt_canon]

        edge_key = (src_canon, tgt_canon, rel.relation.lower())
        if edge_key in seen_edges:
            continue
        seen_edges.add(edge_key)

        G.add_node(display_src)
        G.add_node(display_tgt)
        G.add_edge(
            display_src,
            display_tgt,
            relation=rel.relation,
            source_file=source_file,
            source_text=source_text,
        )

async def build_graph_async(chunks: list[dict], llm_with_tools) -> nx.DiGraph:
    """Build the networkx graph using batched, concurrent LLM extraction."""
    G = nx.DiGraph()
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    batches = _create_batches(chunks, BATCH_SIZE, BATCH_FALLBACK, MAX_BATCH_CHARS)
    total_batches = len(batches)
    completed = 0

    canonical_map = {}
    seen_edges = set()

    logger.info(f"Created {total_batches} batches from {len(chunks)} chunks "
                f"(batch_size={BATCH_SIZE}, fallback={BATCH_FALLBACK})")

    async def process_batch(batch_idx: int, batch: list[dict]):
        nonlocal completed
        async with semaphore:
            texts = [c["page_content"] for c in batch]
            rels = await _extract_batch_async(llm_with_tools, texts)
            completed += 1
            if completed % 5 == 0 or completed == total_batches:
                logger.info(f"Batch progress: {completed}/{total_batches}")
            return rels, batch

    tasks = [process_batch(i, b) for i, b in enumerate(batches)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for result in results:
        if isinstance(result, Exception):
            logger.error(f"  -> Batch failed permanently: {result}")
            continue
        rels, batch = result
        source_file = batch[0]["metadata"].get("source_file", "unknown")
        batch_text_combined = "\n...\n".join(c["page_content"] for c in batch)
        add_relationships_to_graph(G, rels, canonical_map, seen_edges, source_file, batch_text_combined)

    return G


async def async_main():
    if not CONTEXT_DIR_PATH.exists():
        logger.error(f"Error: Directory '{CONTEXT_DIR_PATH}' does not exist.")
        return

    # 1. Setup LLM
    logger.info("Initializing LLM from core.llm...")
    llm = get_llm(temperature=0.0, caller_name="build_knowledge_graph")

    try:
        llm_with_tools = llm.with_structured_output(
            ExtractedGraphData,
            method="json_schema",
            include_raw=False
        )
    except Exception as e:
        logger.error(f"Failed to bind structured output to LLM. Error: {e}")
        return

    # 2. Process documents
    logger.info("\n--- Phase 1: Parsing and Chunking ---")
    chunks = process_documents(CONTEXT_DIR_PATH)

    if not chunks:
        logger.warning("No document chunks to process. Exiting.")
        return

    logger.info(f"Total chunks to process: {len(chunks)}")

    # 3. Build Graph (async + concurrent)
    logger.info("\n--- Phase 2: Extraction and Graph Building ---")
    G = await build_graph_async(chunks, llm_with_tools)

    logger.info("\n--- Phase 3: Verification and Saving ---")
    logger.info(f"Graph nodes: {G.number_of_nodes()}")
    logger.info(f"Graph edges: {G.number_of_edges()}")

    if G.number_of_nodes() > 0:
        nx.write_graphml(G, GRAPH_OUTPUT_FILE)
        logger.info(f"Successfully saved Knowledge Graph to {GRAPH_OUTPUT_FILE}")
    else:
        logger.warning("Graph is empty, nothing to save.")


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
