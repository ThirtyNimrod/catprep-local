# Enhancing Agentic Workflow with Knowledge Graphs (GraphRAG)

While the current FAISS-based vector retrieval provides solid semantic search capabilities, upgrading the system to incorporate **Knowledge Graphs (GraphRAG)** is the optimal next step for an advanced, educational agentic structure.

## 1. Why Knowledge Graphs?

In the context of CAT (Common Admission Test) preparation, knowledge is deeply interconnected. A user struggling with "Time, Speed, and Distance" might actually have foundational weaknesses in "Fractions and Percentages".
- **Vector Stores** find text that is *semantically similar* to the query.
- **Knowledge Graphs** find concepts that are *logically related* to the query, explicitly mapping prerequisite dependencies, topics, formulas, and historical user mastery.

## 2. Proposed System Enhancements

### A. Graph Representation of the CAT Syllabus
Instead of just chunking PDFs, we extract entities (Topics, Formulas, Concepts) and relationships (is_prerequisite_for, relates_to, sub_topic_of) from the study materials using libraries like `NetworkX` or graph databases like `Neo4j` / `FalkorDB`.

*Example Schema:*
`[Basic Algebra] -> (is_prerequisite_for) -> [Quadratic Equations] -> (sub_topic_of) -> [Quantitative Aptitude]`

### B. User Knowledge Tracking (State Graph)
The LangGraph `AgentState` can be enhanced to maintain a persistent user node in the graph. 
- When the `Feedback Agent` analyzes a mock test, it updates the edges connecting the `[User]` node to `[Topic]` nodes with a `mastery_score` property.
- When the `Practice Agent` needs to generate questions, it queries the graph for topics where `[User] - (has_mastery {score < 0.5}) -> [Topic]`.

### C. Multi-hop Reasoning during Routing
The `Router Agent` can use the Knowledge Graph to better contextualize ambiguous queries. If a user asks, "Why did I get question 4 wrong?", the system retrieves the knowledge subgraph surrounding the entities in Question 4, allowing the LLM to provide a highly specific, logical explanation referencing exact formulas and concepts.

## 3. Data Ingestion Pipeline (From Raw PDFs to Graph)

You **cannot** directly dump the raw PDFs from the `context/` folder (like the Mock tests or Puzzles) into a Knowledge Graph. Raw PDFs contain unstructured text, whereas a Knowledge Graph requires **structured entities and relationships**. 

Directly ingesting raw text will just create massive, disconnected text nodes, defeating the purpose of a graph and offering no advantage over the current FAISS vector store.

To achieve best results, the data must be heavily processed and structured.

### Proposed Processing Steps (ETL Pipeline):

1. **Text Extraction & Cleaning**:
   - Parse the PDFs (using `PyMuPDF` or `pdfplumber`).
   - Clean the text to remove headers, footers, and fix formatting issues, especially for mathematical formulas using tools like `Nougat` or `Mathpix`.

2. **Entity & Relationship Extraction (LLM-Powered)**:
   - Feed the cleaned text chunks to an LLM with a strict JSON schema prompt to extract specific entities.
   - *Model Note:* This process demands high accuracy. While local 8B models (via Ollama) can be used with heavy schema-constraining and few-shot examples, toggling the `llm_provider` to `AzureOpenAI` in the `.env` file during the ingestion phase yields significantly more reliable JSON triplet outputs without hallucinated relationships.
   - **Target Entities**: `[Topic]`, `[Formula]`, `[Question]`, `[Solution]`.
   - **Target Relationships**:
     - `[Question] -> (requires_knowledge_of) -> [Topic]`
     - `[Question] -> (solved_using) -> [Formula]`
     - `[Topic] -> (is_prerequisite_for) -> [Topic]`

3. **Graph Construction & Validation**:
   - Insert these extracted structured triplets (Entity-Relationship-Entity) into the Graph Database (e.g., Neo4j).
   - Resolve entity duplication (e.g., merging "Time & Work" and "Time and Work" into a single node).

4. **Hybrid Indexing**:
   - The actual full-text content of a question or solution should still be embedded as a vector and stored *as a property* on the graph node. This allows for vector similarity search to find the entry point node in the graph, followed by graph traversal.

## 4. Implementation Steps

1. **Graph Construction Layer**: Add a new utility `core/knowledge_graph.py` to handle the ETL pipeline described above.
2. **Hybrid Retrieval Strategy**: Modify `core/vector_store.retrieve_documents()` to perform **Hybrid Search**:
   - FAISS for dense vector similarity.
   - Graph Traversal for extracting contextual subgraphs (using LangChain's `GraphQAChain` or custom Cypher queries).
3. **Agent State Updates**: Expand `core/state.py` to allow specialists to read/write properties directly to the user's sub-graph (e.g., updating mastery levels).
4. **Specialist Overhaul**: Update the `Feedback Agent` and `Study Plan Agent` to utilize the graph topology for generating chronologically logical study schedules that respect topic prerequisites.

## 5. Proof of Concept (PoC)

To validate the ETL pipeline, a PoC was built to process raw PDFs into a basic NetworkX Knowledge Graph.
- **Pre-Processor**: `agents/pre_processor.py` was created using `langchain_community` loaders to parse raw PDFs from the `context/` directory. An output example can be seen in `tests/parsed_chunks_output.json`.
- **Entity Extraction Scheme**: `examples/extraction_example.json` defines the strict JSON schema required for passing these text chunks to an 8B LLM to prevent hallucinations.
- **Graph Builder**: `agents/graph_builder.py` ingests the JSON schema and utilizes `networkx` to build and visualize the relationships. A generated visual graph is available at `examples/graph_poc.png`.
