# Code Architecture & Agent Implementation

This document provides a comprehensive overview of the **catprep-local** agentic architecture, current agent implementations, and system scope.

## 1. High-Level Architecture
The system is orchestrated using **LangGraph**, providing a cyclic, stateful, multi-agent workflow. Rather than relying on a single monolithic LLM call, the architecture is divided into distinct specialist "nodes" (agents) managed by a Router.

- **State Management**: The central state (`AgentState` defined in `core/state.py`) tracks the conversation `messages`, as well as specialized keys including `focus_area`, `current_questions`, `previous_summary`, `weak_areas`, and `mock_test_analysis`.
- **Memory**: It utilizes LangGraph's `MemorySaver` checkpointer for thread-based persistent conversation states (e.g., `THREAD_ID = "catprep_session_user_1"`).
- **Retrieval Augmented Generation (RAG)**: Uses FAISS (`vector_store.py`) to retrieve context from local PDF study materials (`context/` directory).
- **LLM Routing**: The system utilizes a dynamic `LLM_PROVIDER` loaded via `.env` in `core/config.py`. It supports both local lightweight models (`ChatOllama`) and premium cloud models (`AzureChatOpenAI`) depending on the `llm_provider` variable.
- **Frontend**: A local web dashboard built with **Streamlit** (`main_ui.py`) rendering chat and LangGraph execution traces side-by-side.

## 2. Agent Implementations & Scope

### 2.1 Router Agent (`agents/router.py`)
- **Role**: The Supervisor. It analyzes the user's latest query and conditionally routes the execution flow to the appropriate specialist agent.
- **Implementation**: Uses a zero-shot classification approach via `ROUTER_PROMPT`.
- **Output Routes**: `study_plan`, `practice`, `feedback`, or `unknown`.

### 2.2 Study Plan Agent (`agents/specialists/study_plan.py`)
- **Role**: Creates and manages customized study plans for the user.
- **Implementation**: 
  - Checks for exit intents.
  - Retrieves top 3 most relevant documents from the vector store (`k=3`).
  - Feeds the user history and context into `STUDY_PLAN_PROMPT` to generate a structured 5-week plan or daily breakdown.

### 2.3 Practice Agent (`agents/specialists/practice.py`)
- **Role**: Generates targeted practice questions and provides answer keys/solutions.
- **Implementation**:
  - Dynamically builds a search query using the state's `focus_area` combined with the user's question, retrieving top 5 documents (`k=5`).
  - Maintains state vectors like `current_questions` to remember active quiz sessions.
  - Implements **memory compression** (`compress_practice_summary`) when the conversation length exceeds 8 messages, optimizing token usage for smaller local LLMs.

### 2.4 Feedback Agent (`agents/specialists/feedback.py`)
- **Role**: Analyzes user performance, notably mock tests, to identify weak areas and suggest improvements.
- **Implementation**:
  - Uses the state's `weak_areas` array to refine vector retrieval (`k=4`).
  - Simulates maintaining ongoing analysis via the `mock_test_analysis` state key.
  - Defaults to analyzing crucial CAT domains (e.g., "QA", "VA/RC") if weak areas aren't explicitly provided but the user asks for weakness analysis.

## 3. Current Scope & Limitations
- **Hybrid Execution Environment**: The system is heavily optimized for token compression and local execution on 8B+ models via Ollama. It also transparently supports high-capacity models via Azure OpenAI, which can be toggled using the `llm_provider` environment variable in the `.env` file.
- **Vector-Only Relations**: Currently, semantic search is limited to flat dense vector retrieval (FAISS). Complex multi-hop reasoning (e.g., connecting a user's failure in "Logarithms" to a prerequisite weakness in "Exponents") is heavily dependent on the LLM's intrinsic knowledge rather than a structured knowledge base.
