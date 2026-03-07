# Multi-Agent CAT Prep System with Ollama & LangGraph

A comprehensive CAT (Common Admission Test) preparation assistant powered by local LLMs through Ollama, featuring specialized agents for study planning, practice questions, and performance feedback.

## 🎯 Features

### Multi-Agent Architecture
- **Router Agent**: Intelligently routes queries to specialist agents
- **Study Plan Agent**: Creates personalized 5-week study plans with daily breakdowns
- **Practice Questions Agent**: Generates targeted practice questions with answer keys
- **Feedback/Review Agent**: Analyzes mock tests and provides actionable improvement strategies

### Optimized for 8B+ Models
- Sliding window memory management
- Compressed conversation summaries
- 100% Knowledge Graph-driven retrieval (Graph RAG)
- Efficient token usage

### Conversational Persistence
- Each agent maintains its own conversation context
- Multi-turn interactions until user exits with "bye", "exit", or "thanks"
- Seamless transitions between agents

## 🏗️ How It Works

### 1. Document Loading & Knowledge Graph Extraction
- Scans `context/` folder for CAT prep PDFs
- Chunks documents into text segments
- `scripts/build_knowledge_graph.py` extracts entities and embeds the raw text snippets directly onto graph edges (`source_text`)
- Resulting `.graphml` file stored in `data/` acts as the sole context database (no vector databases used)

### 2. Query Routing Flow
```
User Query
    ↓
[Router Agent] ← Analyzes intent
    ↓
    ├─→ Study Plan Agent (if asking for schedule/roadmap)
    ├─→ Practice Agent (if asking for questions/practice)
    └─→ Feedback Agent (if discussing mock tests/performance)
    ↓
[Specialist Agent] ← Maintains conversation context
    ↓
(User says "bye/exit/thanks")
    ↓
Back to Router
```

### 3. Agent Capabilities

**Study Plan Agent**
- Requests timeframe if not provided
- Creates detailed daily task breakdowns
- Balances QA, VA/RC, LR, and DI sections
- Allows plan refinement through conversation

**Practice Questions Agent**
- Generates questions from your uploaded materials
- Default set: 1 VA/RC passage, 2 QA, 1 LR, 1 DI
- Respects focus areas (e.g., "only QA questions")
- Matches time constraints (e.g., "15 minutes of practice")
- Provides detailed answer keys with step-by-step solutions
- Remembers current question set for answer key requests
- Compresses older practice sessions to save memory

**Feedback/Review Agent**
- Analyzes mock test performance
- Identifies weak areas and patterns
- Provides targeted improvement strategies
- Suggests practice topics based on weaknesses
- Multi-turn conversation support for deep analysis

## 🚀 Setup and Run

### 1. Prerequisites

**Install Ollama**
- Download from [ollama.ai](https://ollama.ai)
- Start the Ollama service

**Pull a Model**
```bash
# Recommended for testing
ollama pull llama3.1:8b    # 8B model (balanced speed/quality)

# Alternative options
ollama pull mistral:latest  # Recent 7B-8B class model
ollama pull llama3:8b       # Standard 8B model
```

### 2. Install Python Dependencies

```bash
python -m venv venv
venv\Scripts\activate
pip install uv
uv pip install -r pyproject.toml
```

**Package purposes:**
- `langgraph`: Multi-agent orchestration framework
- `langchain` & `langchain_community`: Framework orchestration, parsing, and LLM wrappers
- `langchain_ollama`: Ollama integration for LangChain
- `networkx`: Knowledge Graph structure, traversal, and centrality analysis
- `pypdf`: PDF document parsing
- `streamlit`: Frontend web interface framework
- `pyvis`: Interactive Knowledge Graph visualization (force-directed, drag/zoom)

### 3. Prepare Your Documents

```bash
# Create directories
mkdir context data logs

# Add your CAT prep materials to context/
# - Previous year question papers (PDFs)
# - Study materials
# - Mock test questions

# Build Knowledge Graph (Required First Step)
python scripts/build_knowledge_graph.py
```

**Document Tips:**
- Organize by topic (QA, VA/RC, LR, DI) for better retrieval
- Include answer keys in the same PDFs
- Use clear, structured PDFs for best results

### 4. Configure the Model (Optional)

Edit `core/config.py`:
```python
LOCAL_LLM_MODEL = "llama3.1:8b"  # Change to your preferred model
```

### 5. Run the System

```bash
python main_ui.py
# or: streamlit run main_ui.py
```

- Launches the **Streamlit** dashboard with side-by-side Chat and interactive Knowledge Graph visualization.
- The Knowledge Graph panel features:
  - **Graph Stats**: Node count, edge count, and predicate count.
  - **Triplet Filters**: Filter the graph by Subject, Predicate, or Object.
  - **Node Search**: Highlight a specific node with a gold border.
  - **Interactive PyVis Graph**: Force-directed physics layout with drag, zoom, and tooltips.
  - **Triple Accumulation**: Graph grows across conversation turns, with a Clear button to reset.
- Instantly loads the knowledge graph from `data/knowledge_graph.graphml`
- **Note:** The UI can start without Ollama running, but Ollama must be active when sending queries.

### 6. Logs & Debugging
- Main application logs: `logs/app.log`
- Graph builder logs: `logs/build_knowledge_graph.log`

## 💬 Usage Examples

### Study Plan Creation
```
You: I need a study plan for CAT
Router: [Routes to Study Plan Agent]
Agent: What timeframe works for you? (e.g., 5 weeks, 3 months)
You: 5 weeks
Agent: [Generates detailed 5-week plan with daily tasks]
You: Can you add more focus on QA?
Agent: [Modifies plan with increased QA emphasis]
You: Thanks!
Agent: [Exits back to main menu]
```

### Practice Questions
```
You: Give me 15 minutes of QA practice
Router: [Routes to Practice Agent]
Agent: [Generates QA questions with expected times]

Question 1: [Number Systems problem] (Expected: 3 mins)
Question 2: [Algebra problem] (Expected: 4 mins)
...

You: Show me the answer key
Agent: [Provides detailed solutions]
You: Give me different questions on the same topics
Agent: [Generates new QA questions]
You: Exit
Agent: [Exits back to main menu]
```

### Mock Test Review
```
You: I want feedback on my mock test
Router: [Routes to Feedback Agent]
Agent: Please share your mock test performance details.
You: I scored 60/100. Weak in QA (20/40) and strong in VA/RC (35/40)
Agent: [Analyzes performance, identifies weak areas]
You: Give me practice questions for my weak topics
Agent: [Suggests targeted practice from QA topics you struggled with]
You: Done
Agent: [Exits back to main menu]
```

## 🔧 Commands

- **`rebuild`**: Re-index all PDFs in context folder (use after adding new documents)
- **`quit`**: Exit the entire system (only works at main menu)
- **`bye`/`exit`/`thanks`**: Exit current agent and return to main menu

## 🧠 Memory Management (8B+ Optimization)

### Sliding Window Approach
- **Last 2-3 exchanges**: Kept in full detail
- **Older exchanges**: Compressed into brief summaries
- **Current questions**: Always preserved for answer key requests

### Why This Matters
- 8B+ models: ~16K-32K+ context window recommended
- Long conversations would overflow context
- Compression maintains continuity without memory issues

### Example Memory State
```
Previous Summary: "User completed 3 practice sessions on QA and VA/RC"

Recent History:
User: Give me 5 LR questions
Assistant: [Generates 5 LR questions with times]
User: Show answer key
Assistant: [Current - kept in memory for reference]
```

## 🎨 Customization

### Modify Prompts
Edit the prompt templates in `agents/prompts.py`:
- `ROUTER_PROMPT`
- `STUDY_PLAN_PROMPT`
- `PRACTICE_QUESTIONS_PROMPT`
- `FEEDBACK_PROMPT`

### Adjust Retrieval
```python
# In core/knowledge_graph.py - Change traversal limits
def retrieve_graph_context(query, max_seed_nodes=10, max_triples=30, hops=2)
```

### Modify Memory Limits
```python
# In core/utils.py - Adjust conversation history depth
def format_conversation_history(history, max_turns: int = 3):  # Change max_turns
```

### Change Exit Keywords
```python
# In core/utils.py - Add custom exit phrases
exit_keywords = ["bye", "exit", "quit", "thanks", "thank you", "done", "cancel"]
```

## 🐛 Troubleshooting

### "Knowledge graph not found"
- **Cause**: Graph builder script hasn't been run
- **Solution**: Run `python scripts/build_knowledge_graph.py` to populate `data/`

### "Ollama connection error"
- **Cause**: Ollama service not running
- **Solution**: Start Ollama (`ollama serve` or start the app)

### Agent gives irrelevant answers
- **Cause**: Documents don't contain relevant information
- **Solution**: 
  1. Add more comprehensive study materials
  2. Use `rebuild` command to re-index
  3. Try rephrasing your question

  3. Ensure you are using an 8B class model or larger

### Questions don't match my request
- **Cause**: Retrieval not finding right documents
- **Solution**:
  1. Be more specific (e.g., "QA number systems questions")
  2. Check if the graph builder successfully captured those topics
  3. Organize PDFs by topic

## 📚 Extending the System

### Adding Speech Support

**Speech-to-Text (STT):**
```python
import speech_recognition as sr

def speech_to_text():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        audio = recognizer.listen(source)
        return recognizer.recognize_google(audio)

# Replace input() with:
user_input = speech_to_text()
```

**Text-to-Speech (TTS):**
```python
import pyttsx3

def text_to_speech(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# After displaying answer:
text_to_speech(result["generation"])
```

**Recommended Libraries:**
- **STT**: `whisper.cpp` (best quality, local), `speech_recognition` (simple)
- **TTS**: `pyttsx3` (offline), `gTTS` (online but easy)

### Building a Web UI

**1. Create FastAPI Backend:**
```python
from fastapi import FastAPI, WebSocket
import uvicorn

app_api = FastAPI()

@app_api.post("/chat")
async def chat_endpoint(query: str, agent_type: str = None):
    # Use existing router and agent logic
    result = router_app.invoke({"question": query})
    # ... handle agent routing
    return {"response": final_answer}

uvicorn.run(app_api, host="0.0.0.0", port=8000)
```

**2. Frontend Options:**
- **React/Next.js**: Modern SPA with real-time updates
- **Streamlit**: Quick prototyping, Python-based
- **HTML/JavaScript**: Simple static frontend

**3. WebSocket for Real-Time Chat:**
```python
@app_api.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        message = await websocket.receive_text()
        response = process_with_agent(message)
        await websocket.send_text(response)
```

### Adding More Agents

**Create New Specialist Agent:**
```python
# 1. Define state
class NewAgentState(TypedDict):
    question: str
    documents: List[str]
    generation: str
    conversation_history: List[Dict[str, str]]
    should_exit: bool

# 2. Create prompt
NEW_AGENT_PROMPT = """Your custom prompt here..."""

# 3. Add nodes (retrieve, generate, check_exit)
# 4. Build graph
# 5. Add routing logic in ROUTER_PROMPT
```

## 🔒 Important Notes

### Offline & Private
- **All processing happens locally** - no data sent to cloud
- Your study materials stay on your machine
- Ideal for sensitive or proprietary content

### Model Selection Trade-offs
| Model | Speed | Quality | Memory | Best For |
|-------|-------|---------|--------|----------|
| 4B | ⚡ Fast | ✓ Good | 4-8GB | Quick practice, testing |
| 7B | 🐢 Slower | ✓✓ Better | 8-16GB | Detailed analysis, complex queries |
| 13B+ | 🐌 Slowest | ✓✓✓ Best | 16GB+ | Production, critical accuracy |

### Context Window Limitations
- 4B models: Keep conversations under 10 exchanges
- 7B models: Can handle 20+ exchanges comfortably
- System auto-compresses when approaching limits

## 📄 License

MIT License - Feel free to modify and extend for your needs!

## 🤝 Contributing

Issues and pull requests welcome! Focus areas:
- Better weak area extraction in Feedback Agent
- Metadata tagging for improved question retrieval
- Alternative memory compression strategies
- UI/UX improvements

## ⭐ Credits

Built with:
- [Ollama](https://ollama.ai) - Local LLM runtime
- [LangChain](https://langchain.com) - LLM application framework
- [LangGraph](https://langchain-ai.github.io/langgraph/) - Agent orchestration
- [NetworkX](https://networkx.org/) - Knowledge Graph structure & centrality
- [PyVis](https://pyvis.readthedocs.io/) - Interactive graph visualization
- [Streamlit](https://streamlit.io/) - Web UI Framework

---

**Happy CAT Prep! 📚✨**