import os
import shutil
from typing import List, Dict, TypedDict, Literal, Optional
from datetime import datetime

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from langgraph.graph import StateGraph, END

# --- Configuration ---
CONTEXT_DIR = "context"
VECTORSTORE_PATH = "chroma_db"
LOCAL_LLM_MODEL = "granite4:tiny-h"

# ============================================================
# STATE DEFINITIONS
# ============================================================

class RouterState(TypedDict):
    """State for the Starter/Router Agent"""
    question: str
    next_agent: str  # "study_plan", "practice", "feedback", or "unknown"

class StudyPlanState(TypedDict):
    """State for Study Plan Agent"""
    question: str
    documents: List[str]
    generation: str
    conversation_history: List[Dict[str, str]]  # [{"role": "user/assistant", "content": "..."}]
    timeframe: Optional[str]
    should_exit: bool

class PracticeQuestionsState(TypedDict):
    """State for Practice Questions Agent"""
    question: str
    documents: List[str]
    generation: str
    conversation_history: List[Dict[str, str]]
    current_questions: str  # Store the most recent question set
    previous_summary: str  # Compressed summary of older interactions
    focus_area: Optional[str]  # QA, VA/RC, LR, DI
    timeframe: Optional[str]
    should_exit: bool

class FeedbackState(TypedDict):
    """State for Feedback/Review Agent - Option B (multi-turn)"""
    question: str
    documents: List[str]
    generation: str
    conversation_history: List[Dict[str, str]]
    mock_test_analysis: str  # Initial analysis summary
    weak_areas: List[str]  # Identified weak topics
    should_exit: bool

# ============================================================
# PROMPTS - OPTIMIZED FOR 4B/7B MODELS
# ============================================================

ROUTER_PROMPT = """You are a routing assistant for a CAT exam prep system.

Analyze the user's query and determine which agent should handle it:
- "study_plan": User wants a study plan, schedule, or preparation roadmap
- "practice": User wants practice questions, wants to solve problems, or requests answer keys
- "feedback": User wants mock test review, performance analysis, or discusses their test results
- "unknown": Query doesn't fit above categories

User Query: {question}

Output ONLY one word: study_plan, practice, feedback, or unknown"""

STUDY_PLAN_PROMPT = """You are a CAT Prep Study Plan Expert.

CONTEXT FROM DOCUMENTS:
{context}

CONVERSATION HISTORY:
{history}

CURRENT REQUEST: {question}

RULES:
1. If timeframe not specified, ask: "What timeframe works for you? (e.g., 5 weeks, 3 months)"
2. Create detailed daily breakdown with specific tasks
3. Balance QA (Maths), VA/RC (English), LR, and DI
4. Include time allocations (e.g., "30 mins QA practice")
5. Be concise - avoid long explanations
6. If user wants to edit, modify the existing plan
7. Use information from context documents when relevant

Response:"""

PRACTICE_QUESTIONS_PROMPT = """You are a CAT Prep Practice Question Generator.

CONTEXT FROM DOCUMENTS:
{context}

PREVIOUS PRACTICE SUMMARY:
{previous_summary}

CURRENT QUESTIONS IN SESSION:
{current_questions}

CONVERSATION HISTORY (last 3 exchanges):
{history}

CURRENT REQUEST: {question}

RULES:
1. ALWAYS include for each question:
   - Question text
   - Expected completion time
   - Detailed answer key with solutions

2. Default set (if no specification):
   - 1 VA/RC passage (2-3 questions)
   - 2 QA questions
   - 1 LR set (1-2 questions)
   - 1 DI set (1-2 questions)

3. Respect user's focus area: If they say "QA only", provide ONLY QA questions
4. Match timeframe: "15 mins practice" = questions totaling ~15 mins
5. ALL questions MUST come from the provided context
6. If asked for answer key, provide solutions for current_questions
7. If asked for different questions, generate new ones from context
8. Be concise and clear

Response:"""

FEEDBACK_PROMPT = """You are a CAT Prep Performance Analyst.

CONTEXT FROM DOCUMENTS:
{context}

MOCK TEST ANALYSIS (if available):
{mock_analysis}

IDENTIFIED WEAK AREAS:
{weak_areas}

CONVERSATION HISTORY:
{history}

CURRENT REQUEST: {question}

RULES:
1. Analyze mock test performance: accuracy, time management, topic patterns
2. Identify weak areas and frequently tested topics
3. Provide actionable improvement strategies
4. If user asks for practice on weak areas, suggest specific topics from context
5. Be encouraging and constructive
6. Keep responses focused and clear
7. Use data from context documents for topic-specific advice

Response:"""

# ============================================================
# VECTOR STORE SETUP
# ============================================================

def setup_vector_store(force_rebuild: bool = False):
    """Creates or loads ChromaDB vector store with metadata for filtering"""
    print("Initializing vector store...")
    embeddings = OllamaEmbeddings(model=LOCAL_LLM_MODEL)

    if os.path.exists(VECTORSTORE_PATH) and not force_rebuild:
        print(f"Loading existing vector store from {VECTORSTORE_PATH}...")
        vector_store = Chroma(
            persist_directory=VECTORSTORE_PATH,
            embedding_function=embeddings
        )
        return vector_store
    
    print(f"Creating new vector store. Rebuilding={force_rebuild}")
    if os.path.exists(VECTORSTORE_PATH):
        shutil.rmtree(VECTORSTORE_PATH)

    if not os.path.exists(CONTEXT_DIR) or not os.listdir(CONTEXT_DIR):
        print(f"Error: '{CONTEXT_DIR}' directory is empty or missing.")
        os.makedirs(CONTEXT_DIR, exist_ok=True)
        return None

    print(f"Loading documents from {CONTEXT_DIR}...")
    loader = PyPDFDirectoryLoader(CONTEXT_DIR)
    docs = loader.load()

    if not docs:
        print("No PDF documents found.")
        return None

    print(f"Loaded {len(docs)} document pages.")
    
    # Split with metadata preservation
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,  # Smaller chunks for 4B model
        chunk_overlap=150
    )
    splits = text_splitter.split_documents(docs)
    print(f"Split into {len(splits)} chunks.")

    print("Creating ChromaDB vector store...")
    vector_store = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=VECTORSTORE_PATH
    )
    
    print(f"Vector store saved to {VECTORSTORE_PATH}.")
    return vector_store

# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def check_exit_intent(text: str) -> bool:
    """Check if user wants to exit current agent"""
    exit_keywords = ["bye", "exit", "quit", "thanks", "thank you", "done"]
    return any(keyword in text.lower() for keyword in exit_keywords)

def format_conversation_history(history: List[Dict[str, str]], max_turns: int = 3) -> str:
    """Format conversation history for context (last N turns only)"""
    if not history:
        return "No previous conversation."
    
    recent = history[-max_turns*2:]  # Get last N exchanges (user+assistant pairs)
    formatted = []
    for msg in recent:
        role = msg["role"].capitalize()
        formatted.append(f"{role}: {msg['content']}")
    return "\n".join(formatted)

def compress_practice_summary(history: List[Dict[str, str]]) -> str:
    """Compress old practice sessions into a brief summary"""
    if len(history) <= 6:  # Keep full history if short
        return ""
    
    # Simple compression: just note that previous practice happened
    return "User has completed previous practice sessions in this conversation."

# ============================================================
# ROUTER AGENT
# ============================================================

def route_query(state: RouterState):
    """Determines which specialist agent should handle the query"""
    print("---ROUTER AGENT---")
    question = state["question"]
    
    llm = ChatOllama(model=LOCAL_LLM_MODEL, temperature=0)
    prompt = PromptTemplate(template=ROUTER_PROMPT, input_variables=["question"])
    chain = prompt | llm | StrOutputParser()
    
    result = chain.invoke({"question": question}).strip().lower()
    print(f"Routing to: {result}")
    
    # Validate output
    valid_routes = ["study_plan", "practice", "feedback", "unknown"]
    next_agent = result if result in valid_routes else "unknown"
    
    return {"next_agent": next_agent}

# ============================================================
# STUDY PLAN AGENT
# ============================================================

def study_plan_retrieve(state: StudyPlanState):
    """Retrieve relevant documents for study planning"""
    print("---STUDY PLAN: RETRIEVE---")
    question = state["question"]
    
    documents = vector_store.similarity_search(question, k=3)
    doc_snippets = [doc.page_content for doc in documents]
    
    return {"documents": doc_snippets}

def study_plan_generate(state: StudyPlanState):
    """Generate study plan response"""
    print("---STUDY PLAN: GENERATE---")
    
    question = state["question"]
    documents = state.get("documents", [])
    history = state.get("conversation_history", [])
    
    context_str = "\n\n---\n\n".join(documents) if documents else "No specific context available."
    history_str = format_conversation_history(history)
    
    llm = ChatOllama(model=LOCAL_LLM_MODEL, temperature=0.3)
    prompt = PromptTemplate(
        template=STUDY_PLAN_PROMPT,
        input_variables=["question", "context", "history"]
    )
    chain = prompt | llm | StrOutputParser()
    
    generation = chain.invoke({
        "question": question,
        "context": context_str,
        "history": history_str
    })
    
    # Update conversation history
    updated_history = history + [
        {"role": "user", "content": question},
        {"role": "assistant", "content": generation}
    ]
    
    # Check for exit intent
    should_exit = check_exit_intent(question)
    
    return {
        "generation": generation,
        "conversation_history": updated_history,
        "should_exit": should_exit
    }

def study_plan_check_exit(state: StudyPlanState) -> str:
    """Check if user wants to exit Study Plan agent"""
    return "exit" if state.get("should_exit", False) else "continue"

# ============================================================
# PRACTICE QUESTIONS AGENT
# ============================================================

def practice_retrieve(state: PracticeQuestionsState):
    """Retrieve practice questions based on user request"""
    print("---PRACTICE: RETRIEVE---")
    question = state["question"]
    focus_area = state.get("focus_area")
    
    # Build search query
    search_query = question
    if focus_area:
        search_query = f"{focus_area} {question}"
    
    # Retrieve more documents for practice questions
    documents = vector_store.similarity_search(search_query, k=5)
    doc_snippets = [doc.page_content for doc in documents]
    
    return {"documents": doc_snippets}

def practice_generate(state: PracticeQuestionsState):
    """Generate practice questions or answer keys"""
    print("---PRACTICE: GENERATE---")
    
    question = state["question"]
    documents = state.get("documents", [])
    history = state.get("conversation_history", [])
    current_questions = state.get("current_questions", "No questions in current session.")
    previous_summary = state.get("previous_summary", "")
    
    context_str = "\n\n---\n\n".join(documents) if documents else "No context available."
    history_str = format_conversation_history(history, max_turns=2)  # Only last 2 turns for 4B model
    
    llm = ChatOllama(model=LOCAL_LLM_MODEL, temperature=0.3)
    prompt = PromptTemplate(
        template=PRACTICE_QUESTIONS_PROMPT,
        input_variables=["question", "context", "history", "current_questions", "previous_summary"]
    )
    chain = prompt | llm | StrOutputParser()
    
    generation = chain.invoke({
        "question": question,
        "context": context_str,
        "history": history_str,
        "current_questions": current_questions,
        "previous_summary": previous_summary
    })
    
    # Update state
    updated_history = history + [
        {"role": "user", "content": question},
        {"role": "assistant", "content": generation}
    ]
    
    # Compress old history if too long
    new_previous_summary = previous_summary
    if len(updated_history) > 8:
        new_previous_summary = compress_practice_summary(updated_history[:-6])
        updated_history = updated_history[-6:]  # Keep only last 3 exchanges
    
    # Update current questions if new questions were generated
    new_current_questions = current_questions
    if "answer key" not in question.lower() and "solution" not in question.lower():
        # This is likely a new question request
        new_current_questions = generation
    
    should_exit = check_exit_intent(question)
    
    return {
        "generation": generation,
        "conversation_history": updated_history,
        "current_questions": new_current_questions,
        "previous_summary": new_previous_summary,
        "should_exit": should_exit
    }

def practice_check_exit(state: PracticeQuestionsState) -> str:
    """Check if user wants to exit Practice agent"""
    return "exit" if state.get("should_exit", False) else "continue"

# ============================================================
# FEEDBACK/REVIEW AGENT
# ============================================================

def feedback_retrieve(state: FeedbackState):
    """Retrieve relevant context for feedback analysis"""
    print("---FEEDBACK: RETRIEVE---")
    question = state["question"]
    weak_areas = state.get("weak_areas", [])
    
    # Prioritize weak areas in search
    search_query = question
    if weak_areas:
        search_query = f"{' '.join(weak_areas)} {question}"
    
    documents = vector_store.similarity_search(search_query, k=4)
    doc_snippets = [doc.page_content for doc in documents]
    
    return {"documents": doc_snippets}

def feedback_generate(state: FeedbackState):
    """Generate feedback and analysis"""
    print("---FEEDBACK: GENERATE---")
    
    question = state["question"]
    documents = state.get("documents", [])
    history = state.get("conversation_history", [])
    mock_analysis = state.get("mock_test_analysis", "No mock test analyzed yet.")
    weak_areas = state.get("weak_areas", [])
    
    context_str = "\n\n---\n\n".join(documents) if documents else "No context available."
    history_str = format_conversation_history(history, max_turns=3)
    weak_areas_str = ", ".join(weak_areas) if weak_areas else "Not yet identified."
    
    llm = ChatOllama(model=LOCAL_LLM_MODEL, temperature=0.3)
    prompt = PromptTemplate(
        template=FEEDBACK_PROMPT,
        input_variables=["question", "context", "history", "mock_analysis", "weak_areas"]
    )
    chain = prompt | llm | StrOutputParser()
    
    generation = chain.invoke({
        "question": question,
        "context": context_str,
        "history": history_str,
        "mock_analysis": mock_analysis,
        "weak_areas": weak_areas_str
    })
    
    # Update conversation history
    updated_history = history + [
        {"role": "user", "content": question},
        {"role": "assistant", "content": generation}
    ]
    
    # Extract weak areas if this is initial analysis
    # (Simple keyword extraction - can be improved)
    new_weak_areas = weak_areas
    if not weak_areas and "weak" in question.lower():
        # This is a simplification; in production, use better NER/extraction
        new_weak_areas = ["QA", "VA/RC"]  # Placeholder
    
    # Update mock analysis summary if first time
    new_mock_analysis = mock_analysis
    if mock_analysis == "No mock test analyzed yet." and len(history) == 0:
        new_mock_analysis = generation[:200] + "..."  # Store brief summary
    
    should_exit = check_exit_intent(question)
    
    return {
        "generation": generation,
        "conversation_history": updated_history,
        "mock_test_analysis": new_mock_analysis,
        "weak_areas": new_weak_areas,
        "should_exit": should_exit
    }

def feedback_check_exit(state: FeedbackState) -> str:
    """Check if user wants to exit Feedback agent"""
    return "exit" if state.get("should_exit", False) else "continue"

# ============================================================
# BUILD THE GRAPHS
# ============================================================

# Router Graph
router_graph = StateGraph(RouterState)
router_graph.add_node("route", route_query)
router_graph.set_entry_point("route")
router_graph.add_edge("route", END)
router_app = router_graph.compile()

# Study Plan Graph
study_plan_graph = StateGraph(StudyPlanState)
study_plan_graph.add_node("retrieve", study_plan_retrieve)
study_plan_graph.add_node("generate", study_plan_generate)
study_plan_graph.set_entry_point("retrieve")
study_plan_graph.add_edge("retrieve", "generate")
study_plan_graph.add_conditional_edges(
    "generate",
    study_plan_check_exit,
    {"continue": "retrieve", "exit": END}
)
study_plan_app = study_plan_graph.compile()

# Practice Questions Graph
practice_graph = StateGraph(PracticeQuestionsState)
practice_graph.add_node("retrieve", practice_retrieve)
practice_graph.add_node("generate", practice_generate)
practice_graph.set_entry_point("retrieve")
practice_graph.add_edge("retrieve", "generate")
practice_graph.add_conditional_edges(
    "generate",
    practice_check_exit,
    {"continue": "retrieve", "exit": END}
)
practice_app = practice_graph.compile()

# Feedback Graph
feedback_graph = StateGraph(FeedbackState)
feedback_graph.add_node("retrieve", feedback_retrieve)
feedback_graph.add_node("generate", feedback_generate)
feedback_graph.set_entry_point("retrieve")
feedback_graph.add_edge("retrieve", "generate")
feedback_graph.add_conditional_edges(
    "generate",
    feedback_check_exit,
    {"continue": "retrieve", "exit": END}
)
feedback_app = feedback_graph.compile()

# ============================================================
# MAIN APPLICATION LOOP
# ============================================================

if __name__ == "__main__":
    print("🚀 Multi-Agent CAT Prep System Starting...")
    
    vector_store = setup_vector_store(force_rebuild=False)
    
    if vector_store is None:
        print("\n❌ Failed to initialize vector store. Exiting.")
        print(f"Please add PDFs to '{CONTEXT_DIR}' folder and restart.")
        exit(1)
    
    print("\n✅ System Ready!")
    print("=" * 60)
    print("Commands:")
    print("  'rebuild' - Re-index PDF documents")
    print("  'quit' - Exit the system")
    print("=" * 60)
    
    # Track current agent and its state
    current_agent = None
    agent_state = None
    
    while True:
        try:
            # Determine prompt based on current agent
            if current_agent:
                prompt_msg = f"\n[{current_agent.upper()} Agent] Your message: "
            else:
                prompt_msg = "\n💬 What would you like help with? "
            
            user_input = input(prompt_msg).strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ["quit", "exit"] and current_agent is None:
                print("\n👋 Goodbye! Good luck with your CAT prep!")
                break
            
            if user_input.lower() == "rebuild":
                print("\n🔄 Rebuilding vector store...")
                vector_store = setup_vector_store(force_rebuild=True)
                if vector_store:
                    print("✅ Rebuild complete.")
                else:
                    print("❌ Rebuild failed.")
                continue
            
            # If no active agent, route the query
            if current_agent is None:
                print("\n🤔 Analyzing your request...")
                router_result = router_app.invoke({"question": user_input})
                next_agent = router_result["next_agent"]
                
                if next_agent == "unknown":
                    print("\n❓ I'm not sure how to help with that.")
                    print("I can help with:")
                    print("  📚 Study plans")
                    print("  ✏️  Practice questions")
                    print("  📊 Mock test feedback")
                    continue
                
                # Initialize the appropriate agent
                current_agent = next_agent
                
                if current_agent == "study_plan":
                    print("\n📚 Entering Study Plan Agent...")
                    agent_state = {
                        "question": user_input,
                        "documents": [],
                        "generation": "",
                        "conversation_history": [],
                        "timeframe": None,
                        "should_exit": False
                    }
                    result = study_plan_app.invoke(agent_state)
                
                elif current_agent == "practice":
                    print("\n✏️  Entering Practice Questions Agent...")
                    agent_state = {
                        "question": user_input,
                        "documents": [],
                        "generation": "",
                        "conversation_history": [],
                        "current_questions": "",
                        "previous_summary": "",
                        "focus_area": None,
                        "timeframe": None,
                        "should_exit": False
                    }
                    result = practice_app.invoke(agent_state)
                
                elif current_agent == "feedback":
                    print("\n📊 Entering Feedback Agent...")
                    agent_state = {
                        "question": user_input,
                        "documents": [],
                        "generation": "",
                        "conversation_history": [],
                        "mock_test_analysis": "No mock test analyzed yet.",
                        "weak_areas": [],
                        "should_exit": False
                    }
                    result = feedback_app.invoke(agent_state)
                
                # Update state and display response
                agent_state = result
                print("\n" + "=" * 60)
                print(result["generation"])
                print("=" * 60)
                
                # Check if agent wants to exit immediately
                if result.get("should_exit", False):
                    print(f"\n✅ Exiting {current_agent} agent.")
                    current_agent = None
                    agent_state = None
            
            else:
                # Continue conversation with current agent
                agent_state["question"] = user_input
                agent_state["should_exit"] = False
                
                if current_agent == "study_plan":
                    result = study_plan_app.invoke(agent_state)
                elif current_agent == "practice":
                    result = practice_app.invoke(agent_state)
                elif current_agent == "feedback":
                    result = feedback_app.invoke(agent_state)
                
                agent_state = result
                print("\n" + "=" * 60)
                print(result["generation"])
                print("=" * 60)
                
                # Check exit condition
                if result.get("should_exit", False):
                    print(f"\n✅ Exiting {current_agent} agent. Back to main menu.")
                    current_agent = None
                    agent_state = None
        
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Resetting to main menu...")
            current_agent = None
            agent_state = None