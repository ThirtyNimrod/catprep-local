import os
import shutil
from typing import List, Dict, TypedDict

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import Chroma  # <-- Changed from FAISS
from langchain_ollama import OllamaEmbeddings, ChatOllama  # <-- Updated imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

from langgraph.graph import StateGraph, END

# --- Configuration ---
CONTEXT_DIR = "context"
VECTORSTORE_PATH = "chroma_db"  # <-- Changed to a new directory for Chroma
# Make sure you have an Ollama model pulled, e.g., "llama3" or "mistral"
LOCAL_LLM_MODEL = "granite4:tiny-h" 

# --- 1. Define the Agent's State ---
# This TypedDict defines the state that flows through the graph.
class AgentState(TypedDict):
    question: str               # The user's original question
    documents: List[str]        # A list of retrieved document snippets
    generation: str             # The LLM's final generated answer
    user_prompt_template: str   # Placeholder for the user's custom prompt

# --- 2. Setup the RAG Pipeline (Vector Store) ---
def setup_vector_store(force_rebuild: bool = False):
    """
    Creates or loads a ChromaDB vector store from PDFs in the CONTEXT_DIR.
    """
    print("Initializing vector store...")
    embeddings = OllamaEmbeddings(model=LOCAL_LLM_MODEL) # Define embeddings once

    if os.path.exists(VECTORSTORE_PATH) and not force_rebuild:
        print(f"Loading existing vector store from {VECTORSTORE_PATH}...")
        vector_store = Chroma(
            persist_directory=VECTORSTORE_PATH, 
            embedding_function=embeddings
        )
        return vector_store.as_retriever()
    
    print(f"Creating new vector store. Rebuilding={force_rebuild}")
    if os.path.exists(VECTORSTORE_PATH):
        print(f"Removing old vector store at {VECTORSTORE_PATH}...")
        shutil.rmtree(VECTORSTORE_PATH)

    if not os.path.exists(CONTEXT_DIR) or not os.listdir(CONTEXT_DIR):
        print(f"Error: The '{CONTEXT_DIR}' directory is empty or does not exist.")
        print("Please create it and add your PDF files.")
        os.makedirs(CONTEXT_DIR, exist_ok=True)
        return None

    print(f"Loading documents from {CONTEXT_DIR}...")
    loader = PyPDFDirectoryLoader(CONTEXT_DIR)
    docs = loader.load()

    if not docs:
        print("No PDF documents were found or loaded. Please check your files.")
        return None

    print(f"Loaded {len(docs)} document pages.")
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    print(f"Split documents into {len(splits)} chunks.")

    # Create embeddings and vector store
    print("Creating ChromaDB vector store... (this may take a moment)")
    # Chroma.from_documents automatically handles persistence
    vector_store = Chroma.from_documents(
        documents=splits, 
        embedding=embeddings,
        persist_directory=VECTORSTORE_PATH
    )
    
    print(f"Vector store saved to {VECTORSTORE_PATH}.")
    
    return vector_store.as_retriever()

# --- 3. Define the LangGraph Nodes ---
# These are the functions that will be the "steps" in our graph.

def retrieve_documents(state: AgentState):
    """
    Retrieves relevant documents from the vector store based on the user's question.
    """
    print("---NODE: RETRIEVE_DOCUMENTS---")
    question = state["question"]
    print(f"Retrieving documents for: {question}")
    
    # 'retriever' is injected into the graph's context later
    documents = retriever.invoke(question)
    
    # Format documents for clarity
    doc_snippets = [doc.page_content for doc in documents]
    
    print(f"Retrieved {len(doc_snippets)} document snippets.")
    return {"documents": doc_snippets}

def generate_answer(state: AgentState):
    """
    Generates an answer using the LLM, based on the question and retrieved documents.
    """
    print("---NODE: GENERATE_ANSWER---")
    question = state["question"]
    documents = state["documents"]
    user_prompt_template = state.get("user_prompt_template") # Get the custom prompt

    if not user_prompt_template:
        # Default prompt if the user doesn't provide one
        user_prompt_template = """
Act as a personal 'CAT Prep' expert, providing a personalized, flexible study experience for the user's CAT exam preparation.

Purpose and Goals:

* Deliver structured, detailed 5-week study plans that include a daily breakdown of tasks specific to the user's needs for the CAT exam (Common Admission Test).
* Generate targeted practice questions for Quantitative Aptitude (QA), Verbal Ability & Reading Comprehension (VA/RC), Logical Reasoning (LR), and Data Interpretation (DI), adjusting difficulty based on user progress.
* Provide constructive feedback on user performance, analyzing accuracy, time management, and areas for improvement.
* Review uploaded mock tests to identify question patterns, complexity, and frequently tested topics, tailoring future practice suggestions accordingly.

Behaviors and Rules:

1) Study Plan Assistance:
    a) Whenever requested, create a structured, detailed 5-week study plan with a daily breakdown of tasks.
    b) Unless otherwise specified by the user, prioritize Quantitative Aptitude (QA) for Maths, Verbal Ability & Reading Comprehension (VA/RC) for English, and a mix of Logical Reasoning (LR) and Data Interpretation (DI).
    c) Ensure the weekly breakdown includes specific tasks (e.g., '30 mins of Maths practice', '30 mins of English practice').

2) Question Assistance & Daily Practice:
    a) Maths Questions: Provide questions based on QA concepts (like algebra, arithmetic, geometry, number systems). Start with easier questions and gradually increase difficulty to medium and hard-level problems to improve speed and accuracy.
    b) English Questions: Provide Reading Comprehension (RC) passages, along with questions related to vocabulary (synonyms, antonyms, fill in the blanks), and grammar (subject-verb agreement, tenses, sentence correction).
    c) If the user specifies a focus (e.g., 'Focus on English'), adhere to that focus (e.g., provide an RC passage with related questions). If they request 'Focus on Maths,' provide QA problems. If 'Focus on LR/DI,' provide Logical Reasoning puzzles or Data Interpretation sets.
    d) When asked for questions (e.g., 'Give me a set of 5 questions on Number Systems' or 'Give me a passage with 5 RC questions'), fulfill the specific request.

3) Feedback and Mock Test Review:
    a) When asked for feedback, analyze performance in practice questions, providing specific suggestions for improvement.
    b) For mock test reviews, analyze the attached tests to understand question types, identify patterns (topic frequency, complexity), and provide context-specific feedback to align preparation with the exam's structure and difficulty.

Overall Tone & Style:
* Keep responses concise and clear.
* Provide step-by-step solutions for all practice questions.
* Be encouraging, especially if the user is struggling, offering tips to help them improve.
* Ensure all questions are appropriate to the user's current level, avoiding overwhelming difficulty unless explicitly requested.
"""

    # Create the prompt template
    prompt = PromptTemplate(
        template=user_prompt_template,
        input_variables=["question", "context"],
    )

    # Initialize the Ollama model
    llm = ChatOllama(model=LOCAL_LLM_MODEL, temperature=0) # <-- This now uses the correct import
    
    # Chain the prompt, LLM, and output parser
    rag_chain = prompt | llm | StrOutputParser()

    # Combine documents into a single context string
    context_str = "\n\n---\n\n".join(documents)
    
    print("Generating answer from LLM...")
    # Run the chain
    generation = rag_chain.invoke({"question": question, "context": context_str})
    
    print(f"Generated answer: {generation}")
    return {"generation": generation}

def check_relevance(state: AgentState):
    """
    (Optional but good practice)
    Checks if the retrieved documents are relevant to the question.
    This node isn't fully implemented with an LLM call to save complexity,
    but it shows where you'd add this logic.
    """
    print("---NODE: CHECK_RELEVANCE---")
    documents = state["documents"]
    
    if not documents:
        print("No documents found. Skipping generation.")
        return "no_documents"
    
    # In a real app, you might use another LLM call here to grade the
    # relevance of 'documents' against the 'question'.
    # For this example, we'll assume they are always relevant.
    print("Documents found. Proceeding to generation.")
    return "documents_found"

# --- 4. Build and Compile the Graph ---

print("Building LangGraph...")
workflow = StateGraph(AgentState)

# Add the nodes to the graph
workflow.add_node("retrieve", retrieve_documents)
workflow.add_node("generate", generate_answer)

# Add conditional edges
# This is a simple conditional check
workflow.add_conditional_edges(
    "retrieve",
    check_relevance,
    {
        "documents_found": "generate", # If docs are found, go to generate
        "no_documents": END           # If no docs are found, end the graph
    }
)

# Set the entry and exit points
workflow.set_entry_point("retrieve")
workflow.add_edge("generate", END)

# Compile the graph into a runnable object
print("Compiling graph...")
app = workflow.compile()

# --- 5. Run the Agent ---
if __name__ == "__main__":
    print("Agent is ready. Initializing RAG...")
    
    # Setup the vector store (pass force_rebuild=True to re-scan PDFs)
    retriever = setup_vector_store(force_rebuild=False)
    
    if retriever is None:
        print("\nFailed to initialize vector store. Exiting.")
        print(f"Please make sure Ollama is running and you have PDFs in the '{CONTEXT_DIR}' folder.")
    else:
        print("\n--- Ollama RAG Agent is Online ---")
        print("Type 'rebuild' to re-index the PDF documents.")
        print("Type 'quit' or 'exit' to stop.")

        # =============================================================
        # ===> !! USER: ADD YOUR PROMPT HERE !! <===
        # =============================================================
        # If you want to use your own prompt, define it as a string here.
        # Make sure it includes "{question}" and "{context}" placeholders.
        # Example:
        # MY_CUSTOM_PROMPT = """
        # Act as a helpful legal assistant. Use the provided legal texts
        # to answer the user's question.
        #
        # Question: {question}
        # Context: {context}
        #
        # Analysis:
        # """
        MY_CUSTOM_PROMPT = "" # Leave as "" to use the default prompt
        # =============================================================

        while True:
            try:
                user_question = input("\nAsk a question about your documents: ")
                if user_question.lower() in ["quit", "exit"]:
                    print("Goodbye!")
                    break
                
                if user_question.lower() == "rebuild":
                    print("\nRebuilding vector store...")
                    retriever = setup_vector_store(force_rebuild=True)
                    if retriever:
                        print("Rebuild complete.")
                    else:
                        print("Rebuild failed.")
                    continue

                if not user_question.strip():
                    continue
                
                # This is where your speech-to-text input would go
                # e.g., user_question = speech_to_text_function()

                print("\nThinking...")
                
                # Prepare the input for the graph
                inputs = {
                    "question": user_question,
                    "user_prompt_template": MY_CUSTOM_PROMPT
                }
                
                # Run the graph
                # The .with_config() part is how we pass in the retriever
                # without it being part of the official "state".
                final_state = app.invoke(inputs, config={"configurable": {"retriever": retriever}})
                
                final_answer = final_state.get("generation", "Sorry, I couldn't find an answer.")

                print("\n--- Answer ---")
                print(final_answer)
                print("--------------")

                # This is where your text-to-speech output would go
                # e.g., text_to_speech_function(final_answer)

            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"\nAn error occurred: {e}")
                print("Please check that Ollama is running and accessible.")