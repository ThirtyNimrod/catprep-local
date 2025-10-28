# Ollama RAG Agent with LangGraph

This project provides the backend logic for a Retrieval-Augmented Generation (RAG) agent that runs locally using Ollama, LangGraph, and documents from a local folder.

## How It Works

1. **Document Loading**
   - Scans a folder named `context` for any `.pdf` files.

2. **Indexing**
   - Loads PDFs, splits them into searchable chunks.
   - Stores them in a local vector store (using FAISS).
   - Only happens the first time or when you force a "rebuild".

3. **LangGraph Agent**
   - Uses LangGraph to define a simple, two-step "graph":
     - **Retrieve**: Find relevant document chunks from the vector store based on your question.
     - **Generate**: Pass the question and retrieved context to your local Ollama model (e.g., `llama3`) to generate an answer.

4. **Chat Loop**
   - Provides a simple command-line interface to chat with your documents.

## Setup and Run

### 1. Prerequisites

- You must have Ollama installed and running.
- Pull a model (default: `llama3`).
  ```bash
  ollama pull llama3
  ```

### 2. Install Python Dependencies

```bash
pip install langgraph langchain langchain_community faiss-cpu pypdf ollama
```

- `langgraph`: For building the agent flow.
- `langchain` & `langchain_community`: For loaders, models, and prompts.
- `faiss-cpu`: For the local vector store.
- `pypdf`: To read your PDF files.
- `ollama`: To connect to the Ollama client.

### 3. Add Your Documents

- Create a folder named `context` in the same directory as the script.
- Place all your `.pdf` files inside this folder.

### 4. Add Your Custom Prompt (Optional)

- Open `ollama_rag_agent.py` and find the section marked:
  ```python
  # ===> !! USER: ADD YOUR PROMPT HERE !! <===
  ```
- Define your own prompt template string in the `MY_CUSTOM_PROMPT` variable.
- Ensure it includes `{question}` and `{context}` placeholders!

### 5. Run the Agent

- Once dependencies are installed, `context` folder is filled, and Ollama is running:
  ```bash
  python ollama_rag_agent.py
  ```
- First run will index documents (may take a moment).
- Subsequent runs will load the saved index.

## Important Notes

### 1. "RealTimeAPI" and Speech Support (The Incompatibility)

- "RealTimeAPI" usually refers to proprietary cloud services (like OpenAI's Realtime API or Google's Gemini Live API).
- These are **not compatible** with Ollama.

#### Solution: Use Local Speech Libraries!

**For Speech-to-Text (STT):**
- `speech_recognition`: Uses microphone and hooks into engines like `vosk` or local `Whisper`.
- `Whisper.cpp`: Highest quality local transcription.

**For Text-to-Speech (TTS):**
- `pyttsx3`: Simple, offline TTS library.
- `gTTS`: Easy to use, but requires internet.

> I've marked the places in `ollama_rag_agent.py` where you would add your `speech_to_text_function()` and `text_to_speech_function(final_answer)`.

### 2. Building Your UI

- This script is the **backend** for your agent.
- To create a UI:
  1. Wrap agent logic inside a web framework like **FastAPI**.
  2. Create an endpoint (e.g., `/chat`) that takes a user's question.
  3. Call your `LangGraph` `app.invoke()` function from inside that endpoint.
  4. Return the `final_answer` as a JSON response.
  5. Frontend (e.g., HTML/JS or React) sends requests to this backend and displays answers.