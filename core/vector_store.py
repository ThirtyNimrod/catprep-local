import os
import shutil
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from core.config import CONTEXT_DIR, VECTORSTORE_PATH, LOCAL_LLM_MODEL

def get_embeddings():
    return OllamaEmbeddings(model=LOCAL_LLM_MODEL)

def setup_vector_store(force_rebuild: bool = False):
    """Creates or loads FAISS vector store with metadata for filtering"""
    print("Initializing vector store...")
    embeddings = get_embeddings()

    if os.path.exists(VECTORSTORE_PATH) and not force_rebuild:
        try:
            print(f"Loading existing vector store from {VECTORSTORE_PATH}...")
            vector_store = FAISS.load_local(VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True)
            return vector_store
        except Exception as e:
            print(f"Failed to load existing store: {e}. Rebuilding...")
            force_rebuild = True
    
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

    print("Creating FAISS vector store...")
    vector_store = FAISS.from_documents(splits, embeddings)
    
    # Save the store to disk
    vector_store.save_local(VECTORSTORE_PATH)
    
    print(f"Vector store saved to {VECTORSTORE_PATH}.")
    return vector_store

def retrieve_documents(query: str, k: int = 3):
    vector_store = setup_vector_store()
    if not vector_store:
        return []
    documents = vector_store.similarity_search(query, k=k)
    doc_snippets = [doc.page_content for doc in documents]
    return doc_snippets
