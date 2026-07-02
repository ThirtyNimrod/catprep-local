import os
import sys
from pathlib import Path

# Add project root to sys.path so we can import from core
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import core.config
import core.llm
from core.llm import get_llm

# Skip all tests in this file when running under pytest
if "pytest" in sys.modules:
    import pytest
    pytestmark = pytest.mark.skip(
        reason="Model integration tests are skipped in pytest. Run manually with python tests/test_models.py"
    )

import urllib.request

def is_ollama_running() -> bool:
    try:
        # Check connection to local Ollama server
        with urllib.request.urlopen("http://localhost:11434", timeout=0.5) as response:
            return response.status == 200
    except Exception:
        return False

def test_ollama():
    print("Testing Ollama Model...")
    # Override core.llm variables directly since they are imported there
    core.llm.LLM_PROVIDER = "Ollama"
    # Provide fallback to .env.example values if not set
    if not core.llm.LOCAL_LLM_MODEL:
        core.llm.LOCAL_LLM_MODEL = "llama3.2:8b"
    
    print(f"Using model: {core.llm.LOCAL_LLM_MODEL}")
    if not is_ollama_running():
        print("Ollama is not running on http://localhost:11434. Skipping test.\n")
        return
        
    try:
        llm = get_llm(temperature=0.1)
        response = llm.invoke("Hello, who are you? Please answer in one short sentence.")
        print("Ollama Response:", response.content)
        print("Ollama Test Passed!\n")
    except Exception as e:
        print("Ollama Test Failed:", e, "\n")

def test_llama_cpp():
    print("Testing Llama CPP Model...")
    # Override core.llm variables directly
    core.llm.LLM_PROVIDER = "LlamaCPP"
    
    # Use the path specified by the user
    model_path = r"D:\llama\models\Qwythos-9B-Claude-Mythos-5-1M-Q5_K_M.gguf"
    
    # Use the one from config if it exists, otherwise fallback to the hardcoded test path
    if not core.llm.LLAMA_CPP_MODEL_PATH:
        core.llm.LLAMA_CPP_MODEL_PATH = model_path

    print(f"Using model path: {core.llm.LLAMA_CPP_MODEL_PATH}")
    try:
        if not os.path.exists(core.llm.LLAMA_CPP_MODEL_PATH):
            print(f"Warning: Model file not found at {core.llm.LLAMA_CPP_MODEL_PATH}.")
            print("Please ensure the path is correct or download the model.")
            print("Skipping execution.\n")
            return

        llm = get_llm(temperature=0.1)
        response = llm.invoke("Hello, who are you? Please answer in one short sentence.")
        print("Llama CPP Response:", response.content)
        print("Llama CPP Test Passed!\n")
    except Exception as e:
        print("Llama CPP Test Failed:", e, "\n")

if __name__ == "__main__":
    print("Starting Model Tests...\n")
    test_ollama()
    test_llama_cpp()
