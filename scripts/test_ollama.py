import sys
from pathlib import Path

# Add project root to sys path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from core.llm import get_llm
from core.config import LOCAL_LLM_MODEL, LLM_PROVIDER
from core.logger import get_logger

logger = get_logger("test_ollama")

def test_connection():
    print(f"--- Ollama Connection Test ---")
    print(f"Provider: {LLM_PROVIDER}")
    print(f"Model: {LOCAL_LLM_MODEL}")
    
    try:
        print("\nInitializing LLM...")
        llm = get_llm(caller_name="test_script")
        
        print("Sending test prompt: 'Hi, are you there?'")
        response = llm.invoke("Hi, are you there?")
        
        print("\n[SUCCESS] Response received:")
        print(f"Content: {response.content}")
        
        print("\nCheck 'logs/token_usage.log' to see if tokens were recorded correctly.")
        
    except Exception as e:
        print(f"\n[FAILURE] Error occurred:")
        print(str(e))
        
        if "500" in str(e) and "memory" in str(e).lower():
            print("\nTIP: It looks like the model is too large for your system's RAM/VRAM.")
            print(f"Current model: {LOCAL_LLM_MODEL}")
            print("Try a model like 'llama3.1:8b' or 'mistral:latest'.")
        elif "connection" in str(e).lower() or "111" in str(e):
            print("\nTIP: Make sure the Ollama service is running (`ollama serve`).")

if __name__ == "__main__":
    test_connection()
