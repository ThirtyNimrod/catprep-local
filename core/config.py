import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

CONTEXT_DIR = "context"
VECTORSTORE_PATH = "faiss_db"

# LLM Configurations
LLM_PROVIDER = os.getenv("llm_provider", "Ollama")
LOCAL_LLM_MODEL = os.getenv("OLLAMA_MODEL_NAME", "granite4:tiny-h")

# Azure OpenAI specific configurations
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_CHAT_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME", "gpt-4o-mini")
