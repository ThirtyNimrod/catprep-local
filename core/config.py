import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from project-root .env regardless of current working dir.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")


def _clean_env(value: str | None, default: str = "") -> str:
	"""Normalize env values and strip trailing inline comments."""
	if value is None:
		return default
	cleaned = value.strip()
	if " #" in cleaned:
		cleaned = cleaned.split(" #", 1)[0].strip()
	return cleaned or default

CONTEXT_DIR = os.getenv("CONTEXT_DIR", str(PROJECT_ROOT / "context"))
VECTORSTORE_PATH = os.getenv("VECTORSTORE_PATH", str(PROJECT_ROOT / "faiss_db"))

# LLM Configurations
LLM_PROVIDER = _clean_env(os.getenv("llm_provider") or os.getenv("LLM_PROVIDER"), "Ollama")
LOCAL_LLM_MODEL = _clean_env(os.getenv("OLLAMA_MODEL_NAME"), "granite4:tiny-h")

# Azure OpenAI specific configurations
AZURE_OPENAI_API_KEY = _clean_env(os.getenv("AZURE_OPENAI_API_KEY"), "")
AZURE_OPENAI_ENDPOINT = _clean_env(os.getenv("AZURE_OPENAI_ENDPOINT"), "")
AZURE_OPENAI_API_VERSION = _clean_env(os.getenv("AZURE_OPENAI_API_VERSION"), "")
AZURE_OPENAI_CHAT_DEPLOYMENT_NAME = _clean_env(
	os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"),
	"gpt-5.2",
)
