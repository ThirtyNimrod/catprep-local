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
LOCAL_LLM_MODEL = _clean_env(os.getenv("OLLAMA_MODEL_NAME"), "llama3.1:8b")

# Configuration Tiers based on context size
GRAPH_PROCESSING_CONFIG = {
    "tier3_huge": {"CHUNK_SIZE": 6000, "CHUNK_OVERLAP": 200, "MAX_CONCURRENT": 10, "BATCH_SIZE": 16, "BATCH_FALLBACK": 8, "MAX_BATCH_CHARS": 45000}, # e.g. GPT-5.2 (96k+)
    "tier2_large": {"CHUNK_SIZE": 4000, "CHUNK_OVERLAP": 150, "MAX_CONCURRENT": 6, "BATCH_SIZE": 12, "BATCH_FALLBACK": 6, "MAX_BATCH_CHARS": 30000}, # e.g. GPT-4o (64k)
    "tier1_standard": {"CHUNK_SIZE": 1500, "CHUNK_OVERLAP": 100, "MAX_CONCURRENT": 2, "BATCH_SIZE": 4, "BATCH_FALLBACK": 2, "MAX_BATCH_CHARS": 10000} # e.g. Ollama Local (8b+)
}

def get_processing_config():
    provider = LLM_PROVIDER.lower().replace("_", "").replace("-", "")
    
    if provider in {"azureopenai", "azure"}:
        # Map Azure deployments to tiers
        deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME", "").lower()
        if "gpt-5" in deployment or "gpt5" in deployment:
            return GRAPH_PROCESSING_CONFIG["tier3_huge"]
        else:
            return GRAPH_PROCESSING_CONFIG["tier2_large"]
    
    # Default to standard tier (Ollama)
    return GRAPH_PROCESSING_CONFIG["tier1_standard"]

# Azure OpenAI specific configurations
AZURE_OPENAI_API_KEY = _clean_env(os.getenv("AZURE_OPENAI_API_KEY"), "")
AZURE_OPENAI_ENDPOINT = _clean_env(os.getenv("AZURE_OPENAI_ENDPOINT"), "")
AZURE_OPENAI_API_VERSION = _clean_env(os.getenv("AZURE_OPENAI_API_VERSION"), "")
AZURE_OPENAI_CHAT_DEPLOYMENT_NAME = _clean_env(
	os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"),
	"gpt-5.2",
)
