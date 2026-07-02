import pytest
import core.llm
import core.embeddings
from core.llm import get_llm
from core.embeddings import get_embeddings

def test_get_llm_llamacpp_missing_path(monkeypatch):
    monkeypatch.setattr(core.llm, "LLM_PROVIDER", "LlamaCPP")
    monkeypatch.setattr(core.llm, "LLAMA_CPP_MODEL_PATH", "")
    with pytest.raises(ValueError, match="LLAMA_CPP_MODEL_PATH"):
        get_llm()

def test_get_embeddings_llamacpp_missing_path(monkeypatch):
    monkeypatch.setattr(core.embeddings, "LLM_PROVIDER", "LlamaCPP")
    monkeypatch.setattr(core.embeddings, "LLAMA_CPP_MODEL_PATH", "")
    with pytest.raises(ValueError, match="LLAMA_CPP_MODEL_PATH"):
        get_embeddings()
