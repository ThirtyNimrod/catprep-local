from core.config import (
    LLM_PROVIDER, 
    LOCAL_LLM_MODEL,
    LLAMA_CPP_MODEL_PATH
)

def get_embeddings():
    provider = LLM_PROVIDER.lower().replace("_", "").replace("-", "")
    
    if provider in {"azureopenai", "azure"}:
        from langchain_openai import AzureOpenAIEmbeddings
        from core.config import AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_VERSION
        
        missing = []
        if not AZURE_OPENAI_API_KEY:
            missing.append("AZURE_OPENAI_API_KEY")
        if not AZURE_OPENAI_ENDPOINT:
            missing.append("AZURE_OPENAI_ENDPOINT")
        if not AZURE_OPENAI_API_VERSION:
            missing.append("AZURE_OPENAI_API_VERSION")
            
        if missing:
            raise ValueError(
                "Azure OpenAI configuration missing: " + ", ".join(missing)
            )
            
        return AzureOpenAIEmbeddings(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            openai_api_version=AZURE_OPENAI_API_VERSION,
            api_key=AZURE_OPENAI_API_KEY
        )

    if provider in {"llamacpp", "llama.cpp", "llama-cpp"}:
        if not LLAMA_CPP_MODEL_PATH:
            raise ValueError("LLAMA_CPP_MODEL_PATH is missing. Please set it in .env")
            
        from langchain_community.embeddings import LlamaCppEmbeddings
        
        n_ctx = 16384
        
        return LlamaCppEmbeddings(
            model_path=LLAMA_CPP_MODEL_PATH,
            n_ctx=n_ctx
        )
        
    from langchain_ollama import OllamaEmbeddings
    
    return OllamaEmbeddings(
        model=LOCAL_LLM_MODEL
    )
