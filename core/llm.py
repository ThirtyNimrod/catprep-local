from core.config import (
    LLM_PROVIDER, 
    LOCAL_LLM_MODEL, 
    AZURE_OPENAI_API_KEY, 
    AZURE_OPENAI_ENDPOINT, 
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_CHAT_DEPLOYMENT_NAME
)

from core.token_tracker import TokenUsageCallbackHandler

def get_llm(temperature: float = 0.3, caller_name: str | None = None):
    provider = LLM_PROVIDER.lower().replace("_", "").replace("-", "")
    
    callbacks = []
    if caller_name:
        callbacks.append(TokenUsageCallbackHandler(caller_name))

    if provider in {"azureopenai", "azure"}:
        missing = []
        if not AZURE_OPENAI_API_KEY:
            missing.append("AZURE_OPENAI_API_KEY")
        if not AZURE_OPENAI_ENDPOINT:
            missing.append("AZURE_OPENAI_ENDPOINT")
        if not AZURE_OPENAI_API_VERSION:
            missing.append("AZURE_OPENAI_API_VERSION")
        if not AZURE_OPENAI_CHAT_DEPLOYMENT_NAME:
            missing.append("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")

        if missing:
            raise ValueError(
                "Azure OpenAI configuration missing: " + ", ".join(missing)
            )

        from langchain_openai import AzureChatOpenAI

        return AzureChatOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            openai_api_version=AZURE_OPENAI_API_VERSION,
            azure_deployment=AZURE_OPENAI_CHAT_DEPLOYMENT_NAME,
            api_key=AZURE_OPENAI_API_KEY,
            temperature=temperature,
            callbacks=callbacks
        )

    from langchain_ollama import ChatOllama

    # Determine context window for Ollama
    # 16k is a safe default for modern local models (like Mistral, Qwen, etc)
    # without blowing up the VRAM like 128k does.
    num_ctx = 16384 

    # Default to Ollama
    return ChatOllama(
        model=LOCAL_LLM_MODEL, 
        temperature=temperature,
        callbacks=callbacks,
        num_ctx=num_ctx
    )
