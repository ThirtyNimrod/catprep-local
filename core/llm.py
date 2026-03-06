from core.config import (
    LLM_PROVIDER, 
    LOCAL_LLM_MODEL, 
    AZURE_OPENAI_API_KEY, 
    AZURE_OPENAI_ENDPOINT, 
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_CHAT_DEPLOYMENT_NAME
)

def get_llm(temperature: float = 0.3):
    provider = LLM_PROVIDER.lower().replace("_", "").replace("-", "")

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
            temperature=temperature
        )

    from langchain_ollama import ChatOllama

    # Default to Ollama
    return ChatOllama(model=LOCAL_LLM_MODEL, temperature=temperature)
