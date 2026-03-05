from langchain_ollama import ChatOllama
from langchain_openai import AzureChatOpenAI
from core.config import (
    LLM_PROVIDER, 
    LOCAL_LLM_MODEL, 
    AZURE_OPENAI_API_KEY, 
    AZURE_OPENAI_ENDPOINT, 
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_CHAT_DEPLOYMENT_NAME
)

def get_llm(temperature: float = 0.3):
    if LLM_PROVIDER.lower() == "azureopenai":
        return AzureChatOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            openai_api_version=AZURE_OPENAI_API_VERSION,
            azure_deployment=AZURE_OPENAI_CHAT_DEPLOYMENT_NAME,
            api_key=AZURE_OPENAI_API_KEY,
            temperature=temperature
        )
    else:
        # Default to Ollama
        return ChatOllama(model=LOCAL_LLM_MODEL, temperature=temperature)
