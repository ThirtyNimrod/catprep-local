import core.config as config


def test_clean_env_strips_inline_comment():
    assert config._clean_env("Ollama # Options: Ollama, AzureOpenAI") == "Ollama"


def test_clean_env_none_returns_default():
    assert config._clean_env(None, "default") == "default"


def test_clean_env_blank_returns_default():
    assert config._clean_env("   ", "default") == "default"


def test_clean_env_plain_value_unchanged():
    assert config._clean_env("AzureOpenAI") == "AzureOpenAI"


def test_get_processing_config_defaults_to_standard_tier(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "Ollama")
    assert config.get_processing_config() == config.GRAPH_PROCESSING_CONFIG["tier1_standard"]


def test_get_processing_config_azure_gpt5_uses_huge_tier(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "AzureOpenAI")
    monkeypatch.setenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME", "gpt-5-preview")
    assert config.get_processing_config() == config.GRAPH_PROCESSING_CONFIG["tier3_huge"]


def test_get_processing_config_azure_gpt4o_uses_large_tier(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "AzureOpenAI")
    monkeypatch.setenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME", "gpt-4o")
    assert config.get_processing_config() == config.GRAPH_PROCESSING_CONFIG["tier2_large"]


def test_get_processing_config_provider_name_is_normalized(monkeypatch):
    # "azure-openai" / "azure_openai" / "AzureOpenAI" should all resolve the same way.
    monkeypatch.setattr(config, "LLM_PROVIDER", "azure-openai")
    monkeypatch.setenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME", "gpt-4o")
    assert config.get_processing_config() == config.GRAPH_PROCESSING_CONFIG["tier2_large"]
