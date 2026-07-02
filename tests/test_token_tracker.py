from langchain_core.messages import AIMessage
from langchain_core.messages.ai import UsageMetadata
from langchain_core.outputs import ChatGeneration, Generation, LLMResult

import core.token_tracker as tt


def _logged_messages(monkeypatch):
    """Capture what TokenUsageCallbackHandler would write, without touching
    the real logs/token_usage.log file."""
    logged = []
    monkeypatch.setattr(tt.token_logger, "info", lambda msg: logged.append(msg))
    return logged


def test_on_llm_end_finds_usage_in_llm_output(monkeypatch):
    logged = _logged_messages(monkeypatch)
    handler = tt.TokenUsageCallbackHandler("caller_llm_output")
    result = LLMResult(
        generations=[[Generation(text="hi")]],
        llm_output={"token_usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}},
    )
    handler.on_llm_end(result)
    assert len(logged) == 1
    assert "caller_llm_output" in logged[0]
    assert "Prompt: 10" in logged[0]
    assert "Completion: 5" in logged[0]
    assert "Total: 15" in logged[0]


def test_on_llm_end_finds_usage_in_generation_info(monkeypatch):
    logged = _logged_messages(monkeypatch)
    handler = tt.TokenUsageCallbackHandler("caller_gen_info")
    gen = ChatGeneration(
        message=AIMessage(content="hi"),
        generation_info={"token_usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}},
    )
    handler.on_llm_end(LLMResult(generations=[[gen]]))
    assert "Prompt: 1" in logged[0]
    assert "Completion: 2" in logged[0]
    assert "Total: 3" in logged[0]


def test_on_llm_end_finds_usage_metadata(monkeypatch):
    logged = _logged_messages(monkeypatch)
    handler = tt.TokenUsageCallbackHandler("caller_usage_meta")
    msg = AIMessage(
        content="hi",
        usage_metadata=UsageMetadata(input_tokens=42, output_tokens=7, total_tokens=49),
    )
    handler.on_llm_end(LLMResult(generations=[[ChatGeneration(message=msg)]]))
    assert "Prompt: 42" in logged[0]
    assert "Completion: 7" in logged[0]
    assert "Total: 49" in logged[0]


def test_on_llm_end_finds_ollama_response_metadata(monkeypatch):
    logged = _logged_messages(monkeypatch)
    handler = tt.TokenUsageCallbackHandler("caller_ollama_meta")
    msg = AIMessage(
        content="hi",
        response_metadata={"prompt_eval_count": 20, "eval_count": 8},
    )
    handler.on_llm_end(LLMResult(generations=[[ChatGeneration(message=msg)]]))
    assert "Prompt: 20" in logged[0]
    assert "Completion: 8" in logged[0]
    assert "Total: 28" in logged[0]


def test_on_llm_end_logs_when_no_usage_found(monkeypatch):
    logged = _logged_messages(monkeypatch)
    handler = tt.TokenUsageCallbackHandler("caller_no_usage")
    msg = AIMessage(content="hi")
    handler.on_llm_end(LLMResult(generations=[[ChatGeneration(message=msg)]]))
    assert len(logged) == 1
    assert "No usage found" in logged[0]
    assert "caller_no_usage" in logged[0]
