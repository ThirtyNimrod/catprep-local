import logging
import os
from pathlib import Path
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

# Ensure logs directory exists
PROJECT_ROOT = Path(__file__).parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"
TOKEN_LOG_FILE = LOGS_DIR / "token_usage.log"

os.makedirs(LOGS_DIR, exist_ok=True)

# Separate logger for token usage
token_logger = logging.getLogger("token_usage")
token_logger.setLevel(logging.DEBUG)

# Avoid adding redundant handlers
if not token_logger.handlers:
    fh = logging.FileHandler(TOKEN_LOG_FILE, mode="a", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    token_logger.addHandler(fh)
    token_logger.propagate = False

class TokenUsageCallbackHandler(BaseCallbackHandler):
    """
    Callback handler to track token usage for LLM calls.
    """
    def __init__(self, caller_name: str):
        self.caller_name = caller_name

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        """Log token usage when LLM finishes."""
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        found_usage = False

        # First, check standard llm_output (OpenAI sometimes uses this)
        usage = response.llm_output.get("token_usage") if response.llm_output else None
        if usage:
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)
            found_usage = True

        # If not, dig into the ChatGenerations
        if not found_usage and response.generations:
            for generations in response.generations:
                for gen in generations:
                    # Check for generic generation_info
                    if gen.generation_info and "token_usage" in gen.generation_info:
                        usage = gen.generation_info["token_usage"]
                        prompt_tokens = usage.get("prompt_tokens", 0)
                        completion_tokens = usage.get("completion_tokens", 0)
                        total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)
                        found_usage = True
                        break
                    
                    # Check for ChatGeneration message metadata
                    if hasattr(gen, "message"):
                        msg = gen.message
                        # 1. Standard usage_metadata (langchain core >= 0.2)
                        usage_meta = getattr(msg, "usage_metadata", None)
                        if usage_meta:
                            prompt_tokens = usage_meta.get("input_tokens", 0)
                            completion_tokens = usage_meta.get("output_tokens", 0)
                            total_tokens = usage_meta.get("total_tokens", prompt_tokens + completion_tokens)
                            found_usage = True
                            break
                        
                        # 2. Ollama specific response_metadata
                        resp_meta = getattr(msg, "response_metadata", {})
                        if "prompt_eval_count" in resp_meta or "eval_count" in resp_meta:
                            prompt_tokens = resp_meta.get("prompt_eval_count", 0)
                            completion_tokens = resp_meta.get("eval_count", 0)
                            total_tokens = prompt_tokens + completion_tokens
                            found_usage = True
                            break
                if found_usage:
                    break

        if found_usage:
            token_logger.info(
                f"[{self.caller_name}] Prompt: {prompt_tokens}, "
                f"Completion: {completion_tokens}, Total: {total_tokens}"
            )
        else:
            # For debugging, dump what keys we do have in the message
            debug_info = "None"
            if response.generations and len(response.generations) > 0 and len(response.generations[0]) > 0:
                gen = response.generations[0][0]
                if hasattr(gen, 'message'):
                    debug_info = f"msg_meta: {getattr(gen.message, 'response_metadata', None)}"
            token_logger.info(f"[{self.caller_name}] No usage found. Debug info: {debug_info}")
