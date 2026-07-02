import re
from typing import List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

def contains_phrase(text: str, phrase: str) -> bool:
    """Check if text contains phrase with word-boundary awareness."""
    pattern = r'\b' + re.escape(phrase.lower()) + r'\b'
    return bool(re.search(pattern, text.lower()))

# Unambiguous, standalone exit words: if the message *ends* with one of these,
# treat it as exit intent even if an earlier continuation phrase ("no more",
# "not another") is also present, e.g. "no more questions, bye".
_STRONG_EXIT_WORDS = {"bye", "quit", "exit", "cancel"}

def check_exit_intent(text: str) -> bool:
    """Check if user wants to exit current agent"""
    trailing_words = re.findall(r"[a-zA-Z']+", text.lower())
    if trailing_words and trailing_words[-1] in _STRONG_EXIT_WORDS:
        return True

    continuation_phrases = ["keep going", "give me", "another", "more", "again", "continue"]
    for phrase in continuation_phrases:
        if contains_phrase(text, phrase):
            return False

    exit_keywords = ["bye", "exit", "quit", "thanks", "thank you", "done", "cancel"]
    has_exit = any(contains_phrase(text, keyword) for keyword in exit_keywords)
    is_short = len(text.split()) <= 6

    return has_exit and is_short

def format_conversation_history(messages: List[BaseMessage], max_turns: int = 3) -> str:
    """Format conversation history for context (last N turns only)"""
    if not messages:
        return "No previous conversation."

    # Skip the very last message assuming it's the current query
    recent = messages[-(max_turns*2+1):-1]
    formatted = []
    for msg in recent:
        if isinstance(msg, HumanMessage):
            formatted.append(f"User: {msg.content}")
        elif isinstance(msg, AIMessage):
            formatted.append(f"Assistant: {msg.content}")
    return "\n".join(formatted)

def compress_practice_summary(messages: List[BaseMessage]) -> str:
    """Compress old practice sessions into a brief summary"""
    if len(messages) <= 6:
        return ""

    return "User has completed previous practice sessions in this conversation."
