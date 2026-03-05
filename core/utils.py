from typing import List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

def check_exit_intent(text: str) -> bool:
    """Check if user wants to exit current agent"""
    exit_keywords = ["bye", "exit", "quit", "thanks", "thank you", "done", "cancel"]
    return any(keyword in text.lower() for keyword in exit_keywords)

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
