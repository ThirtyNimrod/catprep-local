from typing import TypedDict, Annotated, Sequence, List, Optional
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    """
    Unified state for the Multi-Agent CAT Prep System.
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]
    active_agent: str
    documents: List[str]
    active_graph_context: List[tuple]
    timeframe: Optional[str]
    current_questions: str
    previous_summary: str
    focus_area: Optional[str]
    mock_test_analysis: str
    weak_areas: List[str]
