from agents.router import router_node
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import Runnable

class FakeLLM(Runnable):
    def invoke(self, input, config=None, **kwargs):
        return AIMessage(content="study_plan")

def test_router_node_unknown():
    state = {"messages": []}
    updates = router_node(state)
    assert updates["active_agent"] == "unknown"
    assert updates["active_graph_context"] == []

def test_router_node_fills_slots(monkeypatch):
    monkeypatch.setattr("agents.router.get_llm", lambda *args, **kwargs: FakeLLM())
    
    state = {"messages": [HumanMessage(content="I want a 2 weeks plan for QA")]}
    updates = router_node(state)
    
    assert updates["active_agent"] == "study_plan"
    assert updates["timeframe"] == "2 weeks"
    assert updates["focus_area"] == "QA"
