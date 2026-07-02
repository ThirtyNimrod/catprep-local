from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import Runnable
from langgraph.checkpoint.memory import MemorySaver

from agents.graph import build_graph, route_to_agent


class FakeLLM(Runnable):
    def __init__(self, response: str):
        self.response = response

    def invoke(self, input, config=None, **kwargs):
        return AIMessage(content=self.response)


def test_route_to_agent_reads_active_agent():
    assert route_to_agent({"active_agent": "practice"}) == "practice"


def test_route_to_agent_defaults_to_end():
    assert route_to_agent({}) == "end"


def test_build_graph_without_memory_compiles():
    graph = build_graph()
    assert graph is not None


def test_build_graph_end_to_end_routes_to_study_plan(monkeypatch):
    monkeypatch.setattr("agents.router.get_llm", lambda *a, **k: FakeLLM("study_plan"))
    monkeypatch.setattr(
        "agents.specialists.study_plan.get_llm", lambda *a, **k: FakeLLM("Your 5 week plan...")
    )
    monkeypatch.setattr(
        "agents.specialists.study_plan.retrieve_graph_context", lambda *a, **k: ("", [])
    )

    graph = build_graph(memory=MemorySaver())
    config = {"configurable": {"thread_id": "test-thread-study-plan"}}
    inputs = {"messages": [HumanMessage(content="I need a study plan")]}
    final_state = graph.invoke(inputs, config=config)

    assert final_state["active_agent"] == "study_plan"
    assert final_state["messages"][-1].content == "Your 5 week plan..."


def test_build_graph_end_to_end_routes_to_end_on_unknown(monkeypatch):
    monkeypatch.setattr("agents.router.get_llm", lambda *a, **k: FakeLLM("unknown"))

    graph = build_graph(memory=MemorySaver())
    config = {"configurable": {"thread_id": "test-thread-unknown"}}
    inputs = {"messages": [HumanMessage(content="asdkjaslkdj qqq")]}
    final_state = graph.invoke(inputs, config=config)

    assert final_state["active_agent"] == "unknown"
    assert final_state["active_graph_context"] == []


def test_build_graph_persists_slot_filled_state_across_turns(monkeypatch):
    monkeypatch.setattr("agents.router.get_llm", lambda *a, **k: FakeLLM("practice"))
    monkeypatch.setattr(
        "agents.specialists.practice.get_llm", lambda *a, **k: FakeLLM("Q1: 2+2=?")
    )
    monkeypatch.setattr(
        "agents.specialists.practice.retrieve_graph_context", lambda *a, **k: ("", [])
    )

    graph = build_graph(memory=MemorySaver())
    config = {"configurable": {"thread_id": "test-thread-persist"}}

    # Turn 1: mentions QA -> the router's slot-filling should set focus_area.
    graph.invoke({"messages": [HumanMessage(content="give me QA practice")]}, config=config)
    state = graph.get_state(config).values
    assert state["focus_area"] == "QA"
    assert state["current_questions"] == "Q1: 2+2=?"

    # Turn 2: no section mentioned -> focus_area must survive, since the
    # router only includes "focus_area" in its update dict when it actually
    # extracts one, relying on LangGraph's partial-update merge to preserve
    # whatever was filled in an earlier turn.
    graph.invoke({"messages": [HumanMessage(content="give me another one")]}, config=config)
    state = graph.get_state(config).values
    assert state["focus_area"] == "QA"
