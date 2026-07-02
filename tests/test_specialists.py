from langchain_core.messages import HumanMessage
from langchain_core.runnables import Runnable

import agents.specialists.study_plan as study_plan_mod
import agents.specialists.practice as practice_mod
import agents.specialists.feedback as feedback_mod


class FakeLLM(Runnable):
    """Minimal Runnable stand-in so `prompt | llm | StrOutputParser()` works
    without a real model."""

    def __init__(self, response: str):
        self.response = response

    def invoke(self, input, config=None, **kwargs):
        from langchain_core.messages import AIMessage
        return AIMessage(content=self.response)


# ---------------------------------------------------------------------------
# study_plan_node
# ---------------------------------------------------------------------------

def test_study_plan_node_empty_messages():
    assert study_plan_mod.study_plan_node({"messages": []}) == {}


def test_study_plan_node_exit_intent_short_circuits(monkeypatch):
    called = {"llm": False}
    monkeypatch.setattr(study_plan_mod, "get_llm", lambda *a, **k: called.__setitem__("llm", True))
    state = {"messages": [HumanMessage(content="bye")]}
    result = study_plan_mod.study_plan_node(state)
    assert result == {"active_agent": "end"}
    assert called["llm"] is False  # never reached the LLM call


def test_study_plan_node_generates_response(monkeypatch):
    monkeypatch.setattr(study_plan_mod, "get_llm", lambda *a, **k: FakeLLM("Here is your plan"))
    monkeypatch.setattr(
        study_plan_mod, "retrieve_graph_context", lambda *a, **k: ("ctx", [("a", "rel", "b")])
    )
    state = {"messages": [HumanMessage(content="I need a study plan")]}
    result = study_plan_mod.study_plan_node(state)
    assert result["messages"][0].content == "Here is your plan"
    assert result["active_graph_context"] == [("a", "rel", "b")]


# ---------------------------------------------------------------------------
# practice_node
# ---------------------------------------------------------------------------

def test_practice_node_empty_messages():
    assert practice_mod.practice_node({"messages": []}) == {}


def test_practice_node_exit_intent_short_circuits():
    state = {"messages": [HumanMessage(content="quit")]}
    assert practice_mod.practice_node(state) == {"active_agent": "end"}


def test_practice_node_uses_focus_area_in_search_query(monkeypatch):
    captured = {}

    def fake_retrieve(query, **kwargs):
        captured["query"] = query
        return ("", [])

    monkeypatch.setattr(practice_mod, "retrieve_graph_context", fake_retrieve)
    monkeypatch.setattr(practice_mod, "get_llm", lambda *a, **k: FakeLLM("Q1: ..."))
    state = {"messages": [HumanMessage(content="give me questions")], "focus_area": "QA"}
    practice_mod.practice_node(state)
    assert "QA" in captured["query"]


def test_practice_node_keeps_current_questions_on_answer_key_request(monkeypatch):
    monkeypatch.setattr(practice_mod, "retrieve_graph_context", lambda *a, **k: ("", []))
    monkeypatch.setattr(practice_mod, "get_llm", lambda *a, **k: FakeLLM("Here's the answer key"))
    state = {
        "messages": [HumanMessage(content="show me the answer key")],
        "current_questions": "Q1: ...",
    }
    result = practice_mod.practice_node(state)
    assert result["current_questions"] == "Q1: ..."


def test_practice_node_updates_current_questions_on_new_request(monkeypatch):
    monkeypatch.setattr(practice_mod, "retrieve_graph_context", lambda *a, **k: ("", []))
    monkeypatch.setattr(practice_mod, "get_llm", lambda *a, **k: FakeLLM("New questions here"))
    state = {"messages": [HumanMessage(content="give me QA questions")], "current_questions": "old"}
    result = practice_mod.practice_node(state)
    assert result["current_questions"] == "New questions here"


def test_practice_node_compresses_summary_after_many_messages(monkeypatch):
    monkeypatch.setattr(practice_mod, "retrieve_graph_context", lambda *a, **k: ("", []))
    monkeypatch.setattr(practice_mod, "get_llm", lambda *a, **k: FakeLLM("more questions"))
    messages = [HumanMessage(content=f"msg {i}") for i in range(9)]
    state = {"messages": messages, "previous_summary": ""}
    result = practice_mod.practice_node(state)
    assert result["previous_summary"] != ""


# ---------------------------------------------------------------------------
# feedback_node
# ---------------------------------------------------------------------------

def test_feedback_node_empty_messages():
    assert feedback_mod.feedback_node({"messages": []}) == {}


def test_feedback_node_exit_intent_short_circuits():
    state = {"messages": [HumanMessage(content="thanks")]}
    assert feedback_mod.feedback_node(state) == {"active_agent": "end"}


def test_feedback_node_uses_weak_areas_in_search_query(monkeypatch):
    captured = {}

    def fake_retrieve(query, **kwargs):
        captured["query"] = query
        return ("", [])

    monkeypatch.setattr(feedback_mod, "retrieve_graph_context", fake_retrieve)
    monkeypatch.setattr(feedback_mod, "get_llm", lambda *a, **k: FakeLLM("analysis"))
    state = {"messages": [HumanMessage(content="how am I doing")], "weak_areas": ["QA", "LR"]}
    feedback_mod.feedback_node(state)
    assert "QA" in captured["query"] and "LR" in captured["query"]


def test_feedback_node_defaults_weak_areas_when_mentioned(monkeypatch):
    monkeypatch.setattr(feedback_mod, "retrieve_graph_context", lambda *a, **k: ("", []))
    monkeypatch.setattr(feedback_mod, "get_llm", lambda *a, **k: FakeLLM("analysis"))
    state = {"messages": [HumanMessage(content="what are my weak areas")], "weak_areas": []}
    result = feedback_mod.feedback_node(state)
    assert result["weak_areas"] == ["QA", "VA/RC"]


def test_feedback_node_sets_mock_analysis_first_time(monkeypatch):
    monkeypatch.setattr(feedback_mod, "retrieve_graph_context", lambda *a, **k: ("", []))
    monkeypatch.setattr(feedback_mod, "get_llm", lambda *a, **k: FakeLLM("x" * 300))
    state = {
        "messages": [HumanMessage(content="review my test")],
        "mock_test_analysis": "No mock test analyzed yet.",
    }
    result = feedback_mod.feedback_node(state)
    assert result["mock_test_analysis"] != "No mock test analyzed yet."
    assert result["mock_test_analysis"].endswith("...")
