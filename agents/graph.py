from langgraph.graph import StateGraph, START, END
from core.state import AgentState
from agents.router import router_node
from agents.specialists.study_plan import study_plan_node
from agents.specialists.practice import practice_node
from agents.specialists.feedback import feedback_node

def route_to_agent(state: AgentState):
    """Conditional routing based on active_agent"""
    return state.get("active_agent", "end")

def build_graph(memory=None):
    workflow = StateGraph(AgentState)

    # Add Nodes
    workflow.add_node("router", router_node)
    workflow.add_node("study_plan", study_plan_node)
    workflow.add_node("practice", practice_node)
    workflow.add_node("feedback", feedback_node)

    # Edge from START to router
    workflow.add_edge(START, "router")
    
    # Conditional Routing from router to specialists
    workflow.add_conditional_edges(
        "router",
        route_to_agent,
        {
            "study_plan": "study_plan",
            "practice": "practice",
            "feedback": "feedback",
            "end": END,
            "unknown": END
        }
    )

    # Return to user (END) after specialist generates a response
    workflow.add_edge("study_plan", END)
    workflow.add_edge("practice", END)
    workflow.add_edge("feedback", END)

    if memory is not None:
        return workflow.compile(checkpointer=memory)
    return workflow.compile()
