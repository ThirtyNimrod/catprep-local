from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage
from core.state import AgentState
from core.llm import get_llm
from core.knowledge_graph import retrieve_graph_context
from core.utils import check_exit_intent, format_conversation_history
from agents.prompts import FEEDBACK_PROMPT
from core.logger import get_logger

logger = get_logger("feedback")

def feedback_node(state: AgentState):
    """Generate feedback and analysis"""
    logger.info("---FEEDBACK AGENT---")
    messages = state.get("messages", [])
    if not messages:
        return {}
        
    question = messages[-1].content
    
    if check_exit_intent(question):
        return {"active_agent": "end"}
        
    weak_areas = state.get("weak_areas", [])
    search_query = f"{' '.join(weak_areas)} {question}" if weak_areas else question

    mock_analysis = state.get("mock_test_analysis", "No mock test analyzed yet.")
    graph_context, raw_triples = retrieve_graph_context(search_query, max_seed_nodes=10, max_triples=30)
    context_str = graph_context if graph_context else "No context available."

    history_str = format_conversation_history(messages, max_turns=3)
    weak_areas_str = ", ".join(weak_areas) if weak_areas else "Not yet identified."
    
    llm = get_llm(temperature=0.3)
    prompt = PromptTemplate(
        template=FEEDBACK_PROMPT,
        input_variables=["question", "context", "history", "mock_analysis", "weak_areas"]
    )
    chain = prompt | llm | StrOutputParser()
    
    generation = chain.invoke({
        "question": question,
        "context": context_str,
        "history": history_str,
        "mock_analysis": mock_analysis,
        "weak_areas": weak_areas_str
    })
    
    new_weak_areas = weak_areas
    if not weak_areas and "weak" in question.lower():
        new_weak_areas = ["QA", "VA/RC"]
        
    new_mock_analysis = mock_analysis
    if mock_analysis == "No mock test analyzed yet.":
        new_mock_analysis = generation[:200] + "..."
        
    return {
        "messages": [AIMessage(content=generation)],
        "documents": [],
        "weak_areas": new_weak_areas,
        "mock_test_analysis": new_mock_analysis,
        "active_graph_context": raw_triples
    }
