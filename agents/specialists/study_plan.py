from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage
from core.state import AgentState
from core.llm import get_llm
from core.knowledge_graph import retrieve_graph_context
from core.utils import check_exit_intent, format_conversation_history
from agents.prompts import STUDY_PLAN_PROMPT
from core.logger import get_logger

logger = get_logger("study_plan")

def study_plan_node(state: AgentState):
    """Generate study plan response"""
    logger.info("---STUDY PLAN AGENT---")
    messages = state.get("messages", [])
    if not messages:
        return {}
    
    question = messages[-1].content
    
    if check_exit_intent(question):
        return {"active_agent": "end"}

    graph_context, raw_triples = retrieve_graph_context(question, max_seed_nodes=8, max_triples=20)
    context_str = graph_context if graph_context else "No specific context available."
    history_str = format_conversation_history(messages)
    
    llm = get_llm(temperature=0.3, caller_name="study_plan_agent")
    prompt = PromptTemplate(
        template=STUDY_PLAN_PROMPT,
        input_variables=["question", "context", "history"]
    )
    chain = prompt | llm | StrOutputParser()
    
    generation = chain.invoke({
        "question": question,
        "context": context_str,
        "history": history_str
    })
    
    return {
        "messages": [AIMessage(content=generation)],
        "documents": [],
        "active_graph_context": raw_triples
    }
