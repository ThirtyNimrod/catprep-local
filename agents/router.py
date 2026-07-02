from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from core.state import AgentState
from core.llm import get_llm
from agents.prompts import ROUTER_PROMPT
from core.logger import get_logger
from core.slot_filling import extract_timeframe, extract_focus_area

logger = get_logger("router")

def router_node(state: AgentState):
    """Determines which specialist agent should handle the query"""
    logger.info("---ROUTER AGENT---")
    messages = state.get("messages", [])
    if not messages:
        return {"active_agent": "unknown", "active_graph_context": []}
    
    question = messages[-1].content
    
    llm = get_llm(temperature=0, caller_name="router_agent")
    prompt = PromptTemplate(template=ROUTER_PROMPT, input_variables=["question"])
    chain = prompt | llm | StrOutputParser()
    
    result = chain.invoke({"question": question}).strip().lower()
    logger.info(f"Routing to: {result}")
    
    valid_routes = ["study_plan", "practice", "feedback", "unknown"]
    next_agent = result if result in valid_routes else "unknown"
    
    updates = {"active_agent": next_agent}
    
    if next_agent == "unknown":
        updates["active_graph_context"] = []
        
    timeframe = extract_timeframe(question)
    if timeframe:
        updates["timeframe"] = timeframe
        
    focus_area = extract_focus_area(question)
    if focus_area:
        updates["focus_area"] = focus_area
        
    return updates
