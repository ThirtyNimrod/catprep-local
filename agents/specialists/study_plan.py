from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage
from core.state import AgentState
from core.llm import get_llm
from core.vector_store import retrieve_documents
from core.utils import check_exit_intent, format_conversation_history
from agents.prompts import STUDY_PLAN_PROMPT

def study_plan_node(state: AgentState):
    """Generate study plan response"""
    print("---STUDY PLAN AGENT---")
    messages = state.get("messages", [])
    if not messages:
        return {}
    
    question = messages[-1].content
    
    if check_exit_intent(question):
        return {"active_agent": "end"}

    documents = retrieve_documents(question, k=3)
    context_str = "\n\n---\n\n".join(documents) if documents else "No specific context available."
    history_str = format_conversation_history(messages)
    
    llm = get_llm(temperature=0.3)
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
        "documents": documents
    }
