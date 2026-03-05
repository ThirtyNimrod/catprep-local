from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage
from core.state import AgentState
from core.llm import get_llm
from core.vector_store import retrieve_documents
from core.utils import check_exit_intent, format_conversation_history, compress_practice_summary
from agents.prompts import PRACTICE_QUESTIONS_PROMPT

def practice_node(state: AgentState):
    """Generate practice questions or answer keys"""
    print("---PRACTICE AGENT---")
    messages = state.get("messages", [])
    if not messages:
        return {}
        
    question = messages[-1].content
    
    if check_exit_intent(question):
        return {"active_agent": "end"}
        
    focus_area = state.get("focus_area")
    search_query = f"{focus_area} {question}" if focus_area else question
    documents = retrieve_documents(search_query, k=5)
    
    current_questions = state.get("current_questions", "No questions in current session.")
    previous_summary = state.get("previous_summary", "")
    
    context_str = "\n\n---\n\n".join(documents) if documents else "No context available."
    history_str = format_conversation_history(messages, max_turns=2)
    
    llm = get_llm(temperature=0.3)
    prompt = PromptTemplate(
        template=PRACTICE_QUESTIONS_PROMPT,
        input_variables=["question", "context", "history", "current_questions", "previous_summary"]
    )
    chain = prompt | llm | StrOutputParser()
    
    generation = chain.invoke({
        "question": question,
        "context": context_str,
        "history": history_str,
        "current_questions": current_questions,
        "previous_summary": previous_summary
    })
    
    new_current_questions = current_questions
    if "answer key" not in question.lower() and "solution" not in question.lower():
        new_current_questions = generation
        
    new_previous_summary = previous_summary
    if len(messages) > 8:
        new_previous_summary = compress_practice_summary(messages)
    
    return {
        "messages": [AIMessage(content=generation)],
        "documents": documents,
        "current_questions": new_current_questions,
        "previous_summary": new_previous_summary
    }
