from nicegui import ui
import asyncio
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from core.vector_store import setup_vector_store
from agents.graph import build_graph

# Initialize components
print("Starting Application Initialization...")
vector_store = setup_vector_store(force_rebuild=False)
memory = MemorySaver()
app_graph = build_graph(memory=memory)

# State initialization
THREAD_ID = "catprep_session_user_1"
config = {"configurable": {"thread_id": THREAD_ID}}

@ui.page('/')
async def main_page():
    ui.add_head_html('<style>body { font-family: "Inter", sans-serif; background-color: #f8fafc; }</style>')
    
    with ui.column().classes('w-full max-w-4xl mx-auto h-screen p-4 flex flex-col'):
        ui.label('CAT Prep Assistant').classes('text-3xl font-bold text-center text-blue-800 mb-4')
        
        # Chat history container
        chat_container = ui.column().classes('w-full flex-grow overflow-y-auto mb-4 bg-white rounded-lg shadow-md p-6')
        
        # Display greeting
        with chat_container:
            with ui.chat_message(name="System", stamp="now", avatar="https://robohash.org/cat").classes('w-full'):
                ui.markdown("Hello! I am your CAT Prep Assistant. I can help you with study plans, practice questions, or mock test feedback.")

        async def send_message():
            user_text = text_input.value.strip()
            if not user_text:
                return
            
            # Clear input
            text_input.value = ''
            
            # Display user message
            with chat_container:
                with ui.chat_message(name="You", stamp="now", avatar="https://robohash.org/user", sent=True).classes('w-full'):
                    ui.label(user_text)
            
            # Show loading spinner
            with chat_container:
                spinner_row = ui.row().classes('items-center gap-2')
                with spinner_row:
                    ui.spinner(size='md', color='primary')
                    ui.label('Thinking...').classes('text-gray-500 italic')
            
            try:
                def run_graph():
                    inputs = {"messages": [HumanMessage(content=user_text)]}
                    responses = []
                    # Stream values from graph
                    for event in app_graph.stream(inputs, config=config, stream_mode="values"):
                        msgs = event.get("messages", [])
                        if msgs and isinstance(msgs[-1], AIMessage):
                            # Append text of the assistant
                            responses.append(msgs[-1].content)
                    return responses[-1] if responses else "I couldn't generate a response."

                response_content = await asyncio.to_thread(run_graph)
                
                # Cleanup spinner
                chat_container.remove(spinner_row)
                
                # Display Assistant message
                with chat_container:
                    with ui.chat_message(name="Assistant", stamp="now", avatar="https://robohash.org/cat").classes('w-full'):
                        ui.markdown(response_content)
                    
            except Exception as e:
                chat_container.remove(spinner_row)
                with chat_container:
                    with ui.chat_message(name="System", stamp="now", avatar="https://robohash.org/cat").classes('w-full'):
                        ui.label(f"Error: {str(e)}").classes('text-red-500 font-bold')
        
        # Input area
        with ui.row().classes('w-full items-center p-4 bg-white rounded-lg shadow-md'):
            text_input = ui.input(placeholder='Ask for a study plan, practice questions...').classes('flex-grow text-lg').on('keydown.enter', send_message)
            ui.button(icon='send', on_click=send_message).classes('ml-4 p-4').props('round color="primary" size="lg"')

if __name__ in {"__main__", "__mp_main__"}:
    if vector_store is None:
        print("Warning: Vector store not initialized properly. Add PDFs to context/")
    ui.run(title="CAT Prep Assistant", port=8080)
