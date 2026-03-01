import streamlit as st
import os
from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.core.prompts import PromptTemplate
from llama_index.llms.openai import OpenAI # Or your preferred LLM


from datetime import datetime


# --- INITIALIZE SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Add this to app.py

# --- 1. CONFIGURATION & STYLING ---
st.set_page_config(page_title="TXST AI Tutor", layout="wide")
st.title("🐱 Boko Buddy: Computer Architecture SP26")

# Sidebar for controls
with st.sidebar:
    st.header("Tutor Settings")
    # In the sidebar
    mode = st.radio(
        "Tutoring Style", 
        ["Strict (Lecture Only)", "Supportive (Lecture + Textbook)", "General AI (No RAG)"]
    )

    # Allow faculty to tweak the temperature
    temp = st.slider("Creativity/Temperature", 0.0, 1.0, 0.1)

    # Special option for summary 
    is_summary_mode = st.sidebar.toggle("Deep Search / Summary Mode")

    if st.button("Clear Chat History"):
        st.session_state.messages = []

# --- SIDEBAR DOWNLOAD BUTTON ---
with st.sidebar:
    st.markdown("---")
    st.header("Study Tools")
    
    if st.session_state.messages:
        # Prepare the chat history as a Markdown string
        chat_export = "# Computer Architecture Study Guide\n\n"
        for msg in st.session_state.messages:
            role = "Student" if msg["role"] == "user" else "Tutor"
            chat_export += f"### {role}\n{msg['content']}\n\n---\n\n"
        
        st.download_button(
            label="💾 Download Study Guide (.md)",
            data=chat_export,
            file_name="architecture_study_notes.md",
            mime="text/markdown",
            help="Click to save this conversation as a study guide."
        )
    else:
        st.caption("Chat with the tutor to generate a study guide.")
        
# --- 2. LOAD THE INDEX (Cached for speed) ---
@st.cache_resource(show_spinner=False)
def get_query_engine(mode):

    current_date = datetime.now().strftime("%A, %B %d, %Y")

    if mode == "General AI (No RAG)":
        # We create a simple chat engine that doesn't use the vector index
        from llama_index.core.chat_engine import SimpleChatEngine
        return SimpleChatEngine.from_defaults(
            system_prompt=f"You are a helpful AI assistant. Today is {current_date}. Answer using only your general knowledge." 
        )

    
    # Rebuild storage context from local directory
    storage_context = StorageContext.from_defaults(persist_dir="./storage")
    index = load_index_from_storage(storage_context)
    
    # --- SYSTEM PROMPT LOGIC ---
    if mode == "Strict (Lecture Only)":
        system_text = (
            "You are an expert Teaching Assistant for the Computer Architecture course at TXST for Spring 2026. Today is {current_date}. "
            "Use ONLY the 'lecture' source files to answer. "
            "If the answer isn't in the slides, refer the student to the textbook to find more"
        )
    else:
        system_text = (
            "You are an expert Teaching Assistant for the Computer Architecture course at TXST for Spring 2026. Today is {current_date}"
            "Your tone is professional, encouraging, and technically precise. "
            "1. Use the 'lecture' sources for high-level concepts and 'textbook' for deep technical details. "
            "2. If code is involved, C/MIPS examples where appropriate. "
            "3. ALWAYS cite your sources at the end of your explanation (e.g., 'Source: Chapter 3, Page 181'). "
            "4. If you don't know the answer based on the provided context, offer to help the student "
            "formulate a question for the Professor's office hours."
        )

    # Set the global LLM settings
    Settings.llm = OpenAI(model="gpt-4o", temperature=temp)
    
    k_value = 15 if is_summary_mode else 5
    # Create the engine
    # Note: We use 'as_chat_engine' to maintain conversation history
    chat_engine = index.as_chat_engine(
        chat_mode="context", 
        system_prompt=system_text,
        similarity_top_k=k_value, #
        streaming=True
    )
    return chat_engine

# --- 3. CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input

# --- MAIN CHAT INTERFACE ---
if prompt := st.chat_input("AMA Computer Architecture..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display the User's question
    with st.chat_message("user"):
        st.markdown(prompt)

    # 1. Create two columns for the comparison
    col_rag, col_general = st.columns(2)

    # 2. LEFT COLUMN: Your Custom RAG Tutor
    with col_rag:
        st.subheader("🎓 TXST AI Tutor")
        with st.chat_message("assistant", avatar="🐱"):
            # Force a RAG mode for this column
            rag_engine = get_query_engine("Supportive (Lecture + Textbook)")
            with st.spinner("Consulting course materials..."):
                today_str = datetime.now().strftime("%Y-%m-%d")
                enhanced_prompt = f"(Context: Today is {today_str}). User asks: {prompt}"
                rag_response = rag_engine.chat(enhanced_prompt)
                st.markdown(rag_response.response)
                
                # Show Citations if they exist
                if hasattr(rag_response, 'source_nodes') and rag_response.source_nodes:
                    with st.expander("📚 Sources Found"):
                        for node in rag_response.source_nodes[:3]: # Top 3
                            st.caption(f"From: {node.metadata.get('source_file')} (Page {node.metadata.get('page_label')})")

    # 3. RIGHT COLUMN: Standard LLM
    with col_general:
        st.subheader("🤖 Standard GPT ")
        with st.chat_message("assistant", avatar="🌐"):
            # Use the SimpleChatEngine we built earlier
            gen_engine = get_query_engine("General AI (No RAG)")
            with st.spinner("Thinking generally..."):
                today_str = datetime.now().strftime("%Y-%m-%d")
                enhanced_prompt = f"(Context: Today is {today_str}). User asks: {prompt}"
                gen_response = gen_engine.chat(enhanced_prompt)
                st.markdown(gen_response.response)

    # Save the RAG response to history (optional)
    st.session_state.messages.append({"role": "assistant", "content": rag_response.response})
    
# if prompt := st.chat_input(""):
#     st.session_state.messages.append({"role": "user", "content": prompt})
#     with st.chat_message("user"):
#         st.markdown(prompt)

#     # Generate Response
#     with st.chat_message("assistant"):
#         engine = get_query_engine(mode)
        
#         # 1. Perform the chat query
#         # Note: We use .chat() here instead of stream_chat for easier source extraction
#         # but you can use streaming with advanced logic if preferred.
#         today_str = datetime.now().strftime("%Y-%m-%d")
#         enhanced_prompt = f"(Context: Today is {today_str}). User asks: {prompt}"

#         response = engine.chat(enhanced_prompt)
#         st.markdown(response.response)

#         # 2. Extract and display Citations
#         if response.source_nodes:
#             with st.expander("📚 View Sources & Citations"):
#                 # Use a set to avoid showing the same file multiple times 
#                 # if multiple chunks came from one source
#                 seen_sources = set()
#                 for node in response.source_nodes:
#                     # Access the metadata we added during ingestion
#                     meta = node.metadata
#                     source_name = meta.get('source_file') or meta.get('chapter') or "Unknown Source"
#                     page = meta.get('page_label') or "N/A"
#                     date = meta.get('date') or "N/A"
                    
#                     source_key = f"{source_name} (Page {page})"
                    
#                     if source_key not in seen_sources:
#                         st.write(f"**Source:** {source_name}")
#                         if date != "N/A":
#                             st.caption(f"Lecture Date: {date}")
#                         # Show a small snippet of the actual text retrieved
#                         st.info(f"... {node.get_content()[:200]} ...")
#                         seen_sources.add(source_key)

#     st.session_state.messages.append({"role": "assistant", "content": response.response})



