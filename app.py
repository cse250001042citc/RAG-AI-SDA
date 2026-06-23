import streamlit as st
import os
import tempfile
import shutil
from langchain_core.messages import HumanMessage, AIMessage

# --- BACKEND FUNCTIONS IMPORT ---
# We use rag_engine_withchatmemory for core chat logic
import rag_engine_withchatmemory as rag
import ingestion as ing  # Fixed: Lowercase file name match

# ==============================================================================
# 1. OPTIMIZED RESOURCE CACHING (Crucial for Demos & Performance)
# ==============================================================================
@st.cache_resource
def initialize_cached_resources():
    """Load heavy machine learning models and database states exactly once."""
    # Ensure directories exist
    os.makedirs("docs", exist_ok=True)
    os.makedirs("db/chroma_db", exist_ok=True)
    
    # Trigger original file variables to ensure embedding_model & db connections are ready
    embedding = rag.embedding_model
    vector_db = rag.db
    llm_model = rag.model
    
    return embedding, vector_db, llm_model

# Run initialization
embedding_model, db, model = initialize_cached_resources()

# ==============================================================================
# 2. STREAMLIT INTERFACE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="Enterprise RAG Assistant", 
    page_icon="🤖", 
    layout="wide"
)

# Main layout header
st.title("🤖 Intelligent Knowledge Base & RAG Chat")
st.caption("Powered by Local Llama 3.2 (Ollama) & ChromaDB Vector Store")
st.write("---")

# ==============================================================================
# 3. SIDEBAR: DOCUMENT INGESTION MANAGER
# ==============================================================================
with st.sidebar:
    st.header("📂 Document Control Center")
    st.write("Upload company files to dynamically update your RAG knowledge base.")
    
    # File uploader widget accepting multiple files
    uploaded_files = st.file_uploader(
        "Upload Source Documents", 
        type=["pdf"], 
        accept_multiple_files=True,
        help="Currently supports PDF format optimization"
    )
    
    if uploaded_files:
        if st.button("🚀 Process & Embed Documents", use_container_width=True):
            # Create a clean temporary workspace for ingestion execution
            with tempfile.TemporaryDirectory() as temp_dir:
                saved_count = 0
                
                # Save uploaded streams to disk so PyPDFDirectoryLoader can process them
                for uploaded_file in uploaded_files:
                    temp_file_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(temp_file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    saved_count += 1
                
                # Execute Ingestion Pipeline
                with st.spinner(f"Parsing and chunking {saved_count} document(s)..."):
                    try:
                        # Step 1: Read out documents from temp path
                        documents = ing.load_documents(docs_path=temp_dir)
                        
                        # Step 2: Split into vector chunks
                        chunks = ing.split_documents(documents)
                        
                        # Step 3: Append/Create vector store embeddings
                        ing.create_vector_store(chunks, persist_directory="db/chroma_db")
                        
                        # Success metrics
                        st.success(f"Successfully processed {saved_count} files!")
                        st.toast("ChromaDB updated successfully!", icon="✅")
                        
                        # Clear Streamlit resource cache to refresh DB connection
                        st.cache_resource.clear()
                        
                    except Exception as e:
                        st.error(f"Ingestion Failed: {str(e)}")

    st.write("---")
    st.markdown("### 📊 System Status")
    try:
        # Fetch current record numbers inside ChromaDB safely for evaluators to view
        doc_count = db._collection.count()
        st.metric(label="Total Document Vectors Stored", value=doc_count)
    except Exception:
        st.metric(label="Total Document Vectors Stored", value="0 (Empty Database)")

# ==============================================================================
# 4. MAIN INTERFACE: SESSION STATE & CONVERSATIONAL MEMORY
# ==============================================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

# Clear chat history function
if st.button("🗑️ Clear Conversation History"):
    st.session_state.messages = []
    rag.chat_history = []  # Wipe structural memory in backend script
    st.rerun()

# Display active past conversation streams
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ==============================================================================
# 5. EXECUTE REAL-TIME USER INTERACTION
# ==============================================================================
if user_query := st.chat_input("Enter your query regarding your knowledge base..."):
    
    # 1. Display user intent instantly
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})

    # 2. Process query with RAG Backend
    with st.chat_message("assistant"):
        with st.spinner("Analyzing context & generating response..."):
            try:
                # Syncing Streamlit history state into your backend chat history list
                rag.chat_history = []
                for msg in st.session_state.messages[:-1]: # exclude current user query
                    if msg["role"] == "user":
                        rag.chat_history.append(HumanMessage(content=msg["content"]))
                    else:
                        rag.chat_history.append(AIMessage(content=msg["content"]))

                # Trigger the analytical logic pipeline inside rag_engine_withchatmemory
                answer = rag.ask_question(user_query)
                
                # Show generated response contents on screen
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                error_msg = f"An execution error occurred inside the engine: {str(e)}"
                st.error(error_msg)
