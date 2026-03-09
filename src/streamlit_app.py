
import streamlit as st
from pathlib import Path
from agent import IntelligentAgent
from ingestor import KnowledgeIngestor
from database import VectorStoreManager
from langchain_classic.chains import ConversationalRetrievalChain
from PIL import Image

# Load the image from your local directory
logo = Image.open("asset/edSpan.jpg")

st.set_page_config(page_title="Intelligent Concierge", page_icon=logo, layout="wide")

# Display logo centered
st.image("asset/edSpan.jpg", width=200)

st.title("EdSpan Global Knowledge Base")
st.caption("A Conversational RAG with Memory")

# -----------------------------
# 1. Initialize Agent (Singleton Pattern)
# -----------------------------
if "agent" not in st.session_state:
    with st.spinner("Initializing AI Engine..."):
        st.session_state.agent = IntelligentAgent()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False

# -----------------------------
# 2. SIDEBAR (PDF Upload Section)
# -----------------------------
st.sidebar.header("📄 EdSpan Knowledge Base")
uploaded_pdf = st.sidebar.file_uploader("Upload a PDF to teach the agent", type=["pdf"])

# --- TRACKER LOGIC ---
# Initialize a tracker to see if the filename has changed
if "last_processed_file" not in st.session_state:
    st.session_state.last_processed_file = None

# If a new file is uploaded that is different from the last one, reset the flag
if uploaded_pdf is not None and uploaded_pdf.name != st.session_state.last_processed_file:
    st.session_state.pdf_processed = False

# # Clear Chat / Reset Button
# if st.sidebar.button("Clear Conversation & Reset"):
#     st.session_state.messages = []
#     # Clear the memory in the actual LangChain object too
#     st.session_state.agent.memory.clear()
#     st.rerun()

# Clear Chat / Reset Button
if st.sidebar.button("Clear Conversation & Reset"):
    st.session_state.messages = []
    st.session_state.agent.memory.clear()
    st.session_state.pdf_processed = False # Reset flag on manual clear too
    st.session_state.last_processed_file = None
    st.rerun()

# Processing Logic
if uploaded_pdf is not None and not st.session_state.pdf_processed:
    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    file_path = raw_dir / uploaded_pdf.name
    output_markdown = processed_dir / f"{Path(uploaded_pdf.name).stem}.md"

    with st.sidebar.status("Processing Document...", expanded=True) as status:
        # Save file
        with open(file_path, "wb") as f:
            f.write(uploaded_pdf.getbuffer())

        # st.write("Checking PDF...")

        # # Convert PDF → Markdown
        # ingestor = KnowledgeIngestor()
        # processed_path = ingestor.process_pdf(uploaded_pdf.name)

        # 2. THE SMART CHECK: Check if conversion is needed
        if output_markdown.exists():
            st.info(f"⏩ Found existing index for {uploaded_pdf.name}. Skipping OCR.")
            processed_path = str(output_markdown)
            # We still need to index it if this is a fresh upload in this session
            should_index = True
        else:
            st.write("🚀 Converting PDF to Markdown (Fast Mode)...")
            ingestor = KnowledgeIngestor()
            processed_path = ingestor.process_pdf(uploaded_pdf.name)
            should_index = True if processed_path else False

        # if processed_path:
        if should_index:
            st.write("Indexing content into Vector DB...")
            manager = VectorStoreManager()
            # manager.build_index()
            manager.build_index(specific_file=Path(processed_path))


            # Update the Agent's knowledge base dynamically
            st.session_state.agent.vector_db = manager.vector_db
            
            # Re-link the retriever to the new database
            st.session_state.agent.qa_chain = ConversationalRetrievalChain.from_llm(
                llm=st.session_state.agent.llm,
                retriever=manager.vector_db.as_retriever(search_kwargs={"k": 5}),
                memory=st.session_state.agent.memory,
                return_source_documents=True
            )
            
        #     st.session_state.pdf_processed = True
        #     status.update(label="Knowledge Base Updated!", state="complete")
        # else:
        #     status.update(label="Ingestion Failed", state="error")
        # This is the "Stop Sign" for Streamlit
            st.session_state.pdf_processed = True 
            st.session_state.last_processed_file = uploaded_pdf.name
            status.update(label="Ready for Chat!", state="complete")

# -----------------------------
# 3. Display Chat History
# -----------------------------
# Use a container for better scrolling behavior
chat_container = st.container()

with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# -----------------------------
# 4. Chat Input & Response
# -----------------------------
if prompt := st.chat_input("Ask me something about your documents..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Assistant Response
    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            try:
                # Use the 'ask' method from your IntelligentAgent class
                response_text = st.session_state.agent.ask(prompt)
                st.markdown(response_text)
                
                # Add assistant message to history
                st.session_state.messages.append({"role": "assistant", "content": response_text})
            except Exception as e:
                st.error(f"An error occurred: {e}")





