
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

# Clear Chat / Reset Button
if st.sidebar.button("Clear Conversation & Reset"):
    st.session_state.messages = []
    # Clear the memory in the actual LangChain object too
    st.session_state.agent.memory.clear()
    st.rerun()

# Processing Logic
if uploaded_pdf is not None and not st.session_state.pdf_processed:
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    file_path = raw_dir / uploaded_pdf.name

    with st.sidebar.status("Processing Document...", expanded=True) as status:
        # Save file
        with open(file_path, "wb") as f:
            f.write(uploaded_pdf.getbuffer())
        st.write("Checking PDF...")

        # Convert PDF → Markdown
        ingestor = KnowledgeIngestor()
        processed_path = ingestor.process_pdf(uploaded_pdf.name)

        if processed_path:
            st.write("Indexing content into Vector DB...")
            manager = VectorStoreManager()
            manager.build_index()

            # Update the Agent's knowledge base dynamically
            st.session_state.agent.vector_db = manager.vector_db
            
            # Re-link the retriever to the new database
            st.session_state.agent.qa_chain = ConversationalRetrievalChain.from_llm(
                llm=st.session_state.agent.llm,
                retriever=manager.vector_db.as_retriever(search_kwargs={"k": 5}),
                memory=st.session_state.agent.memory,
                return_source_documents=True
            )
            
            st.session_state.pdf_processed = True
            status.update(label="✅ Knowledge Base Updated!", state="complete")
        else:
            status.update(label="❌ Ingestion Failed", state="error")

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





# import streamlit as st
# from pathlib import Path
# from agent import IntelligentAgent
# from ingestor import KnowledgeIngestor
# from database import VectorStoreManager
# from langchain_classic.chains import ConversationalRetrievalChain


# st.set_page_config(page_title="Intelligent Concierge", page_icon="🤖")

# st.title("🤖 Intelligent Concierge Agent")
# st.caption("Conversational RAG with Memory")

# # -----------------------------
# # Initialize Agent
# # -----------------------------
# if "agent" not in st.session_state:
#     st.session_state.agent = IntelligentAgent()

# if "messages" not in st.session_state:
#     st.session_state.messages = []


# # -----------------------------
# # Keep track of PDF processing
# # -----------------------------
# if "pdf_processed" not in st.session_state:
#     st.session_state.pdf_processed = False  # <-- Add this here

# # -----------------------------
# # SIDEBAR (PDF Upload Section)
# # -----------------------------
# st.sidebar.header("📄 Upload Knowledge PDF")
# uploaded_pdf = st.sidebar.file_uploader(
#     "Upload a PDF",
#     type=["pdf"]
# )

# if st.sidebar.button("Upload new PDF"):
#     st.session_state.pdf_processed = False
#     st.session_state.messages = []
#     st.rerun()

# if uploaded_pdf is not None and not st.session_state.pdf_processed:
#     raw_dir = Path("data/raw")
#     raw_dir.mkdir(parents=True, exist_ok=True)

#     file_path = raw_dir / uploaded_pdf.name

#     # Save uploaded file
#     with open(file_path, "wb") as f:
#         f.write(uploaded_pdf.getbuffer())

#     st.sidebar.info("Processing PDF...")

#     # Convert PDF → Markdown
#     ingestor = KnowledgeIngestor()
#     processed_path = ingestor.process_pdf(uploaded_pdf.name)

#     # if processed_path:
#     #     manager = VectorStoreManager()
#     #     manager.build_index()

#     #     # Reload vector DB inside agent
#     #     st.session_state.agent.vector_db = manager.vector_db

#     #     #  NEW: Update agent's retriever
#     #     st.session_state.agent.qa_chain.retriever = manager.vector_db.as_retriever(search_kwargs={"k":5})


#     #     st.sidebar.success("PDF processed and indexed successfully!")

#     #     # Set flag to True so this block doesn't run again
#     #     st.session_state.pdf_processed = True
#     if processed_path:
#         manager = VectorStoreManager()
#         manager.build_index()

#         # Reload vector DB in the agent
#         st.session_state.agent.vector_db = manager.vector_db

#         # Update retriever
#         st.session_state.agent.qa_chain = ConversationalRetrievalChain.from_llm(
#             llm=st.session_state.agent.llm,
#             retriever=manager.vector_db.as_retriever(search_kwargs={"k": 5}),
#             memory=st.session_state.agent.memory,
#             return_source_documents=True
#         )
#         results = st.session_state.agent.vector_db.similarity_search("Test query from new PDF", k=3)
#         st.write([r.page_content[:200] for r in results])

#         st.sidebar.success("PDF processed and indexed successfully!")
#         st.session_state.pdf_processed = True
#     else:
#         st.sidebar.error("Failed to process PDF.")
# # -----------------------------
# # Display Chat History
# # -----------------------------
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])

# # -----------------------------
# # Chat Input
# # -----------------------------
# if prompt := st.chat_input("Ask a question..."):

#     st.session_state.messages.append({"role": "user", "content": prompt})

#     with st.chat_message("user"):
#         st.markdown(prompt)

#     with st.chat_message("assistant"):
#         with st.spinner("Thinking..."):
#             response = st.session_state.agent.ask(prompt)
#             st.markdown(response)

#     st.session_state.messages.append({"role": "assistant", "content": response})






