import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI  
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory

# from langchain.chains import ConversationalRetrievalChain # Upgraded
# from langchain.memory import ConversationBufferMemory # Added Memory
from loguru import logger

load_dotenv()

class IntelligentAgent:
    def __init__(self):
        logger.info("Initializing Intelligent Concierge Agent with Memory...")
        
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        self.vector_db = Chroma(
            persist_directory="data/vector_db",
            embedding_function=self.embeddings
        )
        
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0, # Lowered to 0 for more factual accuracy
        )

        # 1. SETUP MEMORY
        # This stores the chat history so "the invoice" makes sense in context
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer" # Tells memory which part of the response to save
        )
        
        # 2. CREATE CONVERSATIONAL CHAIN
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vector_db.as_retriever(search_kwargs={"k": 5}), # Increased k to 5
            memory=self.memory,
            return_source_documents=True # CRUCIAL: This tells us WHY it says "I don't know"
        )

    def ask(self, question: str):
        logger.info(f"Question: {question}")
        
        # Invoke the chain
        response = self.qa_chain.invoke({"question": question})
        
        answer = response["answer"]
        sources = response["source_documents"]

        # 3. DEBUGGING: Print the sources found
        if sources:
            unique_sources = set([doc.metadata.get('source', 'Unknown') for doc in sources])
            logger.debug(f"Retrieved context from: {unique_sources}")
        else:
            logger.warning("No relevant sources found in the database!")

        return answer

if __name__ == "__main__":
    agent = IntelligentAgent()
    
    print("\n🚀 Agent is ready with Memory! Type 'exit' to quit.")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit"]:
            break
            
        answer = agent.ask(user_input)
        print(f"\nOpenAI: {answer}")