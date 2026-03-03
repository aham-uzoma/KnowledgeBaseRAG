import os
from pathlib import Path
# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from loguru import logger

class VectorStoreManager:
    def __init__(self):
        self.processed_dir = Path("data/processed")
        self.db_dir = Path("data/vector_db")
        
        # This model turns text into numbers (vectors)
        # It's small, fast, and runs locally on your CPU
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        self.vector_db = None

    def chunk_markdown(self, file_path: Path):
        """Splits Markdown into chunks based on Headers (#, ##, ###)"""
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        
        # splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        # return splitter.split_text(content)


        # 1. Split by Headers
        header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        sections = header_splitter.split_text(content)

        # 2. Split sections further so they fit the embedding model's limit (usually 256-512 tokens)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800, 
            chunk_overlap=100
        )
    
        return text_splitter.split_documents(sections)
    

    # def build_index(self):
    #     """Reads all processed MD files and saves them to ChromaDB"""
    #     all_chunks = []
    #     md_files = list(self.processed_dir.glob("*.md"))

    #     if not md_files:
    #         logger.warning("No processed markdown files found!")
    #         return

    #     for md_file in md_files:
    #         logger.info(f"Chunking: {md_file.name}")
    #         chunks = self.chunk_markdown(md_file)
    #         all_chunks.extend(chunks)

    #     logger.info(f"Saving {len(all_chunks)} chunks to {self.db_dir}...")
        
    #     self.vector_db = Chroma.from_documents(
    #         documents=all_chunks,
    #         embedding=self.embeddings,
    #         persist_directory=str(self.db_dir)
    #     )
    #     logger.success("Vector Database is ready!")

    
    def build_index(self):
        """Reads all processed MD files and saves them to ChromaDB"""
        all_documents = []
        md_files = list(self.processed_dir.glob("*.md"))

        if not md_files:
            logger.warning("No processed markdown files found!")
            return

        for md_file in md_files:
            logger.info(f"Chunking: {md_file.name}")
            chunks = self.chunk_markdown(md_file)

            # Convert each chunk to a Document with metadata
            # docs = [Document(page_content=chunk, metadata={"source": md_file.name}) for chunk in chunks]
            # all_documents.extend(docs)
            for chunk in chunks:
                # Inject the filename into the content so the LLM knows which file it's reading
                chunk.page_content = f"Source File: {md_file.name}\n\n{chunk.page_content}"
                chunk.metadata["source"] = md_file.name
    
            all_documents.extend(chunks)

        logger.info(f"Saving {len(all_documents)} documents to {self.db_dir}...")

        self.vector_db = Chroma.from_documents(
            documents=all_documents,
            embedding=self.embeddings,
            persist_directory=str(self.db_dir)
        )
        # self.vector_db.persist()
        logger.success("Vector Database is ready!")
    
    
    def search(self, query: str, k: int = 3):
        """Test function to search the DB"""
        if not self.vector_db:
            # Load from disk if not in memory
            self.vector_db = Chroma(persist_directory=str(self.db_dir), embedding_function=self.embeddings)
        
        results = self.vector_db.similarity_search(query, k=k)
        return results

if __name__ == "__main__":
    manager = VectorStoreManager()
    manager.build_index()
    
    # Quick Test
    test_query = "What is this document about?"
    matches = manager.search(test_query)
    
    print("\n--- TEST SEARCH RESULTS ---")
    for i, res in enumerate(matches):
        print(f"\nResult {i+1}:")
        print(res.page_content[:200] + "...")