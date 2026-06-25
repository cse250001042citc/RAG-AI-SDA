import pickle
import os
from langchain_community.document_loaders import PyPDFDirectoryLoader,TextLoader,DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

def load_documents(docs_path="docs"):
    """Load all text files from the docs directory"""
    print(f"Loading documents from {docs_path}...")
    
    # Check if docs directory exists
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The directory {docs_path} does not exist. Please create it and add your company files.")
    
    documents = []
    # Load all .txt files from the docs directory
    loader = DirectoryLoader(
        path=docs_path,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    documents = loader.load()

    # Load all PDF files from the docs directory
    pdf_loader = PyPDFDirectoryLoader(docs_path)
    pdf_documents = pdf_loader.load()
    documents.extend(pdf_documents)

    if len(documents) == 0:
        raise FileNotFoundError(f"No .pdf files found in {docs_path}. Please add your company documents.")
    
    
    # for i, doc in enumerate(documents[:6]):  # Show first 6 documents
    #     print(f"\nDocument {i+1}:")
    #     print(f"  Source: {doc.metadata['source']}")
    #     print(f"  Content length: {len(doc.page_content)} characters")
    #     print(f"  Content preview: {doc.page_content[:100]}...")
    #     print(f"  metadata: {doc.metadata}")
    print("=" * 50)
    return documents



def split_documents(documents, chunk_size=800, chunk_overlap=150):
    """Split documents into smaller chunks with overlap optimized for local embeddings"""
    print(f"Splitting documents into chunks (size: {chunk_size}, overlap: {chunk_overlap})...")
    
    # Using RecursiveCharacterTextSplitter to split smartly on paragraphs and sentences
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    
    chunks = text_splitter.split_documents(documents)
    
    # if chunks:
    #     for i, chunk in enumerate(chunks[:5]):
    #         print(f"\n--- Chunk {i+1} ---")
    #         print(f"Source: {chunk.metadata['source']}")
    #         print(f"Length: {len(chunk.page_content)} characters")
    #         print(f"Content:")
    #         print(chunk.page_content)
    #         print("-" * 50)
        
    #     if len(chunks) > 5:
    #         print(f"\n... and {len(chunks) - 5} more chunks")
            
    print(f"Total chunks created: {len(chunks)}")
    print("=" * 50)
    return chunks

def create_vector_store(chunks, persist_directory="db/chroma_db", bm25_path="db/bm25_store.pkl"):
    """Create and persist ChromaDB vector store using local HuggingFace embeddings"""
    print("Creating local embeddings and storing in ChromaDB...")
        
    # Swapped OpenAI for a free, local model that runs on your computer
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Create ChromaDB vector store
    print("--- Creating vector store ---")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory, 
        collection_metadata={"hnsw:space": "cosine"}
    )
    print("--- Finished creating vector store ---")
    
    print(f"Vector store created and saved to {persist_directory}")

    print("--- Serializing BM25 Text Store ---")
    with open(bm25_path, "wb") as f:
        pickle.dump(chunks, f)
        
    print("✅ Dual-indexing complete. Subsystems ready for Hybrid lookup.")
    return vectorstore


def main():
    """Main ingestion pipeline"""
    # print("MAin function")
    # print("=== RAG Document Ingestion Pipeline ===\n")
    
    # Step 1: Load documents
    documents = load_documents(docs_path="docs")  

#     # Step 2: Split into chunks
    chunks = split_documents(documents)
    
# #     # # Step 3: Create vector store
    vectorstore =create_vector_store(chunks)
    
    # print("\n✅ Ingestion complete! Your documents are now ready for RAG queries.")
    # return vectorstore

if __name__ == "__main__":
    main()






