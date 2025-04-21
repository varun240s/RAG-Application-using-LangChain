import os
from dotenv import load_dotenv
from pinecone import Pinecone  # v3+ client

from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

# Load env variables
load_dotenv()

# Initialize clients
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}  # or 'cuda' if GPU available
)

# Constants
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "rag-index")

# give your own data in pdf format.
PDF_PATH = "pdf_data/Paediatrics.pdf"

def load_and_chunk_pdf():
    """Load and split PDF documents"""
    loader = PyMuPDFLoader(PDF_PATH)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]  # Added for better splitting
    )
    return splitter.split_documents(loader.load())

def ensure_index_exists():
    """Create index if doesn't exist"""
    if INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=INDEX_NAME,
            dimension=384,  # all-MiniLM-L6-v2 dimension
            metric="cosine",
            spec=pc.IndexSpec()  # Default serverless spec
        )
        print(f"Created new index: {INDEX_NAME}")
    return pc.Index(INDEX_NAME)

def main():
    # 1. Load and process documents
    docs = load_and_chunk_pdf()
    
    # 2. Setup Pinecone
    index = ensure_index_exists()
    
    # 3. Store embeddings (with progress tracking)
    print(f"Uploading {len(docs)} chunks to Pinecone....")
    PineconeVectorStore.from_documents(
        documents=docs,
        embedding=embedding_model,
        index_name=INDEX_NAME,
        batch_size=100  # Better for large uploads
    )
    print(f"Successfully uploaded to '{INDEX_NAME}'")

if __name__ == "__main__":
    main()
    
    