import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

class Config:
    # Pinecone Configuration
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "quickstart")  # Default to 'quickstart'
    PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "gcp-starter")  # Default environment
    
    # Groq Configuration
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    
    # Embedding Model Configuration
    EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Fixed model name
    EMBEDDING_DIMENSION = 384  # Dimension for all-MiniLM-L6-v2
    
    # Text Processing Configuration
    CHUNK_SIZE = 512
    CHUNK_OVERLAP = 64
    CONTEXT_WINDOW = 4096
    MAX_TOKENS = 8192
    
    # LLM Configuration
    LLM_MODEL = "llama3-8b-8192"  # Default model
    # LLM_MODEL = "llama3-70b-8192"
    TEMPERATURE = 0.1

def configure_langchain_settings():
    """Initialize LangChain components with these settings"""
    return {
        "llm": ChatGroq(
            temperature=Config.TEMPERATURE,
            model_name=Config.LLM_MODEL,
            groq_api_key=Config.GROQ_API_KEY,
            max_tokens=Config.MAX_TOKENS,
            # top_p = 0.9
            model_kwargs={"top_p": 0.9}  # Moved here
        ),
        
        "embeddings": HuggingFaceEmbeddings(
            model_name=Config.EMBED_MODEL
        ),
        "text_splitter": RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            length_function=len,
            is_separator_regex=False,
        )
    }

if __name__ == "__main__":
    print("Current configuration:")
    print(f"- Embedding Model: {Config.EMBED_MODEL}")
    print(f"- LLM Model: {Config.LLM_MODEL}")
    print(f"- Pinecone Index: {Config.PINECONE_INDEX_NAME}")
