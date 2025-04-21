from pinecone import Pinecone
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from data_loader import load_documents
from config import Config

def ingest_to_pinecone():
    docs = load_documents("data/documents.json")
    
    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name=Config.EMBED_MODEL
    )
    
    # Initialize Pinecone client (v3+)
    pc = Pinecone(api_key=Config.PINECONE_API_KEY)
    
    
    
    # be carefull while creating the index , the index should match to your embedding model.
    # check/create index
    if Config.PINECONE_INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=Config.PINECONE_INDEX_NAME,
            dimension=384,  # For all-MiniLM-L6-v2
            metric="cosine",
            spec={
                "environment": Config.PINECONE_ENVIRONMENT,
                "pod_type": "starter"
            }
        )
    
    
    index = pc.Index(Config.PINECONE_INDEX_NAME)
    
    # Index documents
    vectorstore = PineconeVectorStore.from_documents(
        documents=docs,
        embedding=embeddings,
        index_name=Config.PINECONE_INDEX_NAME,
        text_key="text"
    )
    
    print(f"Successfully indexed {len(docs)} documents")
    return vectorstore