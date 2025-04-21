
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings 
from pinecone import Pinecone

# Initialize embedding model (LangChain-compatible)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)

# Initialize Pinecone client
pc = Pinecone(api_key="pcsk_2sx1dn_9pzfkUivqyaha6iXWUJS7owK2t8Uz2BVk5C5mq3o7ki5vPtcPuS1DNUxWHWwQZ1")

# give you pinnecone index_name

index_name = "quickstart"
index = pc.Index(index_name)

# Create vector store using new class
vectorstore = PineconeVectorStore(
    index=index,
    embedding=embeddings,
    text_key="text"
)

# Create retriever
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 5}
)

# Perform search
query = "what is pediatric asthma symptoms"
results = retriever.invoke(query)

for doc in results:
    print(f"Content: {doc.page_content[:200]}...")
    print(f"Metadata: {doc.metadata}")
    print("---")
