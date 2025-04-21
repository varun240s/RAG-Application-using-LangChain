

import os
from dotenv import load_dotenv
from pinecone import Pinecone

# Load .env variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
INDEX_NAME = "quickstart" 

# Connecting to Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY) 
index = pc.Index(INDEX_NAME)
print(f"Using existing index: {INDEX_NAME}")
