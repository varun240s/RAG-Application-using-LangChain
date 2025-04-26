
# " streamlit run query.py --server.fileWatcherType none " use this command to avoid errors while the application is running


import asyncio
import nest_asyncio
nest_asyncio.apply()        # Fixes async conflicts in Jupyter-like environments
                        # basically this is not used in vscode but in case if you are using colab,jupyter etc.. better to use this method

import os
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"   
# Disables file watcher (fixes PyTorch error)
os.environ["PYTORCH_JIT"] = "0"  
# Disables PyTorch JIT compilation (may improve performance)
import torch
torch._C._jit_set_profiling_mode(False)
# Disables profiling (reduces overhead)

import streamlit as st
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from langchain_groq import ChatGroq
from pinecone import Pinecone
from config import Config
import torch
from dotenv import load_dotenv




load_dotenv()


# we use cache_resource to avoid re-initializing the Pinecone client on every run.
@st.cache_resource(show_spinner="Initializing Pinecone...")
def init_pinecone():
    try:
        pc = Pinecone(
            api_key=Config.PINECONE_API_KEY,
            timeout=30
        )
        return pc
    except Exception as e:
        st.error(f"Pinecone initialization failed: {str(e)}")
        st.stop()

@st.cache_resource(show_spinner="Configuring RAG system...")
def configure_rag_system():
    try:
        pc = init_pinecone()
        index_response = pc.list_indexes()
        available_indexes = index_response.names() if hasattr(index_response, 'names') else index_response

        if Config.PINECONE_INDEX_NAME not in available_indexes:
            st.error(f"Index '{Config.PINECONE_INDEX_NAME}' not found!")
            st.stop()
            
        embeddings = HuggingFaceEmbeddings(
            model_name=Config.EMBED_MODEL,
            model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
        )
        
        vectorstore = PineconeVectorStore.from_existing_index(
            index_name=Config.PINECONE_INDEX_NAME,
            embedding=embeddings,
            text_key="text"
        )
        
        llm = ChatGroq(
            model_name=Config.LLM_MODEL,
            groq_api_key=Config.GROQ_API_KEY,
            temperature=Config.TEMPERATURE,
            model_kwargs={"top_p": 0.9},  # Fixed top_p location
            max_tokens=Config.MAX_TOKENS
        )
        
        return RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
            return_source_documents=True
        )
        
    except Exception as e:
        st.error(f"RAG configuration failed: {str(e)}")
        st.stop()

def main():
    st.set_page_config(
        page_title="LangChain Chatbot",
        page_icon="🤖",
        layout="wide"
    )
    
    qa_chain = configure_rag_system()

    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant",
            "content": "Hello! I'm your AI assistant. Ask me anything!"
        }]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask your question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            try:
                with st.spinner("Thinking..."):
                    result = qa_chain.invoke({"query": prompt})  # Fixed invocation
                    response = result["result"]
                    
                    if "source_documents" in result:
                        with st.expander("See sources"):
                            for doc in result["source_documents"]:
                                st.write(doc.metadata.get("source", "Unknown source"))
                    
                    st.markdown(response)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response
                    })
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")

    if st.button("🔄 Reset Conversation"):
        st.session_state.messages = [{
            "role": "assistant",
            "content": "Hi again! How can I help you?"
        }]



if __name__ == "__main__":
    main()
















