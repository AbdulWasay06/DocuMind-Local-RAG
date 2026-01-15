import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

# --- Page Config ---
st.set_page_config(page_title="Local RAG Chat", page_icon="🤖")
st.title("🤖 Chat with your PDF (Local AI)")

# --- Session State (Memory) ---
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# --- Sidebar: Upload PDF ---
with st.sidebar:
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    if uploaded_file and st.button("Process PDF"):
        with st.spinner("Processing..."):
            # Save the file temporarily
            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # 1. Load & Split
            loader = PyPDFLoader("temp.pdf")
            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(docs)
            
            # 2. Create Embeddings
            embeddings = OllamaEmbeddings(model="nomic-embed-text")
            st.session_state.vectorstore = FAISS.from_documents(splits, embeddings)
            st.success("PDF Loaded! You can now chat.")

# --- Chat Interface ---
if st.session_state.vectorstore is not None:
    llm = ChatOllama(model="tinyllama")
    # Initialize LLM
   
    
    # User Input
    user_query = st.text_input("Ask a question about your document:")
    
    if user_query:
        # Create Chain
        retriever = st.session_state.vectorstore.as_retriever()
        prompt = ChatPromptTemplate.from_template("""
        Answer based on context:
        <context>{context}</context>
        Question: {input}
        """)
        document_chain = create_stuff_documents_chain(llm, prompt)
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        with st.spinner("Thinking..."):
            response = retrieval_chain.invoke({"input": user_query})
            st.write("### Answer:")
            st.write(response["answer"])
else:
    st.info("👈 Please upload a PDF in the sidebar to start.")