import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

# --- 1. Page Configuration (The "Look") ---
st.set_page_config(
    page_title="DocuMind AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 2. Custom CSS (The "Paint") ---
st.markdown("""
    <style>
    /* Main Background */
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }
    
    /* Chat Input Styling */
    .stTextInput > div > div > input {
        background-color: #262730;
        color: #ffffff;
        border-radius: 20px;
    }
    
    /* Header Styling */
    h1 {
        text-align: center;
        background: -webkit-linear-gradient(45deg, #00C9FF, #92FE9D);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
        padding-bottom: 20px;
    }
    
    /* Upload Box Styling */
    .stFileUploader {
        border: 2px dashed #00C9FF;
        border-radius: 10px;
        padding: 20px;
        background-color: #1a1c24;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. Header ---
st.title("🧠 DocuMind: Chat with Documents")
st.markdown("<p style='text-align: center; color: #b0b3b8;'>Upload your PDF below and ask anything. Local. Secure. Fast.</p>", unsafe_allow_html=True)

# --- 4. Session State (Memory) ---
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 5. The File Uploader (Top of Page) ---
# We use a container to keep it neat
with st.container():
    uploaded_file = st.file_uploader("📂 Drop your PDF here", type="pdf")
    
    if uploaded_file and st.session_state.vectorstore is None:
        with st.status("⚙️ Processing Document...", expanded=True) as status:
            st.write("Reading PDF...")
            # Save temp file
            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Load & Split
            loader = PyPDFLoader("temp.pdf")
            docs = loader.load()
            st.write("Splitting text into chunks...")
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(docs)
            
            # Embed
            st.write("Creating Vector Database...")
            embeddings = OllamaEmbeddings(model="nomic-embed-text")
            st.session_state.vectorstore = FAISS.from_documents(splits, embeddings)
            
            status.update(label="✅ Document Ready!", state="complete", expanded=False)

# --- 6. Chat Interface ---
# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input
if st.session_state.vectorstore is not None:
    prompt = st.chat_input("Ask a question about your PDF...")

    if prompt:
        # 1. Show User Message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. Generate AI Answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Initialize Model (Using tinyllama for speed, change to llama3.2 for smarts)
                llm = ChatOllama(model="tinyllama")
                
                # Retrieve Context
                retriever = st.session_state.vectorstore.as_retriever()
                
                # The Prompt Template
                prompt_template = ChatPromptTemplate.from_template("""
                You are a helpful AI assistant. Answer the question based ONLY on the context provided.
                If the answer is not in the context, simply say "I don't know."
                
                Context: {context}
                
                Question: {input}
                """)
                
                # Build Chain
                document_chain = create_stuff_documents_chain(llm, prompt_template)
                retrieval_chain = create_retrieval_chain(retriever, document_chain)
                
                # Get Response
                response = retrieval_chain.invoke({"input": prompt})
                answer = response["answer"]
                
                # Display Answer
                st.markdown(answer)
        
        # 3. Save AI Message
        st.session_state.messages.append({"role": "assistant", "content": answer})

else:
    if not uploaded_file:
        st.info("👆 Please upload a PDF to start chatting.")