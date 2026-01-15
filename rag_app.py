from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# --- 1. Load the PDF ---
print("Loading PDF...")
loader = PyPDFLoader("DL Concepts.pdf")
docs = loader.load()

# --- 2. Split Text into Chunks ---
# We can't feed the whole PDF at once. We cut it into small pieces (chunks).
print("Splitting text...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# --- 3. Create Embeddings & Vector Store ---
# This converts text into numbers so the AI can search it.
print("Creating database (this might take a moment)...")
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)

# --- 4. Create the Retriever ---
# This tool will search the database for the right chunk of text.
retriever = vectorstore.as_retriever()

# --- 5. Create the Chain ---
# We connect the LLM + The Prompt + The Retriever
llm = ChatOllama(model="llama3.2")

# This template tells the AI: "Use the provided context to answer the question."
prompt_template = """
Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}
"""
prompt = ChatPromptTemplate.from_template(prompt_template)

# Create the "Document Chain" (Handles the answer generation)
document_chain = create_stuff_documents_chain(llm, prompt)

# Create the "Retrieval Chain" (Handles finding the data and then generating answer)
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# --- 6. Ask a Question ---
while True:
    user_query = input("\nAsk a question from your PDF (or type 'exit'): ")
    if user_query.lower() == "exit":
        break
    
    print("Thinking...")
    response = retrieval_chain.invoke({"input": user_query})
    
    print("\nAnswer:")
    print(response["answer"])