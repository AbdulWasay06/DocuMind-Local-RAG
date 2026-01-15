# 🧠 DocuMind: Secure Local RAG Assistant

DocuMind is a privacy-first Generative AI application designed to analyze sensitive documents locally. Unlike cloud-based solutions (like ChatGPT), DocuMind processes all data—from ingestion to inference—entirely on your local machine, ensuring **Zero Data Leakage**.

It leverages the power of **Retrieval-Augmented Generation (RAG)** to provide accurate, context-aware answers from your own PDF files without ever connecting to the internet.

---

## 📸 Application Demo

### 1. The Interface (Input)
*(Upload a screenshot of the clean UI with the file uploader here)*
![Input Interface](path/to/your-input-image.png)

### 2. AI Analysis (Output)
*(Upload a screenshot of the Chatbot answering a question from the PDF)*
![AI Output](path/to/your-output-image.png)

---

## 🚀 Key Features

* **🔒 100% Privacy:** Runs entirely offline using local LLMs. No API keys required, and no data leaves the device.
* **📂 Drag & Drop Analysis:** Instantly ingest and process PDF documents.
* **🧠 Context-Aware Memory:** Remembers previous questions in the chat session for a natural conversation flow.
* **⚡ Optimized Performance:** Uses quantized models for fast inference on standard consumer hardware.
* **🎨 Modern UI:** Built with a custom-styled Streamlit interface featuring dark mode and responsive design.

---

## 🛠️ Tech Stack

This project was engineered using the following technologies:

| Component | Technology Used | Purpose |
| :--- | :--- | :--- |
| **LLM Engine** | **Ollama** (Llama 3.2 / TinyLlama) | Local inference engine for generating answers. |
| **Orchestration** | **LangChain** | Manages the document splitting, retrieval chains, and prompt engineering. |
| **Vector Database** | **FAISS** | Facebook AI Similarity Search for efficient vector storage and retrieval. |
| **Embeddings** | **Nomic-Embed-Text** | Converts text chunks into high-dimensional vector representations. |
| **Frontend** | **Streamlit** | Provides the interactive web-based user interface. |
| **Language** | **Python** | Core programming language for logic and integration. |

---

## ⚙️ How It Works (Architecture)

1.  **Ingestion:** The application loads the PDF and splits the text into manageable chunks (e.g., 1000 characters).
2.  **Embedding:** Each chunk is converted into a vector (a list of numbers) using the `nomic-embed-text` model.
3.  **Storage:** These vectors are stored in a temporary, in-memory **FAISS** vector store.
4.  **Retrieval:** When a user asks a question, the system searches the vector store for the most relevant text chunks.
5.  **Generation:** The retrieved chunks + the user's question are sent to the **Llama 3.2** model, which generates a precise answer based *only* on that context.

---

**👨‍💻 Developed by Mohd Abdul Wasay**
