# 🚀 Retrieval-Augmented Generation (RAG) System

## 📌 Overview

This is a **Retrieval-Augmented Generation (RAG) system** that combines **hybrid search** with **LLM-based text generation** to answer user queries. It retrieves relevant context from a document database and generates responses using an LLM (e.g., GPT-3.5).

## 🚀 Features

✅ **Hybrid Retrieval** - Uses **sparse (BM25) & dense (vector-based) search**.\
✅ **LLM-based Response Generation** - Uses **GPT-3.5/Mistral** for text generation.\
✅ **ChromaDB Integration** - Stores and retrieves document embeddings.\
✅ **Caching with Redis** (Planned) - Will speed up frequent queries & remember previous chats.\
✅ **Gradio UI** - Provides a user-friendly chatbot interface.

## 📂 Project Structure

```
data/                    # Stores raw and processed data
src/
  ├── rag/               # Core RAG components
  │   ├── config/        # Stores configuration files (e.g., API keys, ChromaDB settings)
  │   ├── data/          # Handles text preprocessing (e.g., chunking, splitting)
  │   ├── generator/     # Text generation using LLM (GPT, Mistral, etc.)
  │   ├── pipeline/      # Main pipeline orchestrating retrieval + inference
  │   ├── retriever/     # Hybrid search retrieval (Sparse + Dense retrieval)
  │   ├── vector_db/     # Manages ChromaDB vector storage
  │   ├── vectorizer/    # Handles embedding model (text -> vectors)
  ├── run.py             # Gradio interface for RAG chatbot
  ├── Dockerfile         # Docker container for Gradio app
  ├── docker-compose.yml # Runs Gradio + ChromaDB together (Under Maintenance)
```

## 📌 Environment Variables

Set these variables before running the app:

```
CHROMADB_HOST=localhost
CHROMADB_PORT=8000
OPENAI_API_KEY=your-api-key-here
```

## ⚙️ Installation & Setup

### **1️⃣ Clone the Repository**

```bash
git clone https://github.com/your-repo/rag-project.git
cd rag-project
```

### **2️⃣ Install Dependencies**

```bash
pip install -r requirements/main.txt
```

### **3️⃣ Run the RAG System**

#### **Option 1: Run Locally**

start chromadb server using docker
```bash
docker pull chromadb/chroma
docker run -p 8000:8000 chromadb/chroma
```
then run the app
```bash
python app.py
```

This will start the **Gradio UI** at `http://localhost:7860`

#### **Option 2: Run with Docker (Under Maintenance)**

```bash
docker-compose up --build
```

**Note:** Docker setup is still under development and may not be fully functional.

## 🤖 How It Works

1. **User asks a question.**
2. **Retriever finds relevant document chunks** using hybrid search.
3. **LLM generates an answer** using retrieved context.
4. **Cache stores previous queries** to avoid redundant processing (Planned).

## 🛠️ Customization

### **Change LLM Model**

Modify `src/rag/inference/text_generator.py` to use different models (e.g., OpenAI API, Mistral, LLaMA).

### **Modify Retrieval Settings**

Adjust **hybrid search parameters** in `src/rag/retriever/hybrid_search.py`:

```python
relevant_chunks = self.retriever.hybrid_search(
    query=query,
    top_n=5,  # Number of retrieved documents
    sparse_weight=0.5,
    dense_weight=0.5
)
```

## 🏗️ Future Improvements

- **Implement Multi-turn Conversation Memory**
- **Add support for more LLM models (e.g., LLaMA, Falcon)**
- **Improve UI with Streamlit or FastAPI**
- **Complete Docker setup for full deployment**
- **Implement Redis caching for faster responses**

---

**📬 Questions?** Feel free to reach out! 🚀
