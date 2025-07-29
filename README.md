# Local RAG QnA – Ask Questions from Your PDFs with AI

A local, flexible Retrieval-Augmented Generation (RAG) system that lets you upload a PDF and ask questions about its content using Google Gemini or local LLMs (via Ollama). It supports intelligent chunking, vector indexing with OpenSearch, and a beautiful Gradio-powered interface.

## Features

### 1. Smart PDF Chunking with Gemini
- Extracts **images and their captions**, **tables with descriptions**, and **paragraphs**
- Uses **Google Gemini API** to semantically group and process document structure
- Produces **semantic chunks** ready for embedding and retrieval

### 2. Local Embedding via `nomic-embed-text`
- Uses `nomic-embed-text` model running in **Ollama (locally)** to generate vector embeddings
- Sends requests to: `http://localhost:11434/api/embeddings/`
- Fully offline and fast — no external API for embeddings

### 3. OpenSearch Indexing & Ingestion
- Automatically creates OpenSearch index (if missing) with **vector mapping**
- Embeds each chunk and stores it in OpenSearch with metadata (type, token count)
- Scalable ingestion using bulk API

### 4. Flexible Search Options
Supports 3 retrieval strategies:
- **Keyword Search** – Exact text match using OpenSearch `match` queries
- **Semantic Search** – Vector similarity via `knn_vector`
- **Hybrid Search** – Combines keyword + vector results with hybrid scoring

### 5. Dual-Backend Answer Generation
Choose between:
- **Google Gemini Pro API** – High-quality, cloud-based reasoning
- **DeepSeek via Ollama (Docker)** – Local LLM for offline, cost-free inference

### 6. Gradio UI Interface
A responsive frontend built with **Gradio Blocks**:
- Upload PDF
- Enter natural language questions
- Choose LLM (Gemini or Ollama)
- Choose retrieval strategy (Keyword / Semantic / Hybrid)
- Enable or disable streaming
- View answers in real time

### 7. Dockerized OpenSearch Setup
Includes `docker-compose.yml` to launch:
- **OpenSearch** (2.11.0) – Vector search backend on port `9200`
- **OpenSearch Dashboards** – GUI interface on port `5601` to explore indexes

---

## Tech Stack

| Component       | Tool/Service                             |
|-----------------|------------------------------------------|
| UI              | Gradio                                   |
| Chunking        | Gemini API                               |
| Embeddings      | nomic-embed-text via Ollama in Docker    |
| Vector DB       | OpenSearch                               |
| LLM (Cloud)     | Google Gemini Pro                        |
| LLM (Local)     | DeepSeek (Ollama in Docker)              |
| PDF Parsing     | unstructured, PyMuPDF, Tesseract         |
| Containerization| Docker                                   |

## Project Structure
RagPdfChatbot/
├── app.py # Gradio frontend
├── chunking.py # PDF processing & Gemini chunking
├── helper.py # Embedding + OpenSearch connection
├── ingestion.py # Index creation and document ingestion
├── retrieval.py # Keyword, semantic, and hybrid search
├── generation.py # LLM response generation (Gemini / Ollama)
├── docker-compose.yml # OpenSearch + Dashboard setup
├── .env # Secrets (API keys, etc.)
├── requirements.txt
└── README.md


