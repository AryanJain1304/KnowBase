# RAG Smart Assistant Chat

A **Retrieval-Augmented Generation (RAG)** assistant built with **Streamlit**, **FAISS**, **SentenceTransformers**, and **Ollama**.  
It lets you **upload documents** (PDF, Markdown, or text), builds a local vector index, and allows you to **chat with your data** — all locally.


## Features

- Upload and index your documents (PDF, Markdown, TXT)
- Local vector search using FAISS + MiniLM embeddings
- Context-aware chat using Ollama LLMs (default: `mistral`)
- Cited responses with snippet-based grounding
- Fast and private — everything runs locally


## Tech Stack

- [Streamlit](https://streamlit.io/) — UI and interaction  
- [Ollama](https://ollama.ai/) — Local LLM serving API  
- [SentenceTransformers](https://www.sbert.net/) — Text embeddings (`all-MiniLM-L6-v2`)  
- [FAISS](https://faiss.ai/) — Vector similarity search  
- [PyPDF2](https://pypi.org/project/PyPDF2/) — PDF parsing  



## Requirements

- **Python 3.10+**
- **Streamlit**
- **Requests**
- **FAISS**
- **PyPDF2**
- **SentenceTransformers**
- **Ollama** (with a supported LLM like `mistral`, `llama2`, etc.)

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/AryanJain1304/KnowBase.git
cd rag-smart-assistant
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Install and run Ollama
- Download Ollama from https://ollama.ai

- Start Ollama and pull your preferred model:
```bash
ollama pull mistral
```
- Ensure Ollama is running:
```bash
ollama run mistral
```

## Folder Structure
```bash
rag-smart-assistant/
│
├─ app.py                  # Main Streamlit app
├─ ollama_client.py        # LLM query streaming helper
├─ rag_index.py            # RAG indexing and retrieval logic
├─ uploads/                # Folder for uploaded documents
├─ requirements.txt        # Python dependencies
├─ assets/                 # Optional: screenshots, images
└─ README.md
```

## Usage
- Once setup is complete, launch the Streamlit app:
```bash
streamlit run app.py
```
- Upload your documents in the sidebar.
- Build index by clicking "Ingest & Build Index".
- Ask questions using the chat input.
- View answers with concise citation snippets and confidence scores.
