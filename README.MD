A fully local AI agent that analyzes financial 10-K filings and answers strategic investment questions without sending data to the cloud.

Built with **LangChain**, **Ollama (Llama 3.2)**, and **HuggingFace Embeddings**.

## 🚀 Why This Matters
- **Zero Cost:** No OpenAI/Gemini API keys required.
- **Privacy First:** Sensitive financial data never leaves your MacBook.
- **No Rate Limits:** Runs entirely on local hardware (optimized for Apple Silicon).

## 🛠️ Tech Stack
- **LLM:** Meta Llama 3.2 (running via Ollama)
- **Vector Store:** ChromaDB (Local Vector Database)
- **Embeddings:** `all-MiniLM-L6-v2` (HuggingFace/SentenceTransformers)
- **Orchestration:** LangChain

## ⚡ Quick Start

### 1. Prerequisites
Install [Ollama](https://ollama.com/) and pull the lightweight model:
```bash
brew install ollama
ollama serve
ollama pull llama3.2
