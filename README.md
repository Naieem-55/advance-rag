<h1 align="center">Advance RAG</h1>

<p align="center">
  <strong>A powerful Retrieval-Augmented Generation system with Knowledge Graph capabilities</strong>
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#installation">Installation</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#api-reference">API</a> •
  <a href="#visualization">Visualization</a>
</p>

---

## Overview

Advance RAG is an enhanced RAG system built on top of HippoRAG, designed for high-accuracy question answering with knowledge graph-based retrieval. It combines multiple retrieval strategies, cross-encoder reranking, and grounded QA to deliver precise, source-cited responses.

## Features

### Core Capabilities
| Feature | Description |
|---------|-------------|
| **Knowledge Graph Retrieval** | Graph-based context understanding for complex queries |
| **Hybrid Search** | BM25 + dense retrieval for comprehensive document matching |
| **Cross-Encoder Reranking** | Neural reranking for improved result relevance |
| **Grounded QA** | Source-cited answers with hallucination prevention |

### Advanced Features
| Feature | Description |
|---------|-------------|
| **Query Expansion** | Automatic query enhancement for better recall |
| **Answer Verification** | Response validation against source documents |
| **Multilingual Support** | English and Bangla language support |
| **Improved NER** | Enhanced named entity recognition and triple extraction |

### LLM & Embedding Support
| Type | Supported Models |
|------|------------------|
| **LLM Backends** | OpenAI (GPT-4, GPT-4o), Google Gemini, Local models (vLLM) |
| **Embedding Models** | NV-Embed-v2, GritLM, Gemini Embeddings, OpenAI Embeddings |

## Installation

### Prerequisites
- Python 3.10+
- CUDA-compatible GPU (recommended)

### Setup

```bash
# Create conda environment
conda create -n hipporag python=3.10
conda activate hipporag

# Install dependencies
pip install hipporag
```

### Environment Variables

```bash
export OPENAI_API_KEY="your-openai-key"        # For OpenAI models
export GOOGLE_API_KEY="your-google-key"        # For Gemini models
export HF_HOME="/path/to/huggingface/cache"    # HuggingFace cache
```

## Quick Start

### Basic Usage

```python
from hipporag import HippoRAG

# Initialize
hipporag = HippoRAG(
    save_dir='outputs',
    llm_model_name='gpt-4o-mini',
    embedding_model_name='nvidia/NV-Embed-v2'
)

# Index documents
docs = [
    "Einstein developed the theory of relativity.",
    "The theory of relativity revolutionized physics.",
    "Einstein was born in Germany in 1879."
]
hipporag.index(docs=docs)

# Query
results = hipporag.rag_qa(queries=["Where was Einstein born?"])
print(results)
```

### Using Gemini

```python
hipporag = HippoRAG(
    save_dir='outputs',
    llm_model_name='gemini-1.5-flash',
    embedding_model_name='models/text-embedding-004'
)
```

## API Reference

### REST API Server

Start the API server:

```bash
python api_server.py
```

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/index` | Index new documents |
| `POST` | `/ask` | Ask a question |
| `GET` | `/health` | Health check |

### Example Request

```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the theory of relativity?"}'
```

## Visualization

### Knowledge Graph Visualization

Launch the interactive knowledge graph viewer:

```bash
python visualize_kg_web.py
```

This opens a web-based visualization showing:
- Entity nodes and relationships
- Query path highlighting
- Interactive graph exploration

## Project Structure

```
advance-rag/
├── src/hipporag/
│   ├── HippoRAG.py              # Main HippoRAG class
│   ├── embedding_model/         # Embedding model implementations
│   ├── llm/                     # LLM backends (OpenAI, Gemini, vLLM)
│   ├── prompts/templates/       # Prompt templates
│   └── retrieval/               # BM25 & reranker implementations
├── api_server.py                # REST API server
├── visualize_kg_web.py          # Knowledge graph visualization
└── documents/                   # Sample documents
```

## Configuration

### HippoRAG Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `save_dir` | str | Directory for saving indexes and graphs |
| `llm_model_name` | str | LLM model identifier |
| `embedding_model_name` | str | Embedding model identifier |
| `llm_base_url` | str | Custom LLM API endpoint (optional) |
| `embedding_base_url` | str | Custom embedding API endpoint (optional) |

## License

This project is for educational and research purposes.

---

## Special Thanks

<p align="center">
  This project is built upon the excellent work of the OSU NLP Group.
</p>

<p align="center">
  <a href="https://github.com/OSU-NLP-Group/HippoRAG">
    <strong>HippoRAG - OSU-NLP-Group</strong>
  </a>
</p>

<p align="center">
  <sub>Original HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models</sub>
</p>
