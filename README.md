<h1 align="center">Advance RAG</h1>

<p align="center">
  <strong>A production-ready Retrieval-Augmented Generation system with Knowledge Graph capabilities</strong>
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#installation">Installation</a> •
  <a href="#api-reference">API Reference</a> •
  <a href="#configuration">Configuration</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python"/>
  <img src="https://img.shields.io/badge/License-Research-green.svg" alt="License"/>
  <img src="https://img.shields.io/badge/LLM-Multi--Provider-orange.svg" alt="LLM"/>
</p>

---

## Overview

Advance RAG is an enterprise-grade RAG system built on [HippoRAG](https://github.com/OSU-NLP-Group/HippoRAG), designed for high-accuracy question answering in multilingual environments. It combines knowledge graph-based retrieval, hybrid search strategies, and advanced query processing to deliver precise, source-cited responses.

**Key Differentiators:**
- Neurobiologically-inspired memory architecture for complex reasoning
- Native support for Bengali, English, and Banglish queries
- Multi-stage retrieval with cross-encoder reranking
- Intelligent query decomposition for multi-entity questions

---

## Architecture

<p align="center">
  <img src="images/HippoRag2 clean diagram.png" alt="System Architecture" width="800"/>
</p>

<p align="center">
  <img src="images/HippoRag2 diagram2.png" alt="Processing Pipeline" width="800"/>
</p>

---

## Features

### Core Capabilities

| Capability | Description |
|------------|-------------|
| **Knowledge Graph Retrieval** | Graph-based context understanding using entity relationships and semantic connections |
| **Hybrid Search** | Combines BM25 lexical matching with dense vector retrieval for comprehensive coverage |
| **Cross-Encoder Reranking** | Neural reranking using BAAI/bge-reranker-v2-m3 for precision optimization |
| **Grounded QA** | Source-cited responses with built-in hallucination prevention mechanisms |

### Query Processing

| Feature | Description |
|---------|-------------|
| **Query Clarity Detection** | Automatic detection of ambiguous or unclear queries |
| **Query Rewriting** | GPT-4o-mini powered query reformulation for improved retrieval |
| **Context-Aware Expansion** | Automatic query expansion with domain-relevant keywords |
| **Multi-Entity Decomposition** | Intelligent splitting of complex queries for parallel retrieval |
| **University Chunk Tagging** | Source-aware document tagging for accurate institutional filtering |
| **Post-Retrieval Filtering** | Entity-based filtering ensures results match the queried institution |

### Response Handling

| Feature | Description |
|---------|-------------|
| **Answer Verification** | Automated validation against source documents |
| **Contextual Fallbacks** | Intelligent not-found responses with relevant resource links |
| **Multilingual Output** | Native support for Bengali and English responses |

### Model Stack

| Component | Model | Deployment |
|-----------|-------|------------|
| Query Processing | GPT-4o-mini | OpenAI API |
| Embeddings | multilingual-e5-large | Local (GPU) |
| Reranking | BAAI/bge-reranker-v2-m3 | Local (CPU) |
| Answer Generation | Qwen3-80B | Ollama (configurable) |

### Supported Integrations

| Type | Options |
|------|---------|
| **LLM Providers** | OpenAI, Google Gemini, Ollama, vLLM |
| **Embedding Models** | multilingual-e5-large, NV-Embed-v2, GritLM, Gemini Embeddings, OpenAI Embeddings |

---

## Installation

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU (recommended for embeddings)
- 16GB+ RAM

### Setup

```bash
# Create environment
conda create -n hipporag python=3.10
conda activate hipporag

# Install package
pip install hipporag
```

### Environment Configuration

```bash
# Required API keys
export OPENAI_API_KEY="your-openai-key"
export GOOGLE_API_KEY="your-google-key"        # Optional: For Gemini
export HF_HOME="/path/to/huggingface/cache"    # Optional: Custom cache path
```

---

## Quick Start

### Python SDK

```python
from hipporag import HippoRAG

# Initialize
rag = HippoRAG(
    save_dir='outputs',
    llm_model_name='gpt-4o-mini',
    embedding_model_name='intfloat/multilingual-e5-large'
)

# Index documents
documents = [
    "Einstein developed the theory of relativity.",
    "The theory revolutionized modern physics.",
    "Einstein was born in Germany in 1879."
]
rag.index(docs=documents)

# Query
results = rag.rag_qa(queries=["Where was Einstein born?"])
```

### REST API

```bash
# Start server
python api_server.py

# Index documents
curl -X POST "http://localhost:8000/index-folder" \
  -H "Content-Type: application/json" \
  -d '{"folder_path": "documents"}'

# Query
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "JnU B unit exam kobe?"}'
```

---

## API Reference

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/index` | Index document array |
| `POST` | `/index-folder` | Index documents from directory |
| `POST` | `/ask` | Submit question with full pipeline processing |
| `POST` | `/debug-retrieval` | Retrieve passages without answer generation |
| `GET` | `/health` | Service health check |

### Query Processing Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                     REQUEST PROCESSING                          │
├─────────────────────────────────────────────────────────────────┤
│  1. Query Clarity Check    → Detect ambiguous queries           │
│  2. Query Rewrite          → GPT-4o-mini reformulation          │
│  3. Entity Detection       → Identify institutions/entities     │
│  4. Query Expansion        → Add domain keywords                │
│  5. Multi-Entity Split     → Decompose complex queries          │
│  6. Hybrid Retrieval       → BM25 + Dense + KG traversal        │
│  7. Cross-Encoder Rerank   → Precision optimization             │
│  8. Answer Generation      → Grounded response with citations   │
│  9. Fallback Handling      → Contextual not-found responses     │
└─────────────────────────────────────────────────────────────────┘
```

### Contextual Fallback Responses

When information is unavailable, the system provides category-specific guidance:

| Query Category | Fallback Resource |
|----------------|-------------------|
| Internal platform queries | Platform helpdesk/website |
| Medical/Dental admission | DGHS official portal |
| Engineering universities | Individual institution websites |
| General university queries | Respective university portals |
| Cluster admission | GST Admission portal |

---

## Configuration

### Model Switching

The API server supports easy switching between different LLM providers for answer generation. Edit `api_server.py` line 28:

```python
ANSWER_MODEL = "qwen3-80b"  # Change this to switch models
```

**Available Presets:**

| Preset | Model | Description |
|--------|-------|-------------|
| `gpt-4o-mini` | OpenAI GPT-4o-mini | Fast, cheap, good for testing |
| `gpt-4o` | OpenAI GPT-4o | Slower, expensive, better quality |
| `qwen3-80b` | Qwen3-next 80B (Ollama) | Local, free, 32K context |

**Multi-Model Architecture:**
- **NER/Triple Extraction**: GPT-4o (OpenAI) - accurate entity extraction
- **Answer Generation**: Configurable via `ANSWER_MODEL`
- **Embeddings**: multilingual-e5-large (local GPU)
- **Reranking**: bge-reranker-v2-m3 (local CPU)

### HippoRAG Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `save_dir` | str | required | Directory for indexes and graphs |
| `llm_model_name` | str | required | LLM model identifier |
| `embedding_model_name` | str | required | Embedding model identifier |
| `llm_base_url` | str | None | Custom LLM endpoint |
| `embedding_base_url` | str | None | Custom embedding endpoint |

### Retrieval Tuning

| Parameter | Description |
|-----------|-------------|
| `MIN_REFERENCE_SCORE` | Minimum score threshold for references (default: 0.4) |
| `hybrid_alpha` | Balance between dense and sparse retrieval (0-1) |

---

## Project Structure

```
advance-rag/
├── src/hipporag/
│   ├── HippoRAG.py                 # Core RAG implementation
│   ├── embedding_model/            # Embedding backends
│   ├── llm/                        # LLM provider integrations
│   ├── prompts/templates/          # Prompt engineering
│   └── retrieval/                  # BM25, rerankers, hybrid search
├── api_server.py                   # FastAPI REST server
├── visualize_kg_web.py             # Knowledge graph visualization
└── documents/                      # Document store
```

---

## Visualization

Launch the interactive knowledge graph explorer:

```bash
python visualize_kg_web.py
```

Features:
- Entity node visualization with relationship edges
- Query path highlighting
- Interactive graph exploration
- Document-to-entity mapping

---

## Performance Considerations

- **Indexing**: ~2-5 minutes per 100 documents (GPU recommended)
- **Query latency**: 30-120 seconds depending on complexity
- **Memory**: 8GB minimum, 16GB+ recommended for large indexes

---

## License

This project is intended for educational and research purposes.

---

## Acknowledgments

<p align="center">
  Built upon <a href="https://github.com/OSU-NLP-Group/HippoRAG"><strong>HippoRAG</strong></a> by OSU NLP Group
  <br/>
  <sub>Neurobiologically Inspired Long-Term Memory for Large Language Models</sub>
</p>
