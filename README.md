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

## Architecture

<p align="center">
  <img src="images/HippoRag2 clean diagram.png" alt="HippoRAG Architecture" width="800"/>
</p>

<p align="center">
  <img src="images/HippoRag2 diagram2.png" alt="HippoRAG Detailed Flow" width="800"/>
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
| **Query Clarity Detection** | Automatically detects unclear/ambiguous queries |
| **Query Rewrite (GPT-4o-mini)** | Rewrites unclear queries for better understanding |
| **Context-Aware Query Expansion** | Auto-expands queries with relevant keywords (exam dates, fees, etc.) |
| **Multi-Entity Query Decomposition** | Splits complex multi-entity queries for parallel retrieval |
| **University Chunk Tagging** | Auto-tags document chunks with source university for accurate filtering |
| **University-Based Filtering** | Post-retrieval filtering ensures results match queried university |
| **Contextual Not-Found Responses** | Helpful responses with relevant links when information unavailable |
| **Answer Verification** | Response validation against source documents |
| **Multilingual Support** | English and Bangla (including Banglish) language support |
| **Improved NER** | Enhanced named entity recognition and triple extraction |

### LLM Stack
| Purpose | Model | Location |
|---------|-------|----------|
| **Query Rewrite** | GPT-4o-mini | OpenAI API |
| **Query Decomposition** | GPT-4o-mini | OpenAI API |
| **Embeddings** | multilingual-e5-large | Local (GPU) |
| **Reranking** | ms-marco-MiniLM-L-6-v2 | Local (CPU) |
| **Answer Generation** | Qwen3-80B (configurable) | Ollama (Remote) |

### Supported Models
| Type | Supported Models |
|------|------------------|
| **LLM Backends** | OpenAI (GPT-4, GPT-4o-mini), Google Gemini, Ollama, Local models (vLLM) |
| **Embedding Models** | multilingual-e5-large, NV-Embed-v2, GritLM, Gemini Embeddings, OpenAI Embeddings |

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
| `POST` | `/index-folder` | Index all documents from a folder |
| `POST` | `/ask` | Ask a question (with auto query rewrite & expansion) |
| `POST` | `/debug-retrieval` | Debug endpoint to see retrieved passages |
| `GET` | `/health` | Health check |

### Example Requests

```bash
# Ask a question
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "JnU B unit exam kobe?"}'

# Index documents from folder
curl -X POST "http://localhost:8000/index-folder" \
  -H "Content-Type: application/json" \
  -d '{"folder_path": "documents"}'
```

### Query Processing Pipeline

When you call `/ask`, the system automatically:

1. **Query Clarity Check** - Detects if query is unclear/ambiguous
2. **Query Rewrite** - Uses GPT-4o-mini to rewrite unclear queries
3. **Entity Detection** - Identifies universities/entities in query
4. **Query Expansion** - Adds context keywords (exam dates, fees, etc.)
5. **Multi-Entity Decomposition** - Splits multi-entity queries for parallel retrieval
6. **Retrieval & Reranking** - Hybrid search with cross-encoder reranking
7. **Answer Generation** - Grounded QA with source citations
8. **Not-Found Handling** - Contextual responses with helpful links if no answer found

### Not-Found Response Examples

When information is unavailable, the system provides contextual help:

| Question Category | Response Includes |
|-------------------|-------------------|
| Udvash-related (exam, result, batch) | https://udvash.com/HomePage |
| Medical/Dental admission | https://dghs.gov.bd/ |
| Engineering (BUET, CUET, KUET, RUET) | Individual university links |
| Specific University (DU, RU, JnU, etc.) | That university's official site |
| Cluster admission (গুচ্ছ) | https://gstadmission.ac.bd/ |

```
Example Terminal Output:
================================================================================
📥 /ask ENDPOINT - NEW REQUEST
================================================================================
❓ Question: "JnU B unit exam kobe?"
--------------------------------------------------------------------------------
🔍 STEP 0: Query Clarity Check
   ✅ Query is clear, no rewrite needed
--------------------------------------------------------------------------------
🔍 STEP 1: Entity Detection
   ⏱️  Time: 0.01s
   🏷️  Detected 1 entities: [('jnu', 'জগন্নাথ বিশ্ববিদ্যালয়')]
--------------------------------------------------------------------------------
📝 STEP 2: Query Expansion
   ✓ Expanded: "JnU B unit exam kobe? জগন্নাথ বিশ্ববিদ্যালয় ভর্তি পরীক্ষার সময়সূচি..."
--------------------------------------------------------------------------------
✅ SINGLE-ENTITY REQUEST COMPLETE
   📝 Answer Length: 1250 chars
   📚 References: 5
   ⏱️  TOTAL TIME: 2 min 15 sec (135.42s)
================================================================================
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
