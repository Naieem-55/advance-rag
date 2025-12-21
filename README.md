# Advance RAG

An advanced Retrieval-Augmented Generation (RAG) system built on top of HippoRAG, featuring Gemini support and knowledge graph visualization.

## Features

- Knowledge graph-based retrieval for improved context understanding
- Support for multiple LLM backends (OpenAI, Gemini, local models via vLLM)
- Multiple embedding model support (NV-Embed, GritLM, Gemini, OpenAI)
- BM25 retriever for hybrid search
- Cross-encoder reranker for improved result ranking
- Grounded QA with source citations
- Knowledge graph visualization tools
- REST API server for easy integration

## Installation

```bash
conda create -n hipporag python=3.10
conda activate hipporag
pip install hipporag
```

## Quick Start

```python
from hipporag import HippoRAG

# Initialize HippoRAG
hipporag = HippoRAG(
    save_dir='outputs',
    llm_model_name='gpt-4o-mini',
    embedding_model_name='nvidia/NV-Embed-v2'
)

# Index documents
docs = [
    "Your document text here.",
    "Another document."
]
hipporag.index(docs=docs)

# Query
queries = ["Your question here?"]
results = hipporag.rag_qa(queries=queries)
```

## API Server

Run the API server for REST-based access:

```bash
python api_server.py
```

## Visualization

Visualize the knowledge graph:

```bash
python visualize_kg_web.py
```

## Special Thanks

This project is based on the original HippoRAG framework. Special thanks to the OSU NLP Group for their excellent work:

- [HippoRAG - OSU-NLP-Group](https://github.com/OSU-NLP-Group/HippoRAG)
