"""
HippoRAG API Server
Test your knowledge graph QA system via Postman or any HTTP client
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import os
import glob

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Verify Gemini API key is set (used for both LLM and Embeddings)
if not os.getenv("GEMINI_API_KEY"):
    print("WARNING: GEMINI_API_KEY not set!")
    print("Please add your Gemini API key to .env file")
else:
    print("Gemini API Key loaded (for LLM + Embeddings)")

# Initialize FastAPI app
app = FastAPI(
    title="HippoRAG API",
    description="Knowledge Graph based RAG Question Answering API",
    version="1.0.0"
)

# Enable CORS for Postman and browser testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global HippoRAG instance
hipporag_instance = None

# Request/Response Models
class QuestionRequest(BaseModel):
    question: str
    language_instruction: Optional[str] = "IMPORTANT: Respond ONLY in the same language as the question. Do NOT mix languages. Do NOT include your reasoning or thought process. Give a direct, concise answer only."

class Reference(BaseModel):
    content: str
    score: float

class AnswerResponse(BaseModel):
    question: str
    answer: str
    references: List[Reference]

class IndexRequest(BaseModel):
    documents: List[str]

class StatusResponse(BaseModel):
    status: str
    message: str
    indexed_docs: int

class DocumentsFromFolderRequest(BaseModel):
    folder_path: str = "documents"


def chunk_text(text: str, max_chars: int = 1500, overlap: int = 200) -> List[str]:
    """Split text into smaller chunks with overlap."""
    if len(text) <= max_chars:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars

        # Try to break at a sentence or paragraph boundary
        if end < len(text):
            # Look for paragraph break
            para_break = text.rfind('\n\n', start, end)
            if para_break > start + max_chars // 2:
                end = para_break
            else:
                # Look for sentence break
                sentence_break = text.rfind('। ', start, end)  # Bangla sentence end
                if sentence_break == -1:
                    sentence_break = text.rfind('. ', start, end)  # English sentence end
                if sentence_break > start + max_chars // 2:
                    end = sentence_break + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = end - overlap if end < len(text) else len(text)

    return chunks


def load_documents_from_folder(folder_path: str) -> List[str]:
    """Load documents from a folder, splitting by page markers and chunking large texts."""
    documents = []
    txt_files = glob.glob(os.path.join(folder_path, "*.txt"))

    for file_path in txt_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Split by page markers if they exist
        if "=== Page" in content:
            pages = content.split("=== Page")
            for page in pages:
                page = page.strip()
                if page and not page.startswith("==="):
                    # Remove the page number line
                    lines = page.split("\n", 1)
                    if len(lines) > 1:
                        page_content = lines[1].strip()
                        if page_content:
                            # Chunk if too large
                            chunks = chunk_text(page_content, max_chars=2000)
                            documents.extend(chunks)
        else:
            # No page markers, chunk the whole content
            if content.strip():
                chunks = chunk_text(content.strip(), max_chars=2000)
                documents.extend(chunks)

    print(f"Loaded {len(documents)} document chunks from {len(txt_files)} files")
    return documents


def get_hipporag():
    """Get or initialize HippoRAG instance."""
    global hipporag_instance

    if hipporag_instance is None:
        raise HTTPException(
            status_code=400,
            detail="HippoRAG not initialized. Call /index or /index-folder first."
        )

    return hipporag_instance


@app.get("/", response_model=StatusResponse)
async def root():
    """Health check and status endpoint."""
    global hipporag_instance

    if hipporag_instance is None:
        return StatusResponse(
            status="not_initialized",
            message="HippoRAG not initialized. Call /index or /index-folder to load documents.",
            indexed_docs=0
        )

    # Get passage count from graph
    passage_count = 0
    if hasattr(hipporag_instance, 'passage_node_idxs'):
        passage_count = len(hipporag_instance.passage_node_idxs)
    elif hasattr(hipporag_instance, 'graph') and hipporag_instance.graph:
        # Count chunk nodes from graph
        for v in hipporag_instance.graph.vs:
            if 'hash_id' in hipporag_instance.graph.vs.attributes():
                if v['hash_id'].startswith('chunk'):
                    passage_count += 1

    return StatusResponse(
        status="ready",
        message="HippoRAG is ready to answer questions.",
        indexed_docs=passage_count
    )


@app.post("/index", response_model=StatusResponse)
async def index_documents(request: IndexRequest):
    """Index a list of documents."""
    global hipporag_instance

    if not request.documents:
        raise HTTPException(status_code=400, detail="No documents provided")

    try:
        from hipporag import HippoRAG

        hipporag_instance = HippoRAG(
            save_dir='outputs',
            llm_model_name='gemini/gemini-2.5-flash',
            embedding_model_name='gemini/gemini-embedding-001'
        )

        hipporag_instance.index(docs=request.documents)

        return StatusResponse(
            status="success",
            message=f"Successfully indexed {len(request.documents)} documents.",
            indexed_docs=len(request.documents)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/index-folder", response_model=StatusResponse)
async def index_from_folder(request: DocumentsFromFolderRequest):
    """Index documents from a folder."""
    global hipporag_instance

    if not os.path.exists(request.folder_path):
        raise HTTPException(status_code=400, detail=f"Folder not found: {request.folder_path}")

    try:
        documents = load_documents_from_folder(request.folder_path)

        if not documents:
            raise HTTPException(status_code=400, detail="No documents found in folder")

        from hipporag import HippoRAG

        hipporag_instance = HippoRAG(
            save_dir='outputs',
            llm_model_name='gemini/gemini-2.5-flash',
            embedding_model_name='gemini/gemini-embedding-001'
        )

        hipporag_instance.index(docs=documents)

        return StatusResponse(
            status="success",
            message=f"Successfully indexed {len(documents)} documents from {request.folder_path}",
            indexed_docs=len(documents)
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """Ask a question and get an answer with references."""
    hipporag = get_hipporag()

    try:
        # Add language instruction
        query_with_instruction = f"{request.question}\n\n({request.language_instruction})"

        # Get answer from HippoRAG
        # Returns: Tuple[List[QuerySolution], List[str], List[Dict]]
        query_solutions, response_messages, metadata_list = hipporag.rag_qa(queries=[query_with_instruction])

        if query_solutions and len(query_solutions) > 0:
            # QuerySolution has: question, docs, doc_scores, answer
            query_solution = query_solutions[0]

            answer = query_solution.answer if query_solution.answer else "No answer found"

            # Extract references from docs and doc_scores
            references = []
            docs = query_solution.docs if query_solution.docs else []
            scores = query_solution.doc_scores if query_solution.doc_scores is not None else []

            for i, doc in enumerate(docs[:5]):  # Top 5 references
                score = float(scores[i]) if i < len(scores) else 0.0
                references.append(Reference(
                    content=doc[:500] + "..." if len(doc) > 500 else doc,
                    score=score
                ))

            return AnswerResponse(
                question=request.question,
                answer=answer,
                references=references
            )
        else:
            return AnswerResponse(
                question=request.question,
                answer="No answer found",
                references=[]
            )

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/debug-retrieval")
async def debug_retrieval(request: QuestionRequest):
    """Debug endpoint to see retrieved passages without QA."""
    hipporag = get_hipporag()

    try:
        query_with_instruction = f"{request.question}\n\n({request.language_instruction})"

        # Get full results
        query_solutions, response_messages, metadata_list = hipporag.rag_qa(queries=[query_with_instruction])

        if query_solutions and len(query_solutions) > 0:
            qs = query_solutions[0]

            # Show all retrieved docs with scores
            retrieved = []
            docs = qs.docs if qs.docs else []
            scores = qs.doc_scores if qs.doc_scores is not None else []

            for i, doc in enumerate(docs):
                score = float(scores[i]) if i < len(scores) else 0.0
                retrieved.append({
                    "rank": i + 1,
                    "score": score,
                    "content": doc
                })

            return {
                "question": request.question,
                "answer": qs.answer,
                "total_retrieved": len(docs),
                "retrieved_passages": retrieved,
                "metadata": metadata_list[0] if metadata_list else {}
            }

        return {"error": "No results"}

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/graph-stats")
async def get_graph_stats():
    """Get knowledge graph statistics."""
    hipporag = get_hipporag()

    try:
        graph = hipporag.graph if hasattr(hipporag, 'graph') else None

        if graph is None:
            return {"message": "Graph not available"}

        # Count node types
        entity_count = 0
        chunk_count = 0

        for v in graph.vs:
            hash_id = v['hash_id'] if 'hash_id' in graph.vs.attributes() else ''
            if hash_id.startswith('entity'):
                entity_count += 1
            elif hash_id.startswith('chunk'):
                chunk_count += 1

        return {
            "total_nodes": graph.vcount(),
            "total_edges": graph.ecount(),
            "entity_nodes": entity_count,
            "chunk_nodes": chunk_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/visualize-query")
async def visualize_query(request: QuestionRequest):
    """Generate a visualization showing which nodes have high relevance for a query."""
    hipporag = get_hipporag()

    try:
        from visualize_query import get_query_relevance_scores, create_query_visualization

        # Get scores
        scores_data = get_query_relevance_scores(hipporag, request.question)

        if "error" in scores_data and scores_data.get("error"):
            return {"error": scores_data["error"]}

        # Create visualization HTML
        output_path = create_query_visualization(hipporag, request.question)

        # Return summary + file path
        result = {
            "query": request.question,
            "visualization_file": output_path,
            "query_entities": scores_data.get("query_entities", []),
            "top_facts": scores_data.get("top_facts", [])[:5],
            "top_passages": scores_data.get("top_passages", [])[:5],
            "total_nodes": scores_data.get("total_nodes", 0),
            "message": f"Visualization saved to {output_path}. Open in browser to view."
        }

        if scores_data.get("warning"):
            result["warning"] = scores_data["warning"]
        if scores_data.get("use_dpr_only"):
            result["mode"] = "DPR only (no knowledge graph facts matched)"

        return result

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/debug-facts")
async def debug_facts(request: QuestionRequest):
    """Debug endpoint to see fact matching and reranking details."""
    hipporag = get_hipporag()

    try:
        import numpy as np

        if not hipporag.ready_to_retrieve:
            hipporag.prepare_retrieval_objects()

        # Get query embedding
        hipporag.get_query_embeddings([request.question])

        # Get fact scores
        query_fact_scores = hipporag.get_fact_scores(request.question)

        # Get top facts before reranking
        link_top_k = hipporag.global_config.linking_top_k

        if len(query_fact_scores) == 0:
            return {
                "error": "No fact scores computed",
                "total_facts_in_index": len(hipporag.fact_node_keys) if hasattr(hipporag, 'fact_node_keys') else 0
            }

        # Get candidate facts
        if len(query_fact_scores) <= link_top_k:
            candidate_fact_indices = np.argsort(query_fact_scores)[::-1].tolist()
        else:
            candidate_fact_indices = np.argsort(query_fact_scores)[-link_top_k:][::-1].tolist()

        candidate_facts_info = []
        for idx in candidate_fact_indices[:20]:  # Top 20
            fact_id = hipporag.fact_node_keys[idx]
            fact_row = hipporag.fact_embedding_store.get_row(fact_id)
            if fact_row:
                candidate_facts_info.append({
                    "fact": fact_row.get('content', ''),
                    "score": float(query_fact_scores[idx]),
                    "fact_id": fact_id
                })

        # Run reranking
        top_k_fact_indices, top_k_facts, rerank_log = hipporag.rerank_facts(request.question, query_fact_scores)

        return {
            "query": request.question,
            "total_facts_in_index": len(hipporag.fact_node_keys),
            "facts_before_rerank": candidate_facts_info,
            "facts_after_rerank": [
                {"subject": f[0], "predicate": f[1], "object": f[2]}
                for f in top_k_facts
            ],
            "rerank_log": rerank_log
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/query-scores/{query}")
async def get_query_scores(query: str):
    """Get PPR scores for all nodes given a query (JSON API)."""
    hipporag = get_hipporag()

    try:
        from visualize_query import get_query_relevance_scores
        scores_data = get_query_relevance_scores(hipporag, query)

        # Return top scored nodes only (to avoid huge response)
        ppr_scores = scores_data.get("ppr_scores", {})
        sorted_nodes = sorted(ppr_scores.items(), key=lambda x: x[1].get('ppr_score', 0), reverse=True)[:50]

        return {
            "query": query,
            "query_entities": scores_data.get("query_entities", []),
            "top_facts": scores_data.get("top_facts", []),
            "top_passages": scores_data.get("top_passages", []),
            "top_nodes_by_ppr": [
                {"name": name, **data}
                for name, data in sorted_nodes
            ]
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reload")
async def reload_from_cache():
    """Reload HippoRAG from existing cache/index."""
    global hipporag_instance

    try:
        from hipporag import HippoRAG

        hipporag_instance = HippoRAG(
            save_dir='outputs',
            llm_model_name='gemini/gemini-2.5-flash',
            embedding_model_name='gemini/gemini-embedding-001'
        )

        # Load existing index if available
        hipporag_instance.load()

        return StatusResponse(
            status="success",
            message="HippoRAG reloaded from cache",
            indexed_docs=len(hipporag_instance.docs) if hasattr(hipporag_instance, 'docs') else 0
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def auto_load_hipporag():
    """Try to auto-load HippoRAG from existing cache on startup."""
    global hipporag_instance

    try:
        from hipporag import HippoRAG
        import os

        # Check if cached data exists
        cache_dir = 'outputs/gemini_gemini-2.5-flash_gemini_gemini-embedding-001'
        if not os.path.exists(cache_dir):
            cache_dir = 'outputs/gemini_gemini-2.0-flash_gemini_gemini-embedding-001'
        if not os.path.exists(cache_dir):
            cache_dir = 'outputs/gpt-4o_text-embedding-3-large'

        if os.path.exists(cache_dir):
            print(f"Found existing cache at {cache_dir}")
            print("Auto-loading HippoRAG from cache...")

            hipporag_instance = HippoRAG(
                save_dir='outputs',
                llm_model_name='gemini/gemini-2.5-flash',
                embedding_model_name='gemini/gemini-embedding-001'
            )

            # Try to load existing index by preparing retrieval objects
            hipporag_instance.prepare_retrieval_objects()
            print("HippoRAG loaded successfully from cache!")
        else:
            print("No existing cache found. Call /index-folder to create index.")

    except Exception as e:
        print(f"Auto-load failed: {e}")
        print("Call /index-folder to initialize HippoRAG.")


if __name__ == "__main__":
    print("="*60)
    print("  HippoRAG API Server")
    print("="*60)
    print("\nEndpoints:")
    print("  GET  /              - Status check")
    print("  POST /index         - Index documents (JSON body)")
    print("  POST /index-folder  - Index from folder")
    print("  POST /ask           - Ask a question")
    print("  POST /debug-retrieval - Debug retrieved passages")
    print("  GET  /graph-stats   - Get graph statistics")
    print("  POST /visualize-query - Visualize query relevance on KG")
    print("  GET  /query-scores/{q} - Get PPR scores for query")
    print("  POST /reload        - Reload from cache")
    print("\nSwagger Docs: http://localhost:8000/docs")
    print("="*60)

    # Auto-load from cache if available
    auto_load_hipporag()

    uvicorn.run(app, host="127.0.0.1", port=8000)
