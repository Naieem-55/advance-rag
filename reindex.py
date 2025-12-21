"""
Re-index script for HippoRAG
Run this after updating NER prompts to rebuild the knowledge graph
"""

import os
import shutil
import json

def clear_cache():
    """Clear cached OpenIE results, graph files, and LLM cache."""
    files_to_delete = [
        "outputs/openie_results_ner_gemini_gemini-2.5-flash.json",
        "outputs/gemini_gemini-2.5-flash_gemini_gemini-embedding-001/graph.pickle",
        # Clear LLM cache to force re-extraction with new prompts
        "outputs/llm_cache/gemini_gemini-2.5-flash_cache.sqlite",
    ]

    for f in files_to_delete:
        if os.path.exists(f):
            os.remove(f)
            print(f"Deleted: {f}")
        else:
            print(f"Not found (skip): {f}")

    # Clear entity embeddings (will be regenerated)
    entity_dir = "outputs/gemini_gemini-2.5-flash_gemini_gemini-embedding-001/entity_embeddings"
    if os.path.exists(entity_dir):
        shutil.rmtree(entity_dir)
        print(f"Deleted directory: {entity_dir}")

    # Clear fact embeddings (will be regenerated)
    fact_dir = "outputs/gemini_gemini-2.5-flash_gemini_gemini-embedding-001/fact_embeddings"
    if os.path.exists(fact_dir):
        shutil.rmtree(fact_dir)
        print(f"Deleted directory: {fact_dir}")

def load_original_documents():
    """
    Load your original documents.
    Modify this function to load YOUR documents.
    """
    # Option 1: Load from a JSON file
    docs_file = "documents.json"
    if os.path.exists(docs_file):
        with open(docs_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict) and 'docs' in data:
                return data['docs']

    # Option 2: Load from a text file (one doc per line)
    txt_file = "documents.txt"
    if os.path.exists(txt_file):
        with open(txt_file, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]

    # Option 3: Load from a directory of text files
    docs_dir = "documents"
    if os.path.exists(docs_dir):
        docs = []
        for filename in os.listdir(docs_dir):
            if filename.endswith('.txt'):
                with open(os.path.join(docs_dir, filename), 'r', encoding='utf-8') as f:
                    docs.append(f.read())
        if docs:
            return docs

    print("ERROR: No documents found!")
    print("Please create one of:")
    print("  - documents.json (list of strings)")
    print("  - documents.txt (one document per line)")
    print("  - documents/ folder with .txt files")
    return None

def reindex():
    """Main re-indexing function."""
    from src.hipporag import HippoRAG

    print("=" * 60)
    print("  HippoRAG Re-indexing Tool")
    print("=" * 60)

    # Step 1: Clear cache
    print("\n[Step 1] Clearing cached files...")
    clear_cache()

    # Step 2: Load documents
    print("\n[Step 2] Loading documents...")
    docs = load_original_documents()

    if not docs:
        return

    print(f"Found {len(docs)} documents")

    # Step 3: Initialize HippoRAG
    print("\n[Step 3] Initializing HippoRAG...")
    hipporag = HippoRAG(
        save_dir='outputs',
        llm_model_name='gemini/gemini-2.5-flash',
        embedding_model_name='gemini/gemini-embedding-001'
    )

    # Step 4: Re-index
    print("\n[Step 4] Re-indexing documents (this may take a while)...")
    hipporag.index(docs=docs)

    print("\n" + "=" * 60)
    print("  Re-indexing complete!")
    print("=" * 60)
    print(f"\nGraph now has:")
    print(f"  - {len(hipporag.graph.vs)} nodes")
    print(f"  - {len(hipporag.graph.es)} edges")

if __name__ == "__main__":
    reindex()
