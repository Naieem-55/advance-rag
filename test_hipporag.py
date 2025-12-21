from hipporag import HippoRAG
import os
import glob

# Path to your documents folder
DOCUMENTS_FOLDER = "documents"

def load_documents_from_folder(folder_path):
    """Load all TXT files from a folder."""
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' not found!")
        return None

    all_docs = []
    txt_files = glob.glob(os.path.join(folder_path, "*.txt"))

    if not txt_files:
        print(f"No .txt files found in '{folder_path}'")
        return None

    for file_path in txt_files:
        print(f"Reading: {os.path.basename(file_path)}")
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Split by page markers if present
        if "=== Page" in content:
            pages = content.split("=== Page")
            for page in pages:
                # Clean up page content
                lines = page.strip().split('\n')
                # Skip the page number line (e.g., "1 ===")
                if lines and "===" in lines[0]:
                    lines = lines[1:]
                text = '\n'.join(lines).strip()
                if text and len(text) > 50:  # Skip very short chunks
                    all_docs.append(text)
        else:
            # Split by double newline (paragraphs)
            paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
            for para in paragraphs:
                if len(para) > 50:  # Skip very short paragraphs
                    all_docs.append(para)

    return all_docs

def main():
    # Load documents
    print(f"Loading documents from '{DOCUMENTS_FOLDER}/' folder...")
    docs = load_documents_from_folder(DOCUMENTS_FOLDER)

    if docs is None or len(docs) == 0:
        print("No documents loaded!")
        return

    print(f"\nFound {len(docs)} document chunks.")
    print("\nSample documents:")
    for i, doc in enumerate(docs[:3]):
        preview = doc[:100].replace('\n', ' ')
        print(f"  {i+1}. {preview}...")
    print()

    # Initialize HippoRAG
    print("Initializing HippoRAG with GPT-4o...")
    hipporag = HippoRAG(
        save_dir='outputs',
        llm_model_name='gpt-4o',
        embedding_model_name='text-embedding-3-large'
    )

    # Index documents
    print("Indexing documents (this may take a while)...")
    hipporag.index(docs=docs)
    print("Indexing complete!\n")

    # Interactive query loop
    print("="*60)
    print("  HippoRAG Ready!")
    print("  Ask questions about 'অপরিচিতা' in Bangla or English.")
    print("  Type 'exit' or 'quit' to stop.")
    print("="*60)

    while True:
        print()
        query = input("আপনার প্রশ্ন / Your question: ").strip()

        if query.lower() in ['exit', 'quit', 'q', 'বাহির', 'প্রস্থান']:
            print("বিদায়! / Goodbye!")
            break

        if not query:
            print("Please enter a question.")
            continue

        # Get answer (instruct to respond in same language as query)
        print("চিন্তা করছি... / Thinking...")
        query_with_instruction = f"{query}\n\n(Please respond in the same language as the question.)"
        results = hipporag.rag_qa(queries=[query_with_instruction])

        answer = results[0][0].answer
        retrieved_docs = results[0][0].docs[:3]  # Top 3 sources
        doc_scores = results[0][0].doc_scores[:3] if results[0][0].doc_scores is not None else None

        print("\n" + "="*60)
        print("উত্তর / Answer:")
        print("-"*60)
        print(answer)
        print("\n" + "="*60)
        print("📚 তথ্যসূত্র / References:")
        print("-"*60)
        for i, doc in enumerate(retrieved_docs):
            score = f" (Score: {doc_scores[i]:.4f})" if doc_scores is not None else ""
            print(f"\n[{i+1}]{score}")
            print("-"*40)
            # Show full document content (first 500 chars)
            preview = doc[:500].replace('\n', '\n   ')
            print(f"   {preview}")
            if len(doc) > 500:
                print(f"   ... (truncated)")
        print("="*60)

if __name__ == "__main__":
    main()
