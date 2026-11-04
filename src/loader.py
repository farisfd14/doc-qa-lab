"""
Milestone 1 - load documents and split them into chunks.

Why chunk at all?
    LLMs have a limited context window, and retrieval works best on
    short, focused passages. So we cut documents into overlapping
    pieces of ~500 characters. The overlap means a sentence that
    straddles a boundary still appears intact in at least one chunk.
"""

from pathlib import Path


def load_documents(docs_dir: str = "docs") -> list:
    """Read every .txt/.md file into a list of (filename, text) pairs."""
    docs = []
    for path in sorted(Path(docs_dir).iterdir()):
        if path.suffix.lower() in {".txt", ".md"}:
            text = path.read_text(encoding="utf-8")
            docs.append((path.name, text))
    if not docs:
        raise SystemExit(f"No .txt or .md files found in {docs_dir}/")
    return docs


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list:
    """
    Split text into overlapping character windows.

    Example with chunk_size=500, overlap=100:
        chunk 0 = characters   0..500
        chunk 1 = characters 400..900
        chunk 2 = characters 800..1300
        ...
    """
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += chunk_size - overlap # step forward, keeping the overlap
    return chunks


def build_chunks(docs_dir: str = "docs") -> list:
    """Load all documents and return a flat list of chunk dictionaries."""
    all_chunks = []
    for filename, text in load_documents(docs_dir):
        for i, chunk in enumerate(chunk_text(text)):
            all_chunks.append({
                "source": filename,  # which fie this came from
                "chunk_id": i,      # which piece of that file
                "text": chunk,
            })
    return all_chunks


if __name__ == "__main__":
    chunks = build_chunks()
    print(f"Loaded {len(chunks)} chunks.")
    print("--- first chunk ---")
    print(chunks[0]["text"][:200])