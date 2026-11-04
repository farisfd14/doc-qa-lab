"""
Milestone 3b - retrieve chunks, then have a local LLM write the answer.

The flow (this is the whole RAG pattern in one file):
  question -> retriever finds top-3 chunks -> we build a prompt
  containing ONLY those chunks -> Ollama writes the answer.
"""

import os
import sys
import time

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from loader import build_chunks             # Milestone 1
from embedder import EmbeddingRetriever     # Milestone 3a

OLLAMA_URL = "http://localhost:11434/api/generate"
LLM_MODEL = "llama3.2:1b"


def build_prompt(question: str, chunks: list) -> str:
    """Stuff the retrieved chunks into a prompt with strict instructions."""
    context = "\n\n".join(
        f"[{c['source']}#{c['chunk_id']}] {c['text']}" for c in chunks
    )
    return (
        "Answer the question using ONLY the context below. "
        'If the context does not contain the answer, say "I don\'t know".\n\n'
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


def ask_ollama(prompt: str) -> tuple:
    """Send the prompt to the local Ollama server. Returns (answer, seconds)."""
    start = time.perf_counter()
    resp = requests.post(
        OLLAMA_URL,
        json={"model": LLM_MODEL, "prompt": prompt, "stream": False},
        timeout=300,
    )
    elapsed = time.perf_counter() - start
    return resp.json()["response"].strip(), elapsed


def main():
    print("Loading retriever...")
    retriever = EmbeddingRetriever(build_chunks())
    print("Ready! Ask questions about your documents. Empty line to quit.\n")

    while True:
        question = input("Ask: ").strip()
        if not question:
            break

        chunks = [c for _, c in retriever.search(question, k=3)]
        answer, seconds = ask_ollama(build_prompt(question, chunks))

        print(f"\nAnswer ({seconds:.1f}s): {answer}")
        print("Sources:", ", ".join(f"{c['source']}#{c['chunk_id']}" for c in chunks), "\n")


if __name__ == "__main__":
    main()