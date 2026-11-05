"""
Milestone 3b - retrieve chunks, then have a local LLM write the answer.

The flow (this is the whole RAG pattern in one file):
  question -> retriever finds top-3 chunks -> we build a prompt
  containing ONLY those chunks -> Ollama writes the answer.

Hard-won lessons baked into this file (same bugs we fixed in the
voice-assistant project, now fixed here too):
  1. trust_env=False -> bypasses the machine proxy so localhost
     traffic is not routed through 127.0.0.1:3213
  2. pick_answer()   -> we never hardcode a JSON field name like
     "answer" or "result", because Ollama renames fields between
     versions and hardcoding caused a KeyError crash.
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

# One session reused for every request.
# trust_env=False tells requests: "ignore the system proxy settings".
# Without this, requests tried to route even Localhost traffic
# through the machine's proxy (127.0.0.1:3213) and everyhing broke.
_session = requests.Session()
_session.trust_env = False


def pick_answer(data: dict) -> str:
    """The answer is the longest string in Ollama's JSON response.

    We never hardcode a field name (like "answer" or "result")
    because Ollama renames fields between versions.

    Assumption: the answer is a full sentence, so it is always
    longer than metadata strings like the model name.
    """
    strings = [v for v in data.values() if isinstance(v, str)]
    return max(strings, key=len).strip()

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
    return pick_answer(resp.json()), elapsed


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