"""
Milestone 3a - retrieval with neural embeddings.

Same interface as TfidfRetriever (search(query, k)), but instead of
counting words, we map text -> a 384-dimensional vector that encodes
MEANING. Chunks with similar meaning get similar vectors.
"""

import os
import sys

import numpy as np
from sentence_transformers import SentenceTransformer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from loader import build_chunks     # Milestone 1


class EmbeddingRetriever:
    """Embed all chunks once, then find the closest chunks to any query."""

    def __init__(self, chunks: list, model_name: str = "all-MiniLM-L6-v2"):
        self.chunks = chunks
        print(f"Loading embedding model '{model_name}' ...")
        self.model = SentenceTransformer(model_name)

        texts = [c["text"] for c in chunks]
        # normalize_embeddings=True makes every vector Length 1,
        # so a plain dot product IS cosine similarity.
        self.chunk_vectors = self.model.encode(texts, normalize_embeddings=True)

    def search(self, query: str, k: int = 3) -> list:
        q_vec = self.model.encode([query], normalize_embeddings=True)[0]
        scores = self.chunk_vectors @ q_vec     # one dot product per chunk
        top = np.argsort(scores)[::-1][:k]      # indices of the k highest scores

        return [(float(scores[i]), self.chunks[i]) for i in top]


if __name__ == "__main__":
    retriever = EmbeddingRetriever(build_chunks())
    question = "Where does the water flow?"    # <- try your synonym test here too
    print(f"\nQ: {question}\n")
    for score, chunk in retriever.search(question):
        print(f"score={score:.3f}  source={chunk['source']}#{chunk['chunk_id']}")
        print(chunk["text"][:150].replace("\n", " "), "\n")
