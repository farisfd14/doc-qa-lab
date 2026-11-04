"""
Milestone 2 - TF-IDF retrieval, written from scratch.

Pipeline:  question -> tokens -> TF-IDF vector
           chunks   -> tokens -> TF-IDF vectors
           score every chunk by cosine similarity -> return top k
"""

import math
import os
import re
import sys
from collections import Counter

# Let this script find Loader.py (Milestone 1) in the same folder.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from loader import build_chunks


def tokensize(text: str) -> list:
    """Lowercase and split into word tokens: 'Hello, world!' -> ['hello', 'world']"""
    return re.findall(r"[a-z0-9]+", text.lower())


class TfidfRetriever:
    """Indexes chunks once, then answers queries with the top-k best chunks."""

    def __init__(self, chunks: list):
        self.chunks = chunks
        self.n = len(chunks)
        self.chunk_tokens = [tokensize(c["text"]) for c in chunks]

        # df[word] = in how many DIFFERENT chunks does this word appear?
        # (set() so repeated words in one chunk count only once)
        self.df = Counter()
        for tokens in self.chunk_tokens:
            for word in set(tokens):
                self.df[word] += 1

        # Pre-compute the TF-IDF vector of every chunk (expensive, do it once)
        self.chunk_vectors = [self._vectorize(t) for t in self.chunk_tokens]

    def _vectorize(self, tokens: list) -> dict:
        """Turn a token list into a TF-IDF vector: {word: tf * idf}."""
        tf = Counter(tokens)                    # how often each word appears HERE

        vec = {}
        for word, count in tf.items():
            if word not in self.df:
                continue                        # word never seen in the documents

            idf = math.log(self.n / self.df[word]) # rare word -> high idf
            vec[word] = count * idf
        return vec

    @staticmethod
    def cosine(a: dict, b: dict) -> float:
        """Cosine similarity between two sparse vectors stored as dicts."""
        dot = sum(w * b.get(word, 0.0) for word, w in a.items())
        norm_a = math.sqrt(sum(w * w for w in a.values()))
        norm_b = math.sqrt(sum(w * w for w in b.values()))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def search(self, query: str, k: int = 3) -> list:
        """Return the k chunks most similar to query."""
        q_vec = self._vectorize(query)
        scored = [
            (self.cosine(q_vec, c_vec), self.chunks[i])
            for i, c_vec in enumerate(self.chunk_vectors)
        ]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return scored[:k]


if __name__ == "__main__":
    retriever = TfidfRetriever(build_chunks())
    question = "What is Chimacum Creek?" # <- change this to test
    print(f"Q: {question}\n")
    for score, chunk in retriever.search(question):
        print(f"score={score:.3f}  source={chunk['source']}#{chunk['chunk_id']}")
        print(chunk["text"][:150].replace("\n", " "), "\n")

