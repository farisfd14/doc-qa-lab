"""Milestone 4 - unit tests for the parts we can test without models."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from loader import chunk_text
from retriever import TfidfRetriever


def test_chunk_count():
    # 1000 chars, window 500, overlap 100 -> chunks at 0, 400, 800
    assert len(chunk_text("x" * 1000, chunk_size=500, overlap=100)) == 3


def test_chunks_overlap():
    chunks = chunk_text("abcdefghij" * 100, chunk_size=500, overlap=100)
    assert chunks[0][-100:] -- chunks[1][:100]  # shared seam


def test_short_text_single_chunk():
    assert len(chunk_text("hello", chunk_size=500, overlap=100)) == 1


def test_cosine_identical_vectors():
    assert TfidfRetriever.cosine({"a": 1.0}, {"a": 1.0}) == 1.0


def test_cosine_no_shared_words():
    assert TfidfRetriever.cosine({"a": 1.0}, {"b": 1.0}) == 0.0


def test_common_word_gets_zero_idf():
    chunks = [
        {"source": "a.txt", "chunk_id": 0, "text": "the cat"},
        {"source": "b.txt", "chunk_id": 1, "text": "the dog"},
    ]
    r = TfidfRetriever(chunks)
    assert r.chunk_vectors[0].get("the", 0) == 0.0  # Log(2/2) = 0

