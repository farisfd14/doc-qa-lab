"""
Milestone 4 - the Lab: measure retrieval quality and speed.

For each test question:
  - run the retriever, keep the top 3 chunks
  - a HIT means the expected keyword appears in one of them
  - measure how long the search took

Usage:
    py src/evaluate.py                  # embedding retriever (default)
    py src/evaluate.py --engine tfidf   # keyword retriever (Milestone 2)
"""

import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from loader import build_chunks


def get_retriever(engine: str, chunks: list):
    if engine == "tfidf":
        from retriever import TfidfRetriever
        return TfidfRetriever(chunks)
    from embedder import EmbeddingRetriever
    return EmbeddingRetriever(chunks)


def evaluate(engine: str, csv_path: str = "tests/qa_tests.csv", k: int = 3):
    retriever = get_retriever(engine, build_chunks())

    with open(csv_path, newline="", encoding="utf-8") as f:
        tests = list(csv.DictReader(f))

    hits = 0
    rows = []
    for t in tests:
        question = t["question"]
        keyword = t["expected_keyword"].lower()

        start = time.perf_counter()
        results = retriever.search(question, k=k)
        elapsed = time.perf_counter() - start

        hit = any(keyword in c["text"].lower() for _, c in results)
        hits += hit
        rows.append({
            "engine": engine,
            "question": question,
            "hit": int(hit),
            "search_time_s": round(elapsed, 4),
            "top_source": results[0][1]["source"],
        })
        print(f"{'HIT ' if hit else 'MISS'}  {elapsed:6.3f}s  {question}")

    hit_rate = hits / len(tests)
    print(f"\nEngine: {engine}  hit rate: {hits}/{len(tests)} = {hit_rate:.0%}")

    os.makedirs("results", exist_ok=True)
    out = f"results/eval_{engine}.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved - > {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", default="embed", choices=["embed", "tfidf"])
    args = parser.parse_args()
    evaluate(args.engine)
