# Doc Q&A Lab

A local "ask your documents" tool built with RAG (Retrieval-Augmented Generation) -
and a benchmark that measures how well the retrieval actually works.

## What it does
- Load a folder of .txt/.md notes
- Split them into overlapping chunks (loader.py)
- Retrieve relevant chunks - TF-IDF from scratch (retriever.py) OR neural
embeddings (embedder.py)
- Generate grounded answers with a local LLM via Ollama (answer.py)
- Measure retrieval quality: hit rate on a hand-written test set (evaluate.py)

# How it works (the RAG idea)
question -> retriever (librarian) -> top-3 chunks -> prompt -> LLM (writer) ->
answer + sources

## Setup
1. py -m pip install -r requirements.txt
2. Install Ollama (ollama.com) 'ollama pull llama3.2:1b'
3. Put notes in docs/ and questions in tests/qa_tests.csv

## Usage
    py src/loader.py
    py src/retriever.py
    py src.embedded.py
    py src/evaluate.py --engine tfidf
    py src/evaluate.py --engine embed
    py src/answer.py
    py -m pytest tests/ -v

## Results (TODO: your numbers, from results/eval_*.csv)
| Engine | Hit rate | Avg search time |
| --- | --- | --- |
| TF-IDF | 0/? (TODO) | ? s |
| Embeddings | 0/? (TODO) | ? s |

## What I learned
1. TODO: TF-IDF matches exact words; embeddings match meaning. My synonym
questions: TF-IDF hit ?/8, embeddings hit ?/8.
2. TODO: what questions(s) both engines missed and why.
3. Normalizing vectors makes dot product equal to cosine - free speed.
4. Retriever = search (librarian), LLM = writing (writer) - keeping them separate
makes each testable.

## Project structure
    src/loader.py, ssrc/retriever.py, src/embedder.py, src/answer.py,
src/evaluate.py
    tests/qa_tests.csv, tests/test_core.py