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
2. Install Ollama (ollama.com) `ollama pull llama3.2:1b`
3. Put notes in docs/ and questions in tests/qa_tests.csv

## Usage
    py src/loader.py
    py src/retriever.py
    py src.embedder.py
    py src/evaluate.py --engine tfidf
    py src/evaluate.py --engine embed
    py src/answer.py
    py -m pytest tests/ -v

## Results (TODO: your numbers, from results/eval_*.csv)
| Engine | Hit rate    | Avg search time |
| --- |-------------| --- |
| TF-IDF | 0/13 (TODO) | ? s |
| Embeddings | 0/13 (TODO) | ? s |

## What I learned
1. My first test set mixed up two things: answerable questions and unanswerable ones. Both engines
   scored 5/13, but on the 5 questions the document can answer, both were perfect; the other 7 were
   correctly rejected. Retrieval quality can't be measured with questions that have no answer in the corpus.
2. Embeddings are ~150x slower than TF-IDF and gave identical hit rates on this document - their
   advantage (synonyms) only shows when the test set has in-corpus synonym questions. Next step: add
   those.
3. A RAG system that returns nothing for out-of-scope questions is doing its job - the LLM then answers
   "I don't know" instead of hallucinating.
4. Retriever = search (librarian), LLM = writing (writer) - keeping them separate
makes each testable.

## Project structure
    src/loader.py, src/retriever.py, src/embedder.py, src/answer.py,
src/evaluate.py
    tests/qa_tests.csv, tests/test_core.py