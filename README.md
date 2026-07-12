# E-Mualim: Corrective RAG for Hallucination Mitigation in Educational Q&A

This repository contains the source code, evaluation pipeline, and test data for the paper:

> **Corrective Retrieval-Augmented Generation for Hallucination Mitigation in Educational Q&A: A Case Study on E-Mualim**

E-Mualim is a speech-enabled AI tutoring system that implements a two-stage corrective RAG pipeline to reduce hallucination in educational Q&A responses.

## Architecture

The system is implemented as a LangGraph pipeline with **7 nodes** across 4 processing layers:

| Layer | Nodes | Models |
|-------|-------|--------|
| **Input** | Groq Whisper STT | — |
| **Retrieve** | 1. Intent classifier, 2. RAG retriever, 3. CRAG grader | `llama-3.1-8b-instant` (classifier + grader) |
| **Generate** | 4. Generator, 5. Hallucination checker | `llama-3.3-70b-versatile` (generator), `llama-3.1-8b-instant` (checker) |
| **Output** | 6. Response expander, 7. Response router | `llama-3.1-8b-instant` (expander) |

### Two-stage corrective pipeline

- **Stage 1 — CRAG Grader**: Scores all retrieved chunks in a single batched LLM call (0–10 scale). Chunks scoring ≥ 6 pass through; others are discarded. If all chunks are discarded, the generator falls back to parametric knowledge.
- **Stage 2 — Hallucination Checker**: Verifies that the generated response does not contradict or go beyond the source material. Triggers up to 2 retries on failure.

## Repository Structure

```
emualim-crag/
├── rag/                          # Core pipeline code
│   ├── graph.py                  # LangGraph pipeline (7 nodes)
│   ├── retriever.py              # MongoDB Atlas vector search
│   ├── embedder.py               # Cohere embed-english-v3.0
│   ├── session_state.py          # Redis/in-memory engagement store
│   └── tutor_llm.py              # LiveKit LLM wrapper
│
├── eval/                         # Evaluation pipeline
│   ├── run_eval.py               # Run pipeline in 4 configurations
│   ├── score_ragas.py            # Compute RAGAS metrics
│   └── analyze_results.py        # Per-category analysis + qualitative examples
│
├── data/
│   ├── knowledge_base/           # Source PDFs (chunked and indexed in MongoDB)
│   │   └── intro_python_ch1-4.pdf
│   └── test_sets/
│       └── test_set_v1.csv       # 60 curated Q&A pairs across 5 categories
│
├── paper/                        # LaTeX source for the paper
│   ├── emualim_ieee.tex
│   └── references_ieee.bib
│
├── figures/                      # Pipeline and CRAG grader diagrams
├── .env.example                  # Required environment variables
├── requirements.txt
└── README.md
```

## Setup

### Prerequisites

- Python 3.10+
- A [Groq](https://console.groq.com) API key (free tier works)
- A [Cohere](https://dashboard.cohere.com) API key (trial works)
- A [MongoDB Atlas](https://www.mongodb.com/atlas) cluster with Vector Search index

### Installation

```bash
git clone https://github.com/Maryam-qureshi/emualim-crag.git
cd emualim-crag
python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your API keys
```

### MongoDB Atlas Setup

1. Create a free cluster on [MongoDB Atlas](https://www.mongodb.com/atlas).
2. Create a database `emualim` with a collection `chunks`.
3. Ingest your knowledge base documents (chunked with metadata).
4. Create a Vector Search index named `vector_index` with `numDimensions: 1024`.

## Reproducing the Evaluation

### Step 1: Run the pipeline on the test set

```bash
# Run all 4 configurations (takes ~30 min on Groq free tier)
python -m eval.run_eval

# Or run a single configuration
python -m eval.run_eval --config naive_rag
python -m eval.run_eval --config full_emualim

# Adjust delay between API calls if hitting rate limits
python -m eval.run_eval --delay 2.0
```

This produces JSON files in `eval/results/`:
- `naive_rag.json`
- `crag_only.json`
- `hall_only.json`
- `full_emualim.json`

### Step 2: Compute RAGAS scores

```bash
# Score all configurations (uses Groq as evaluator by default)
python -m eval.score_ragas

# Or use OpenAI as evaluator
python -m eval.score_ragas --evaluator openai
```

Output: a comparison table with Faithfulness, Answer Relevancy, Context Precision, Context Recall, and Retry Rate for each configuration.

### Step 3: Analyze results

```bash
python -m eval.analyze_results
```

Produces:
- Per-category breakdown (concept, howto, mistake, complex, out_of_scope)
- Intent classification accuracy report
- Qualitative comparison examples for the paper (Table III)

## Test Set

The test set (`data/test_sets/test_set_v1.csv`) contains 60 curated Q&A pairs distributed across 5 categories:

| Category | Count | Description |
|----------|-------|-------------|
| `concept` | ~25% | "What is X" questions — single-chunk retrieval |
| `howto` | ~25% | "How do I" questions — code-example retrieval |
| `mistake` | ~20% | Error-reporting — narrow `common_mistake` retrieval |
| `complex` | ~15% | Multi-chunk questions — tests context assembly |
| `out_of_scope` | ~15% | Topics beyond ch.1–4 — triggers `needs_fallback` |

Ground-truth answers were written directly from knowledge base content by the authors.

## Evaluation Configurations

| Configuration | CRAG Grader | Hallucination Checker | Description |
|---------------|:-----------:|:---------------------:|-------------|
| `naive_rag` | ✗ | ✗ | Baseline — all chunks pass to generator |
| `crag_only` | ✓ | ✗ | Isolates Stage 1 contribution |
| `hall_only` | ✗ | ✓ | Isolates Stage 2 contribution |
| `full_emualim` | ✓ | ✓ | Production configuration |

## Citation

If you use this code or dataset in your research, please cite:

```bibtex
@inproceedings{emualim2026,
  title     = {Corrective Retrieval-Augmented Generation for Hallucination
               Mitigation in Educational Q\&A: A Case Study on E-Mualim},
  author    = {Author One and Author Two and Author Three},
  booktitle = {TODO: Conference Name},
  year      = {2026},
}
```

## License

This project is released for academic and research purposes.
