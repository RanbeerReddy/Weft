# Weft Evaluation & Audit Suite

This directory contains the tools, scripts, and data used to evaluate and audit the retrieval quality of the Weft system. The architecture is modularized to support targeted testing (like chunking vs. embeddings) and automated A/B benchmarking.

## 📂 Directory Structure

### `run_all_phases.py`
The master execution script. Run this to sequentially trigger all audit phases and compile a unified `final_report.md` summarizing the overall health of the retrieval pipeline.

### `/phases`
Contains the sequential scripts that make up the formal pipeline audit:
- `ingestion_audit.py`: Verifies that data was successfully parsed and inserted into the DB.
- `corpus_validator.py`: Checks if expected benchmark answers actually exist in the database.
- `failure_analysis.py`: Classifies retrieval failures into distinct categories (Data Missing, Ranking Bug, etc.).
- `embedding_analysis.py`: Evaluates vector distance distributions to diagnose embedding weaknesses.
- `chunk_analysis.py`: Analyzes the impact of the current text splitting strategy on context preservation.
- `search_experiments.py`: Runs baseline benchmarks (Top-K hit rates, MRR) against the vector database.
- `reranking_dataset.py`: Prepares a candidate pool of top-100 results for offline reranker evaluation.

### `/core`
Shared utilities used across various evaluation and benchmark scripts:
- `metrics.py` / `memory_metrics.py`: Calculators for standard search metrics (Hit@K, MRR).
- `retrieval_eval.py` / `memory_retrieval_eval.py`: Wrappers and test suites for scoring baseline retrievers.

### `/benchmarks`
Standalone scripts used for deep-dive investigations and experimental architecture testing:
- `benchmark_reranker.py`: Live A/B testing of the Cross-Encoder pipeline against the Vector baseline.
- `benchmark_reranker_offline.py`: Fast offline execution of reranker metrics using pre-fetched datasets.
- `reranker_audit_script.py`: Targeted tests (ablation, chunk coherence) to debug cross-encoder behavior.

### `/data`
Static inputs and intermediate data structures:
- `test_queries.json` / `memory_queries.json`: The ground-truth questions and expected answers used for testing.
- `reranking_candidates.json` / `reranking_dataset.json`: Pre-retrieved Top-100 candidate pools used for offline testing.

### `/reports`
The final output destination. All generated JSON statistics, markdown summaries, code reviews, and the unified `final_report.md` are saved here after running the evaluation scripts.
