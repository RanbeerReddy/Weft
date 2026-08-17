# Memory Engine

The memory engine remains an experimental component in the current release.

## Current status

The repository contains memory-related models and extraction utilities (`extract_memories.py`), but the feature is not yet a polished public-facing workflow. 

Currently, memory extraction relies on deterministic rule-based (regex) matching and JSON string formatting, rather than LLM-based extraction. The system uses basic heuristics to identify potential memories (e.g., goals, experiences) and formats them into JSON strings for embedding. The `mock_llm_extract` function is present as a stub for future integration. 

The current release focuses on core ingestion, indexing, and retrieval.

## Recommendation

Treat the memory engine as an experimental extension for future releases rather than a highlighted v1.0 feature.
