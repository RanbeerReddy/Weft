# Architecture

Weft is a local-first pipeline that turns ChatGPT export archives into a searchable knowledge base.

## High-level flow

1. Import a ChatGPT export archive.
2. Extract and merge the exported JSON files.
3. Reconstruct conversation trees into markdown-friendly output under the vault directory.
4. Load conversations and messages into the relational schema.
5. Split messages into chunks and generate embeddings.
6. Search and benchmark the indexed corpus with the retrieval pipeline.

## Main components

- Core ingestion: [Weft/core/Extract_data.py](../Weft/core/Extract_data.py) and [Weft/core/reconstructing_chats.py](../Weft/core/reconstructing_chats.py)
- Storage layer: [Weft/storage/models.py](../Weft/storage/models.py) and [Weft/storage/database.py](../Weft/storage/database.py)
- Retrieval: [Weft/core/retrieval.py](../Weft/core/retrieval.py)
- Evaluation: [Weft/evaluation](../Weft/evaluation)

## Data model

- Conversations hold high-level metadata.
- Messages capture individual turns in a conversation tree.
- Chunks store text segments derived from messages.
- Embeddings represent chunk vectors in pgvector.

## Operational notes

The current release focuses on a robust local workflow and clear CLI entry points. The system intentionally avoids any external SaaS dependencies for indexing and retrieval.
