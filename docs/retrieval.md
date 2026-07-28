# Retrieval

Weft ships with a multi-stage retrieval pipeline that combines embedding similarity and lexical ranking.

## Retrieval modes

- Semantic retrieval uses sentence-transformer embeddings and pgvector cosine distance.
- Lexical retrieval uses PostgreSQL full-text search.
- Hybrid retrieval combines both sources with either reciprocal rank fusion or linear fusion.

## Current behavior

The CLI exposes a search command that uses the hybrid retrieval pipeline by default. This is the recommended mode for first-time usage because it tends to produce more stable results across the bundled benchmark set.

## Limitations

The retrieval stack is intentionally simple and is not designed as a research-grade ranking system. It is suitable for local indexing, experimentation, and release-quality demos rather than production-scale search infrastructure.
