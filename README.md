# Weft

Weft is an open-source Python project for turning exported ChatGPT conversations into a searchable, local-first knowledge base.

## What is Weft?

Weft helps you preserve and reuse the knowledge contained in long-running AI conversations. Instead of letting chats disappear into a history tab, Weft reconstructs conversations, stores them locally, and makes them searchable through a lightweight retrieval pipeline.

## Features

- Extract and parse ChatGPT export archives.
- Reconstruct conversation trees into markdown-friendly vault output.
- Store conversations, messages, chunks, and embeddings in PostgreSQL with pgvector.
- Search the indexed corpus with semantic, lexical, or hybrid retrieval.
- Run built-in benchmark and evaluation workflows.

## Architecture overview

Weft follows a simple pipeline:

1. Ingest a ChatGPT export archive.
2. Reconstruct conversations into local markdown output.
3. Import conversation data into PostgreSQL.
4. Split message text into chunks and generate embeddings.
5. Search, benchmark, and evaluate the resulting index.

A fuller overview is available in [docs/architecture.md](docs/architecture.md).

## Installation

### Requirements

- Python 3.11
- PostgreSQL with the pgvector extension
- Docker (recommended for local development)

### Quick start

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

### Start PostgreSQL

```bash
docker compose up -d postgres
```

### Initialize the database

```bash
alembic upgrade head
```

## CLI usage

Weft now exposes a Typer-based CLI:

```bash
weft init
weft ingest Data/Raw\ Data/reddyranbeer\ openAI\ Data.zip
weft embed
weft search "what learning strategies do I use"
weft benchmark
weft evaluate
weft stats
```

### Search example

```bash
weft search "tell me about the Weft project"
```

### Benchmark example

```bash
weft benchmark
```

## Project structure

- [Weft/core](Weft/core) — ingestion and reconstruction logic.
- [Weft/storage](Weft/storage) — database models, chunking, embeddings, and search helpers.
- [Weft/evaluation](Weft/evaluation) — benchmark and evaluation workflows.
- [docs](docs) — user and contributor documentation.
- [tests](tests) — unit and integration coverage.

## Roadmap

- Improve retrieval quality and benchmark coverage.
- Harden ingestion against more export variants.
- Continue polishing the local-first memory workflow.

## Limitations

Weft is a strong first release for a local knowledge workflow, but it is not intended to be a production-grade search engine or a complete agent platform. The current release emphasizes simplicity, transparency, and reproducibility.

## Documentation

- [docs/architecture.md](docs/architecture.md)
- [docs/database.md](docs/database.md)
- [docs/retrieval.md](docs/retrieval.md)
- [docs/benchmarks.md](docs/benchmarks.md)
- [docs/memory_engine.md](docs/memory_engine.md)
- [docs/development.md](docs/development.md)

Inspired by:

* second brain systems
* knowledge graphs
* semantic search
* AI memory architectures
* personal knowledge management
* research workflows
* long-term thinking systems

---

# Contributing

Still heavily experimental.

But if you care about:

* AI memory
* knowledge systems
* semantic retrieval
* local-first AI
* personal context engines
* Obsidian workflows

feel free to explore, fork, or contribute.

---

# Final Thought

Most AI conversations disappear.

Weft exists because maybe they shouldn’t.

```
```
