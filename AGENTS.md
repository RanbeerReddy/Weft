# AGENTS.md

## What this project is
- Weft is an experimental Python system for turning ChatGPT export files into a structured local knowledge graph.
- The main pipeline ingests exported conversation JSON, reconstructs conversation trees, produces markdown/Obsidian-friendly output, and stores chunks/embeddings in PostgreSQL with pgvector.
- The repo is primarily a backend/data pipeline project today; there is no complete FastAPI service implementation in the source tree yet.

## Key directories
- `Weft/core/` — ingestion and conversation reconstruction scripts.
- `Weft/storage/` — SQLAlchemy models, DB setup, chunk creation, and embedding creation.
- `Weft/utils/` — helper utilities, zip extraction, and exception handling.
- `alembic/` — database migration configuration and revisions.
- `tests/` — integration-style tests, especially `tests/test_pgvector.py`.

## Important files
- `README.md` — project vision, goals, and high-level design.
- `plan.md` — architectural notes and feature direction.
- `docker-compose.yml` — local PostgreSQL with pgvector setup.
- `requirements.txt` — Python dependency list and package versions.

## Common development commands
- Install dependencies: `python -m pip install --upgrade pip && pip install -r requirements.txt`
- Start the local database: `docker compose up -d postgres`
- Run tests: `pytest tests/`

## Primary Python scripts
- `python Weft/core/Extract_data.py` — extract exported ZIP data into `Data/Extracted Data/`.
- `python Weft/core/reconstructing_chats.py` — merge ChatGPT export files and create vault markdown.
- `python Weft/storage/create_convo_msg.py` — populate conversation/message records.
- `python Weft/storage/create_chunks.py` — split message text into chunks.
- `python Weft/storage/create_embedding.py` — encode chunks into pgvector embeddings.

## Repo conventions
- The package is imported as `Weft.*` from the repository root.
- DB models use SQLAlchemy 2.0 ORM and `pgvector.Vector(384)` for embeddings.
- The default local DB URL is `postgresql+psycopg2://weft_user:weft_123@localhost:5432/weft_db`.
- Markdown outputs are written into `vault/conversations`.

## Key modules and their responsibilities
- `Weft/core/` — ingestion pipeline (ZIP extraction, conversation reconstruction, markdown export).
- `Weft/storage/` — database models, chunking, and embedding generation.
- `Weft/evaluation/` — retrieval evaluation framework with metrics and benchmarking.

## What agents should focus on first
- Stabilize the ingestion and conversation reconstruction pipeline.
- Keep the local DB/pgvector schema in sync with `Weft/storage/models.py` and `alembic/`.
- Use the evaluation framework (`Weft/evaluation/`) to measure retrieval quality before and after improvements.
- Prefer using referenced docs instead of copying large design content from `README.md` or `plan.md`.

## What to avoid
- Do not assume there is a production-ready web API; no FastAPI app module is present.
- Do not create React/TypeScript frontend work unless the repo later adds frontend sources.
- Avoid changing database credentials or production deployment settings without explicit instruction.
