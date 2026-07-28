# Database

Weft uses PostgreSQL with the pgvector extension for embedding storage and similarity search.

## Schema overview

- conversations: conversation metadata and title.
- messages: individual message turns linked to a conversation.
- chunks: text chunks derived from message content.
- embeddings: vector representations stored in pgvector.

## Local setup

1. Start PostgreSQL with Docker:
   - docker compose up -d postgres
2. Copy [.env.example](../.env.example) to .env and confirm the database URL.
3. Run the Alembic migrations:
   - alembic upgrade head

## Notes

The database layer is intentionally simple and should be treated as a local-first persistence substrate for indexing and retrieval.
