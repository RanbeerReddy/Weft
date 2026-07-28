# Development

## Setup

1. Create a Python 3.11 environment.
2. Install dependencies with pip or Poetry.
3. Copy [.env.example](../.env.example) to .env.
4. Start PostgreSQL and run Alembic migrations.

## Testing

Run the test suite with:

- pytest tests/

## Linting

The repository is expected to pass:

- black --check Weft tests
- isort --check-only Weft tests
- ruff check Weft tests
- mypy Weft --no-incremental
