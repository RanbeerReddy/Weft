import os
import sys

import pytest
import sqlalchemy
from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker

# Add Weft to PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# Use the existing local pgvector instance for integration testing
# In CI, this will connect to the service container
TEST_DATABASE_URL = os.getenv(
    "TEST_DATABASE_URL",
    "postgresql+psycopg2://weft_user:weft_123@localhost:5432/weft_db",
)


@pytest.fixture(scope="session")
def engine():
    engine = create_engine(TEST_DATABASE_URL)
    # Ensure vector extension exists
    with engine.connect() as conn:
        conn.execute(sqlalchemy.text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()

    # We do NOT drop all tables because Alembic manages the schema.
    # We assume the schema is up to date via Alembic.
    yield engine
    engine.dispose()


@pytest.fixture(scope="function")
def db_session(engine):
    """
    Returns an sqlalchemy session, and after the test tears down,
    rolls back the transaction to ensure an isolated test environment.
    """
    connection = engine.connect()
    transaction = connection.begin()

    Session = scoped_session(sessionmaker(bind=connection))
    session = Session()

    yield session

    session.close()
    transaction.rollback()
    connection.close()


@pytest.fixture(autouse=True)
def mock_sentence_transformer(monkeypatch):
    """
    Automatically mock SentenceTransformer to prevent downloading large models
    during tests, speeding up CI.
    """

    class MockModel:
        def __init__(self, *args, **kwargs):
            pass

        def encode(self, sentences, *args, **kwargs):
            # Return a deterministic mock vector of dimension 384 (as expected by pgvector)
            # If multiple sentences, return a list of vectors
            dim = 384
            if isinstance(sentences, str):
                return [0.1] * dim
            return [[0.1] * dim for _ in sentences]

    # Patch the SentenceTransformer globally in the sentence_transformers module
    monkeypatch.setattr("sentence_transformers.SentenceTransformer", MockModel)
