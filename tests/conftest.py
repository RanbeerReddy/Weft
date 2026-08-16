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
from Weft.config.settings import settings

TEST_DATABASE_URL = os.getenv(
    "TEST_DATABASE_URL",
    settings.DATABASE_URL,
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


@pytest.fixture(scope="function", autouse=True)
def db_session(engine, monkeypatch):
    """
    Returns an sqlalchemy session, and after the test tears down,
    rolls back the transaction to ensure an isolated test environment.
    """
    connection = engine.connect()
    transaction = connection.begin()

    Session = scoped_session(sessionmaker(bind=connection))
    session = Session()
    
    # Patch SessionLocal globally so all code uses this transactional session
    monkeypatch.setattr("Weft.storage.database.SessionLocal", lambda: session)

    yield session

    session.close()
    transaction.rollback()
    connection.close()


@pytest.fixture(autouse=True)
def mock_sentence_transformer(request, monkeypatch):
    """
    Automatically mock SentenceTransformer to prevent downloading large models
    during tests, speeding up CI.
    Skips mocking for tests marked with @pytest.mark.integration.
    """
    if "integration" in request.node.keywords:
        return

    class MockModel:
        def __init__(self, *args, **kwargs):
            pass

        def encode(self, sentences, *args, **kwargs):
            import numpy as np
            dim = 384
            if isinstance(sentences, str):
                return np.array([0.1] * dim)
            return [np.array([0.1] * dim) for _ in sentences]

    # Patch the SentenceTransformer globally in the sentence_transformers module
    monkeypatch.setattr("sentence_transformers.SentenceTransformer", MockModel)
    
    # Also patch direct imports
    import Weft.core.context_assembler
    if hasattr(Weft.core.context_assembler, "SentenceTransformer"):
        monkeypatch.setattr("Weft.core.context_assembler.SentenceTransformer", MockModel)
        
    import Weft.storage.create_embedding
    if hasattr(Weft.storage.create_embedding, "SentenceTransformer"):
        monkeypatch.setattr("Weft.storage.create_embedding.SentenceTransformer", MockModel)
    
    # Reset any cached model so the mock is picked up
    if hasattr(Weft.storage.create_embedding, "_model"):
        monkeypatch.setattr("Weft.storage.create_embedding._model", None)

