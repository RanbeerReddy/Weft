import pytest
from sentence_transformers import SentenceTransformer
from sqlalchemy import select

from Weft.storage.models import Embedding


@pytest.mark.integration
def test_embedding_vector_distance(db_session):
    """
    Test pgvector cosine distance integration.
    This test verifies that the database successfully executes cosine_distance
    queries using the pgvector extension.
    """
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    query = "test query"
    vector_embedding = model.encode(query)

    # Note: If the test DB is empty, this query will just return an empty list,
    # which is fine. We are testing that the query executes without SQL errors.
    distance_attr = Embedding.embedding_vector.cosine_distance(vector_embedding)

    stmt = (
        select(
            Embedding.conversation_id,
            Embedding.message_id,
            Embedding.chunk_order,
            distance_attr.label("distance"),
        )
        .order_by("distance")
        .limit(1)
    )

    result = db_session.execute(stmt).fetchall()
    # It should not raise an exception
    assert isinstance(result, list)
