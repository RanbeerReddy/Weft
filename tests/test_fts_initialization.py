import uuid

import pytest
from sqlalchemy import text

from Weft.storage.database import SessionLocal
from Weft.storage.models import Chunk, Conversation, Message


@pytest.mark.integration
def test_fts_initialization():
    db = SessionLocal()
    convo_id = str(uuid.uuid4())
    msg_id = str(uuid.uuid4())

    # Clean up first if needed, though db should be transactional if handled properly
    # Using explicit cleanup for idempotency in test

    try:
        # Create convo
        convo = Conversation(
            id=convo_id,
            title="FTS Test",
            current_node=msg_id,
        )
        db.merge(convo)

        # Create msg
        msg = Message(
            id=msg_id,
            conversation_id=convo_id,
            role="user",
            content="Testing full text search functionality.",
        )
        db.merge(msg)
        db.commit()

        # Insert chunk directly to trigger FTS
        chunk = Chunk(
            conversation_id=convo_id,
            message_id=msg_id,
            chunk_order=0,
            chunk_text="Testing full text search functionality.",
        )
        db.add(chunk)
        db.commit()

        # Retrieve chunk to check tsvector
        inserted_chunk = db.get(Chunk, chunk.id)
        assert inserted_chunk is not None

        # Querying chunk_tsvector via text() to verify Postgres handled it
        result = db.execute(
            text("SELECT chunk_tsvector FROM chunks WHERE id = :chunk_id"),
            {"chunk_id": chunk.id},
        ).scalar()

        # 'test' and 'functionality' should be lexemes in the tsvector
        assert result is not None, "chunk_tsvector is None, trigger did not run!"
        assert "test" in result or "'test'" in result

    finally:
        db.execute(
            text("DELETE FROM chunks WHERE conversation_id = :cid"), {"cid": convo_id}
        )
        db.execute(
            text("DELETE FROM messages WHERE conversation_id = :cid"), {"cid": convo_id}
        )
        db.execute(text("DELETE FROM conversations WHERE id = :cid"), {"cid": convo_id})
        db.commit()
        db.close()
