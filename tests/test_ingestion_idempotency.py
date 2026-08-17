import json
import os
import tempfile
import uuid

from sqlalchemy import select

from Weft.storage.create_chunks import build_chunks
from Weft.storage.create_convo_msg import parse_export
from Weft.storage.create_embedding import create_embeddings
from Weft.storage.database import SessionLocal
from Weft.storage.models import Chunk, Conversation, Embedding, Message


def test_ingestion_idempotency():
    # Setup temporary export file
    fd, path = tempfile.mkstemp(suffix=".json")

    convo_id = str(uuid.uuid4())
    msg_id = str(uuid.uuid4())

    mock_data = [
        {
            "id": convo_id,
            "title": "Idempotency Test Convo",
            "create_time": 1718304000,
            "update_time": 1718304000,
            "current_node": msg_id,
            "default_model_slug": "gpt-4o",
            "is_archived": False,
            "mapping": {
                msg_id: {
                    "id": msg_id,
                    "message": {
                        "id": msg_id,
                        "author": {"role": "user"},
                        "create_time": 1718304000,
                        "content": {"parts": ["Hello idempotency."]},
                    },
                }
            },
        }
    ]

    with os.fdopen(fd, "w", encoding="utf8") as f:
        json.dump(mock_data, f)

    try:
        # Run ingestion the first time
        parse_export(path)
        build_chunks()
        create_embeddings()

        db = SessionLocal()
        count1_convos = len(
            db.scalars(select(Conversation).where(Conversation.id == convo_id)).all()
        )
        count1_messages = len(
            db.scalars(select(Message).where(Message.id == msg_id)).all()
        )
        count1_chunks = len(
            db.scalars(select(Chunk).where(Chunk.message_id == msg_id)).all()
        )

        chunk_ids = [
            c.id
            for c in db.scalars(select(Chunk).where(Chunk.message_id == msg_id)).all()
        ]
        count1_embeddings = (
            len(
                db.scalars(
                    select(Embedding).where(Embedding.chunk_order.in_(chunk_ids))
                ).all()
            )
            if chunk_ids
            else 0
        )

        assert count1_convos == 1
        assert count1_messages == 1
        assert count1_chunks > 0
        assert count1_embeddings > 0
        db.close()

        # Run ingestion a second time
        parse_export(path)
        build_chunks()
        create_embeddings()

        db = SessionLocal()
        count2_convos = len(
            db.scalars(select(Conversation).where(Conversation.id == convo_id)).all()
        )
        count2_messages = len(
            db.scalars(select(Message).where(Message.id == msg_id)).all()
        )
        count2_chunks = len(
            db.scalars(select(Chunk).where(Chunk.message_id == msg_id)).all()
        )

        chunk_ids2 = [
            c.id
            for c in db.scalars(select(Chunk).where(Chunk.message_id == msg_id)).all()
        ]
        count2_embeddings = (
            len(
                db.scalars(
                    select(Embedding).where(Embedding.chunk_order.in_(chunk_ids2))
                ).all()
            )
            if chunk_ids2
            else 0
        )
        db.close()

        assert count1_convos == count2_convos
        assert count1_messages == count2_messages
        assert count1_chunks == count2_chunks
        assert count1_embeddings == count2_embeddings

    finally:
        os.remove(path)
