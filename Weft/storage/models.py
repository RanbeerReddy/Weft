from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import TSVECTOR
from sqlalchemy.orm import Mapped, mapped_column

from Weft.storage.database import Base, engine


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(String, primary_key=True)

    title: Mapped[str | None] = mapped_column(Text)

    create_time: Mapped[datetime | None] = mapped_column(DateTime)

    update_time: Mapped[datetime | None] = mapped_column(DateTime)

    current_node: Mapped[str | None] = mapped_column(String)

    model_slug: Mapped[str | None] = mapped_column(String)

    is_archived: Mapped[bool] = mapped_column(Boolean, default=False)


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[str] = mapped_column(String, primary_key=True)

    conversation_id: Mapped[str] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE")
    )

    parent_id: Mapped[str | None] = mapped_column(String)

    role: Mapped[str | None] = mapped_column(String)

    content: Mapped[str | None] = mapped_column(Text)

    create_time: Mapped[datetime | None] = mapped_column(DateTime)


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)

    conversation_id: Mapped[str] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE")
    )

    message_id: Mapped[str] = mapped_column(
        ForeignKey("messages.id", ondelete="CASCADE")
    )

    chunk_order: Mapped[int] = mapped_column(Integer)

    chunk_text: Mapped[str] = mapped_column(Text)

    chunk_tsvector = mapped_column(TSVECTOR)

    __table_args__ = (
        Index("ix_chunks_tsvector", "chunk_tsvector", postgresql_using="gin"),
        UniqueConstraint("message_id", "chunk_order", name="uq_chunk_msg_order"),
    )


class Embedding(Base):
    __tablename__ = "embeddings"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)

    conversation_id: Mapped[str] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE")
    )

    message_id: Mapped[str] = mapped_column(
        ForeignKey("messages.id", ondelete="CASCADE")
    )

    chunk_order: Mapped[int] = mapped_column(
        ForeignKey("chunks.id", ondelete="CASCADE")
    )

    embedding_vector: Mapped[list[float]] = mapped_column(Vector(384))

    __table_args__ = (UniqueConstraint("chunk_order", name="uq_embedding_chunk_id"),)


class MemoryType(Base):
    __tablename__ = "memory_types"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str | None] = mapped_column(String, unique=True)
    description: Mapped[str | None] = mapped_column(Text)


class Memory(Base):
    __tablename__ = "memories"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    type_id: Mapped[str | None] = mapped_column(
        ForeignKey("memory_types.id", ondelete="SET NULL")
    )
    value: Mapped[dict | None] = mapped_column(JSON)
    embedding_vector: Mapped[list[float] | None] = mapped_column(
        Vector(384), nullable=True
    )
    status: Mapped[str | None] = mapped_column(String)
    create_time: Mapped[datetime | None] = mapped_column(DateTime)
    update_time: Mapped[datetime | None] = mapped_column(DateTime)


class MemoryEvidence(Base):
    __tablename__ = "memory_evidence"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    memory_id: Mapped[str | None] = mapped_column(
        ForeignKey("memories.id", ondelete="CASCADE")
    )
    message_id: Mapped[str | None] = mapped_column(
        ForeignKey("messages.id", ondelete="CASCADE")
    )
    extracted_at: Mapped[datetime | None] = mapped_column(DateTime)
    confidence_score: Mapped[float | None] = mapped_column(Float)
    reasoning: Mapped[str | None] = mapped_column(Text)


class Entity(Base):
    __tablename__ = "entities"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str | None] = mapped_column(String)
    category: Mapped[str | None] = mapped_column(String)


class Relationship(Base):
    __tablename__ = "relationships"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    source_entity_id: Mapped[str | None] = mapped_column(
        ForeignKey("entities.id", ondelete="CASCADE")
    )
    target_entity_id: Mapped[str | None] = mapped_column(
        ForeignKey("entities.id", ondelete="CASCADE")
    )
    relationship_type: Mapped[str | None] = mapped_column(String)
    memory_id: Mapped[str | None] = mapped_column(
        ForeignKey("memories.id", ondelete="SET NULL")
    )


if __name__ == "__main__":
    # Create all tables defined in models.py
    Base.metadata.create_all(bind=engine)
