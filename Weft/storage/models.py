from datetime import datetime

from sqlalchemy import (
    String,
    Text,
    Boolean,
    DateTime,
    ForeignKey
)

from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column

from db.database import Base


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(
        String,
        primary_key=True
    )

    title: Mapped[str | None] = mapped_column(
        Text
    )

    create_time: Mapped[datetime | None] = mapped_column(
        DateTime
    )

    update_time: Mapped[datetime | None] = mapped_column(
        DateTime
    )

    current_node: Mapped[str | None] = mapped_column(
        String
    )

    model_slug: Mapped[str | None] = mapped_column(
        String
    )

    is_archived: Mapped[bool] = mapped_column(
        Boolean,
        default=False
    )


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[str] = mapped_column(
        String,
        primary_key=True
    )

    conversation_id: Mapped[str] = mapped_column(
        ForeignKey(
            "conversations.id",
            ondelete="CASCADE"
        )
    )

    parent_id: Mapped[str | None] = mapped_column(
        String
    )

    role: Mapped[str | None] = mapped_column(
        String
    )

    content: Mapped[str | None] = mapped_column(
        Text
    )

    create_time: Mapped[datetime | None] = mapped_column(
        DateTime
    )



class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[int] = mapped_column(
        BigInteger,
        primary_key=True,
        autoincrement=True
    )

    conversation_id: Mapped[str] = mapped_column(
        ForeignKey(
            "conversations.id",
            ondelete="CASCADE"
        )
    )

    message_id: Mapped[str] = mapped_column(
        ForeignKey(
            "messages.id",
            ondelete="CASCADE"
        )
    )

    chunk_order: Mapped[int] = mapped_column(
        Integer
    )

    chunk_text: Mapped[str] = mapped_column(
        Text
    )