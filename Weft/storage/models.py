from sqlalchemy import (
    Column,
    Integer,
    String,
    DateTime,
    Text,
    DateTime,
    ForeignKey
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

class Conversation(Base):
    __tablename__ = 'conversations'
    
    id = Column(Integer, primary_key=True)
    title = Column(Text, nullable=False)
    created_at = Column(DateTime)
    update_time = Column(DateTime)

class Message(Base):
    __tablename__ = "messages"

    id = Column(String, primary_key=True)
    conversation_id = Column(
        String,
        ForeignKey("conversation.id", ondelete="CASCADE"),
        nullable=False,
    )
    role = Column(String(50))
    content = Column(Text)
    create_time = Column(DateTIme)


class Chunk(Base):
    __tablename__ = "chunks"

    id = Column(Integer, primary_keys=True)
    message_id = Column(
        String,
        ForeignKey("messages.id", ondelete="CASCADE"),
        nullable=False,
    )
    conversation_id = Column(
        String,
        ForeignKey("conversation.id", ondelete="CASCADE")
        nullable=False
    )

    chunk_order = Column(Integer)
    chunk_text = Column(Text)