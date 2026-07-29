from langchain_text_splitters import RecursiveCharacterTextSplitter
from sqlalchemy import select

from Weft.config.settings import settings
from Weft.storage.database import SessionLocal
from Weft.storage.models import Chunk, Message
from Weft.utils.exceptions import WeftException
from Weft.utils.logger import logger

splitter = RecursiveCharacterTextSplitter(
    chunk_size=settings.CHUNK_SIZE,
    chunk_overlap=settings.CHUNK_OVERLAP,
    length_function=len,
)


def split_text(text: str):
    if not text:
        return []

    return splitter.split_text(text)


def build_chunks():
    db = SessionLocal()

    try:
        messages = db.scalars(select(Message)).all()

        logger.info(f"Found {len(messages)} messages")

        total_chunks = 0

        for message in messages:
            if not message.content:
                continue

            chunks = split_text(message.content)

            for idx, chunk_text in enumerate(chunks):
                db.add(
                    Chunk(
                        conversation_id=message.conversation_id,
                        message_id=message.id,
                        chunk_order=idx,
                        chunk_text=chunk_text,
                    )
                )

                total_chunks += 1

        db.commit()

        logger.info(f"Created {total_chunks} chunks")

    except WeftException:
        db.close()


if __name__ == "__main__":
    build_chunks()
