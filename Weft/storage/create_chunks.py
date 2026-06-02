from sqlalchemy import select
from Weft.utils.exceptions import WeftException

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from Weft.storage.database import (
    SessionLocal
)

from Weft.storage.models import (
    Message,
    Chunk
)


splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    length_function=len
)


def split_text(text: str):

    if not text:
        return []

    return splitter.split_text(text)


def build_chunks():

    db = SessionLocal()

    try:

        messages = db.scalars(
            select(Message)
        ).all()

        print(
            f"Found {len(messages)} messages"
        )

        total_chunks = 0

        for message in messages:

            if not message.content:
                continue

            chunks = split_text(
                message.content
            )

            for idx, chunk_text in enumerate(
                chunks
            ):

                db.add(
                    Chunk(
                        conversation_id=
                            message.conversation_id,

                        message_id=
                            message.id,

                        chunk_order=idx,

                        chunk_text=chunk_text
                    )
                )

                total_chunks += 1

        db.commit()

        print(
            f"Created {total_chunks} chunks"
        )

    except WeftException as e:

        db.close()


if __name__ == "__main__":
    build_chunks()