from sentence_transformers import SentenceTransformer
from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert

from Weft.config.settings import settings
from Weft.storage.database import SessionLocal
from Weft.storage.models import Chunk, Embedding
from Weft.utils.exceptions import WeftException
from Weft.utils.logger import logger

_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(settings.EMBEDDING_MODEL)
    return _model

def clear_embeddings():
    logger.info("Connecting to database...")
    db = SessionLocal()
    try:
        logger.info("Deleting all embeddings...")
        db.execute(delete(Embedding))
        db.commit()
        logger.info("All embeddings have been deleted.")
    except Exception as e:
        logger.error(f"Failed to delete embeddings: {str(e)}")
        raise WeftException(str(e), e) from e
    finally:
        db.close()
        logger.info("Database session closed.")


def create_embeddings():
    logger.info("Connecting to database...")
    db = SessionLocal()
    try:
        logger.info("Fetching chunks from database...")
        chunks = db.scalars(select(Chunk)).all()

        logger.info(f"Found {len(chunks)} chunks to process.")
        if not chunks:
            logger.info("Stopping: No chunks found in the database. Add chunks first!")
            return

        logger.info("Starting embedding generation (this might take a moment)...")
        for i, chunk in enumerate(chunks, 1):
            # Diagnostic progress counter
            logger.info(f"Encoding chunk {i}/{len(chunks)} (ID: {chunk.id})...")

            embedding_vector = get_model().encode(
                chunk.chunk_text, normalize_embeddings=True
            ).tolist()

            stmt = (
                insert(Embedding)
                .values(
                    conversation_id=chunk.conversation_id,
                    message_id=chunk.message_id,
                    chunk_order=chunk.id,
                    embedding_vector=embedding_vector,
                )
                .on_conflict_do_nothing(index_elements=["chunk_order"])
            )
            db.execute(stmt)
        db.commit()
        logger.info(f"Successfully created embeddings for {len(chunks)} chunks.")

    except WeftException as e:
        db.rollback()
        logger.error(f"Execution failed: {str(e)}")
    finally:
        db.close()
        logger.info("Database session closed.")


if __name__ == "__main__":
    # clear_embeddings()
    create_embeddings()
