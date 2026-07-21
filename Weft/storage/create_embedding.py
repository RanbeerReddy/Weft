from sentence_transformers import SentenceTransformer
from sqlalchemy import select

from Weft.storage.database import SessionLocal
from Weft.storage.models import Chunk, Embedding
from Weft.utils.exceptions import WeftException

model = SentenceTransformer("BAAI/bge-small-en-v1.5")

from sqlalchemy import delete


def clear_embeddings():
    print("Connecting to database...")
    db = SessionLocal()
    try:
        print("Deleting all embeddings...")
        db.execute(delete(Embedding))
        db.commit()
        print("All embeddings have been deleted.")
    except Exception as e:
        raise WeftException(str(e), e) from e
        print(f"[ERROR] Failed to delete embeddings: {str(e)}")
    finally:
        db.close()
        print("Database session closed.")


def create_embeddings():
    print("Connecting to database...")
    db = SessionLocal()
    try:
        print("Fetching chunks from database...")
        chunks = db.scalars(select(Chunk)).all()

        print(f"Found {len(chunks)} chunks to process.")
        if not chunks:
            print("Stopping: No chunks found in the database. Add chunks first!")
            return

        print("Starting embedding generation (this might take a moment)...")
        for i, chunk in enumerate(chunks, 1):
            # Diagnostic progress counter
            print(f"Encoding chunk {i}/{len(chunks)} (ID: {chunk.id})...", end="\r")

            embedding_vector = model.encode(
                chunk.chunk_text, normalize_embeddings=True
            ).tolist()

            embedding = Embedding(
                conversation_id=chunk.conversation_id,
                message_id=chunk.message_id,
                chunk_order=chunk.id,
                embedding_vector=embedding_vector,
            )
            db.add(embedding)
        db.commit()
        print(f"\nSuccessfully created embeddings for {len(chunks)} chunks.")

    except WeftException as e:
        db.rollback()
        print(f"\n[CRITICAL ERROR] Execution failed: {str(e)}")
    finally:
        db.close()
        print("Database session closed.")


if __name__ == "__main__":
    # clear_embeddings()
    create_embeddings()
