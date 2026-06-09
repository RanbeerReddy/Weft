from sentence_transformers import SentenceTransformer
from Weft.storage.models import Embedding, Chunk
from Weft.utils.exceptions import WeftException
from sqlalchemy import select

from Weft.storage.database import SessionLocal


model = SentenceTransformer(
    "BAAI/bge-small-en-v1.5"
)
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

        embeddings_to_add = []
        
        print("Starting embedding generation (this might take a moment)...")
        for i, chunk in enumerate(chunks, 1):
            # Diagnostic progress counter
            print(f"Encoding chunk {i}/{len(chunks)} (ID: {chunk.id})...", end="\r")
            
            embedding_vector = model.encode(
                chunk.chunk_text,
                normalize_embeddings=True
            ).tolist()

            embedding = Embedding(
                conversation_id=chunk.conversation_id,
                message_id=chunk.message_id,
                chunk_order=chunk.id,
                embedding_vector=embedding_vector
            )
            embeddings_to_add.append(embedding)
        
        print("\nAll embeddings generated. Committing to PostgreSQL...")
        if embeddings_to_add:
            db.add_all(embeddings_to_add)
            db.commit()
            print(f"Success! Stored {len(embeddings_to_add)} vector embeddings.")
            
    except Exception as e:
        db.rollback()
        print(f"\n[CRITICAL ERROR] Execution failed: {str(e)}")
    finally:
        db.close()
        print("Database session closed.")

if __name__ == "__main__":
    create_embeddings()
    
