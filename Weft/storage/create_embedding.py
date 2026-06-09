from sentence_transformers import SentenceTransformer
from Weft.storage.models import Embedding, Chunk
from Weft.utils.exceptions import WeftException
from sqlalchemy import select

from Weft.storage.database import SessionLocal


model = SentenceTransformer(
    "BAAI/bge-small-en-v1.5"
)

def create_embeddings():
    db = SessionLocal()
    try:
        # 1. Fetch all chunks safely so we don't hold the DB cursor open during heavy ML encoding
        chunks = db.scalars(select(Chunk)).all()
        
        embeddings_to_add = []
        
        for chunk in chunks:
            # pgvector natively accepts this standard Python list of floats
            embedding_vector = model.encode(
                chunk.chunk_text,
                normalize_embeddings=True
            ).tolist()

            embedding = Embedding(
                conversation_id=chunk.conversation_id,
                message_id=chunk.message_id,
                chunk_order=chunk.id,   # Using chunk.id maps nicely to chunk_order
                embedding_vector=embedding_vector
            )
            embeddings_to_add.append(embedding)
        
        # 2. Performance Fix: Bulk insert instead of adding one-by-one in a loop
        if embeddings_to_add:
            db.add_all(embeddings_to_add)
            db.commit()
            print(f"Successfully generated and stored {len(embeddings_to_add)} vector embeddings.")
            
    except WeftException as e: # Catch all database errors, not just custom WeftExceptions
        db.rollback()      # Crucial: Roll back the transaction if something fails
        print(f"Error creating embeddings: {str(e)}")
    finally:
        db.close()
    

if __name__ == "__main__":
    create_embeddings()
    
