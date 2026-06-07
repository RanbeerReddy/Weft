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
        chunks = db.scalars(
            select(Chunk)
        )
        for chunk in chunks:
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
            db.add(embedding)
        db.commit()
    except WeftException as e:
        print(
            f"Error creating embeddings: {str(e)}"
        )
    finally:
        db.close()
    

if __name__ == "__main__":
    create_embeddings()