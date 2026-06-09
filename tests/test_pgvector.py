import sqlalchemy
from sqlalchemy import select, text
from pgvector.sqlalchemy import Vector
from sentence_transformers import SentenceTransformer
from Weft.storage.models import Embedding, Chunk   
from Weft.utils.exceptions import WeftException 
from Weft.storage.database import SessionLocal


model = SentenceTransformer(
    "BAAI/bge-small-en-v1.5"
)

def test_embedding_vector(query: str):
    db = SessionLocal()
    vector_embedding = model.encode(query, normalize_embeddings=True).tolist()

    distance_attr = Embedding.embedding_vector.cosine_distance(vector_embedding)

    print(f"Embedding vector for '{query}': {vector_embedding}")
    print(f"Vectors from database: {len(list(vectors))}")

    stmt = (
        select(
            Embedding.conversation_id,
            Embedding.message_id,
            Embedding.chunk_order,
            distance_attr.label("distance")
        )
        .order_by("distance")
        .limit(10)
    )
    try:
        result = db.execute(
            stmt,
            {"vector_embedding": vector_embedding}
        ).fetchall()
        print("Top 10 closest embeddings:")
        for row in result:
            print(f"Conversation ID: {row.conversation_id}, Message ID: {row.message_id}, Chunk Order: {row.chunk_order}, Distance: {row.distance}")
    except WeftException as e:
        print(f"Error executing query: {str(e)}")
    finally:
        db.close()


if __name__ == "__main__":
    test_embedding_vector("what is my CP project?")
