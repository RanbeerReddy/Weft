import sqlalchemy
from sqlalchemy import select, text

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

    vectors = db.scalars(
        select(Embedding.embedding_vector)
    )
    print(f"Embedding vector for '{query}': {vector_embedding}")
    print(f"Vectors from database: {len(list(vectors))}")

    sql_query = text("""
SELECT
                     conversation_id,
                     message_id,
                     chunk_order,
                     embedding_vector <=> :vector_embedding AS distance
FROM embeddings
ORDER BY distance
                     Limit 10         
"""
    )
    try:
        result = db.execute(
            sql_query,
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
