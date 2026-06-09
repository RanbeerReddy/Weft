import sqlalchemy
from sqlalchemy import select
from sentence_transformers import SentenceTransformer
from Weft.storage.models import Embedding
from Weft.storage.database import SessionLocal

model = SentenceTransformer(
    "BAAI/bge-small-en-v1.5"
)

def test_embedding_vector(query: str):
    db = SessionLocal()
    try:
        # 1. Generate the query vector
        vector_embedding = model.encode(query, normalize_embeddings=True).tolist()
        print(f"Generated embedding vector for '{query}' (Dimensions: {len(vector_embedding)})")

        # 2. Define the pgvector distance attribute calculation
        distance_attr = Embedding.embedding_vector.cosine_distance(vector_embedding)

        # 3. Build the ORM compiled statement
        stmt = (
            select(
                Embedding.conversation_id,
                Embedding.message_id,
                Embedding.chunk_order,
                distance_attr.label("distance")
            )
            .order_by("distance") # Strings or labels work perfectly here
            .limit(10)
        )
        
        # 4. Execute (No extra parameters dictionary needed)
        result = db.execute(stmt).fetchall()
        
        print(f"\nTop {len(result)} closest embeddings from database:")
        for row in result:
            print(
                f"Conversation ID: {row.conversation_id} | "
                f"Message ID: {row.message_id} | "
                f"Chunk Order: {row.chunk_order} | "
                f"Distance: {row.distance:.4f}" # Format distance to 4 decimal places
            )
            
    except Exception as e:
        print(f"Error executing query: {str(e)}")
    finally:
        db.close()


if __name__ == "__main__":
    test_embedding_vector("what is my CP project?")