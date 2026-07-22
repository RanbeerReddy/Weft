import sys

from sentence_transformers import SentenceTransformer
from sqlalchemy import select

from Weft.storage.database import SessionLocal
from Weft.storage.models import Embedding, Memory, MemoryType, Message
from Weft.utils.exceptions import WeftException


def search_memories(query: str):
    print(f"--- Dual Retrieval Search for: '{query}' ---")
    db = SessionLocal()
    try:
        # 1. Global Context & Structured Memories
        # In a real system, we'd use NER. Here we do simple keyword matching on JSON string.  # noqa: E501
        print("\n[Structured Memories]")
        memories = db.scalars(select(Memory).where(Memory.status == "active")).all()
        found_memories = []
        for mem in memories:
            val_str = str(mem.value).lower()
            if any(word.lower() in val_str for word in query.split()):
                found_memories.append(mem)

        if found_memories:
            for mem in found_memories:
                t = db.get(MemoryType, mem.type_id)
                t_name = t.name if t else "Unknown"
                print(f" - [{t_name}]: {mem.value}")
        else:
            print(" - No relevant structured memories found.")

        # 2. Semantic Search on Chunks (Legacy Pipeline)
        print("\n[Semantic Conversation Context]")
        try:
            model = SentenceTransformer("BAAI/bge-small-en-v1.5")
            query_embedding = model.encode(query, normalize_embeddings=True).tolist()

            # Using pgvector L2 distance or cosine similarity
            # Since the Vector column is setup, we can use l2_distance or cosine_distance  # noqa: E501
            # We'll use l2_distance for simplicity.
            results = db.execute(
                select(
                    Embedding.message_id,
                    Embedding.embedding_vector.l2_distance(query_embedding).label(
                        "dist"
                    ),
                )
                .order_by("dist")
                .limit(3)
            ).all()

            if results:
                for row in results:
                    msg = db.get(Message, row.message_id)
                    if msg:
                        print(
                            f" - [Message {msg.id}] (Dist: {row.dist:.4f}): {msg.content[:100]}..."  # noqa: E501
                        )
            else:
                print(" - No semantic chunks found. (Have embeddings been created?)")

        except Exception as vec_err:
            raise WeftException(str(vec_err), vec_err) from vec_err

    finally:
        db.close()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = "Sensovibe internship"
    search_memories(query)
