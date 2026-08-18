import sys

from sentence_transformers import SentenceTransformer
from sqlalchemy import select

from Weft.config.settings import settings
from Weft.storage.database import SessionLocal
from Weft.storage.models import Embedding, Memory, MemoryType, Message
from Weft.utils.exceptions import WeftException
from Weft.utils.logger import logger


def search_memories(query: str):
    logger.info(f"--- Dual Retrieval Search for: '{query}' ---")
    db = SessionLocal()
    try:
        # 1. Global Context & Structured Memories
        # In a real system, we'd use NER. Here we do simple keyword matching on JSON string.  # noqa: E501
        logger.info("[Structured Memories]")

        try:
            model = SentenceTransformer(settings.EMBEDDING_MODEL)
            query_embedding = model.encode(query, normalize_embeddings=True).tolist()

            # Retrieve memories with L2 distance < 1.0
            # (equivalent to cosine similarity > 0.5)
            # Ordered by distance ascending, limited to top 5
            memories = db.scalars(
                select(Memory)
                .where(Memory.status == "active")
                .where(Memory.embedding_vector.is_not(None))
                .order_by(Memory.embedding_vector.l2_distance(query_embedding))
                .limit(5)
            ).all()

            found_memories = [
                mem
                for mem in memories
                if mem.embedding_vector
                and sum(
                    (a - b) ** 2 for a, b in zip(mem.embedding_vector, query_embedding)
                )
                < 1.0
            ]

            # Note: SQLAlchemy doesn't return the distance
            # as a property of the scalar directly unless we
            # select it explicitly. We'll recalculate it or
            # just trust the DB order but we need to enforce
            # the threshold. Since pgvector l2_distance works
            # in SQL, we could also do it there, but for
            # simplicity we just filter locally or in SQL.

        except Exception as e:
            logger.error(f"Failed to encode or retrieve memories: {e}")
            found_memories = []

        if found_memories:
            for mem in found_memories:
                t = db.get(MemoryType, mem.type_id)
                t_name = t.name if t else "Unknown"
                logger.info(f" - [{t_name}]: {mem.value}")
        else:
            logger.info(" - No relevant structured memories found.")

        # 2. Semantic Search on Chunks (Legacy Pipeline)
        logger.info("[Semantic Conversation Context]")
        try:
            model = SentenceTransformer(settings.EMBEDDING_MODEL)
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
                        content_snippet = (msg.content or "")[:100]
                        logger.info(
                            f" - [Message {msg.id}] (Dist: {row.dist:.4f}): {content_snippet}..."  # noqa: E501
                        )
            else:
                logger.info(
                    " - No semantic chunks found. (Have embeddings been created?)"
                )

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
