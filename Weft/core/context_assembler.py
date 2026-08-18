import sys

from sentence_transformers import SentenceTransformer
from sqlalchemy import select

from Weft.config.settings import settings
from Weft.storage.database import SessionLocal
from Weft.storage.models import Embedding, Memory, MemoryType, Message
from Weft.utils.exceptions import WeftException


def assemble_context(query: str) -> str:  # noqa: C901
    db = SessionLocal()
    try:
        context_parts = []

        # 1. Global Preferences (Always Injected)
        preferences = db.scalars(
            select(Memory)
            .join(MemoryType)
            .where(Memory.status == "active", MemoryType.name == "Preference")
        ).all()

        if preferences:
            context_parts.append("### User Preferences ###")
            for p in preferences:
                context_parts.append(f"- {p.value}")

        # 2. Query-Specific Memories
        # (Semantic search for Phase 4)
        try:
            model = SentenceTransformer(settings.EMBEDDING_MODEL)
            query_embedding = model.encode(query, normalize_embeddings=True).tolist()

            # Retrieve memories with L2 distance < 1.0
            # (equivalent to cosine similarity > 0.5)
            # Ordered by distance ascending, limited to top 5
            memories = db.scalars(
                select(Memory)
                .join(MemoryType)
                .where(Memory.status == "active", MemoryType.name != "Preference")
                .where(Memory.embedding_vector.is_not(None))
                .order_by(Memory.embedding_vector.l2_distance(query_embedding))
                .limit(5)
            ).all()

            # Deduplicate by memory ID
            # (db should return unique ones, but good practice)
            # Enforce threshold locally since distance isn't
            # easily returned in scalar
            seen_mems = set()
            relevant_mems = []
            for m in memories:
                if m.id not in seen_mems:
                    if (
                        m.embedding_vector is not None
                        and sum(
                            (a - b) ** 2
                            for a, b in zip(m.embedding_vector, query_embedding)
                        )
                        < 1.0
                    ):
                        relevant_mems.append(m)
                        seen_mems.add(m.id)

            if relevant_mems:
                context_parts.append("\n### Relevant Long-term Memories ###")
                for mem in relevant_mems:
                    t = db.get(MemoryType, mem.type_id)
                    t_name = t.name if t else "Unknown"
                    context_parts.append(f"- [{t_name}]: {mem.value}")
        except Exception as e:
            # Fallback if something fails, log or ignore
            raise WeftException(f"Failed to retrieve long-term memories: {e}") from e

        # 3. Semantic Search on Conversation History
        try:
            model = SentenceTransformer(settings.EMBEDDING_MODEL)
            query_embedding = model.encode(query, normalize_embeddings=True).tolist()

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
                context_parts.append("\n### Semantic Conversation History ###")
                for row in results:
                    msg = db.get(Message, row.message_id)
                    if msg:
                        context_parts.append(f"[Message {msg.id}]: {msg.content}")
        except Exception as e:
            raise WeftException(str(e), e) from e

        # 4. Assembly
        final_prompt = (
            "You are a helpful assistant with access to the user's long-term memory and past conversations.\n"  # noqa: E501
            "Use the provided context to answer the query accurately and adhere to the user's preferences.\n\n"  # noqa: E501
            f"{chr(10).join(context_parts)}\n\n"
            f"### User Query ###\n{query}\n"
        )
        return final_prompt

    finally:
        db.close()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        q = " ".join(sys.argv[1:])
    else:
        q = "What should I focus on for my Sensovibe internship?"

    print(assemble_context(q))
