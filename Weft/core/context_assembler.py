import sys

from sentence_transformers import SentenceTransformer
from sqlalchemy import select

from Weft.storage.database import SessionLocal
from Weft.storage.models import Embedding, Memory, MemoryType, Message
from Weft.utils.exceptions import WeftException


def assemble_context(query: str) -> str:
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
        # (Naive string matching for Phase 4)
        memories = db.scalars(
            select(Memory)
            .join(MemoryType)
            .where(Memory.status == "active", MemoryType.name != "Preference")
        ).all()

        query_words = query.lower().split()
        relevant_mems = [
            m for m in memories if any(w in str(m.value).lower() for w in query_words)
        ]

        if relevant_mems:
            context_parts.append("\n### Relevant Long-term Memories ###")
            for mem in relevant_mems:
                t = db.get(MemoryType, mem.type_id)
                t_name = t.name if t else "Unknown"
                context_parts.append(f"- [{t_name}]: {mem.value}")

        # 3. Semantic Search on Conversation History
        try:
            model = SentenceTransformer("BAAI/bge-small-en-v1.5")
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
