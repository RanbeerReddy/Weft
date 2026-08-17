import uuid

import pytest

from Weft.core.context_assembler import assemble_context
from Weft.storage.database import SessionLocal
from Weft.storage.models import Memory, MemoryType


@pytest.mark.integration
def test_core_retrieval_path():
    db = SessionLocal()
    mem_type_id = str(uuid.uuid4())
    mem_id = str(uuid.uuid4())

    try:
        # Create memory type
        mem_type = MemoryType(
            id=mem_type_id,
            name="Test Goal",
            description="Testing core retrieval",
        )
        db.add(mem_type)
        db.commit()

        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("BAAI/bge-small-en-v1.5")

        unique_target = "FLUBBERGAST_DEPLOYMENT_OF_MOCK_DOCKER_CONTAINER_12345_XYZ."
        unique_query = (
            "How to FLUBBERGAST_DEPLOYMENT_OF_MOCK_DOCKER_CONTAINER_12345_XYZ?"
        )

        # Insert a memory with a known embedding vector
        mem = Memory(
            id=mem_id,
            type_id=mem_type_id,
            status="active",
            value={"target": unique_target},
            embedding_vector=list(
                model.encode(unique_target, normalize_embeddings=True)
            ),
        )
        db.merge(mem)
        db.commit()

        # Now run context assembly with a very similar semantic query
        context = assemble_context(unique_query)

        # The retrieved context should contain the memory value
        assert unique_target in context, "Retrieval core path failed to find memory."

    finally:
        db.delete(db.get(Memory, mem_id))
        db.delete(db.get(MemoryType, mem_type_id))
        db.commit()
        db.close()
