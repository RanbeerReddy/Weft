from unittest.mock import patch

import pytest

from Weft.core.context_assembler import assemble_context
from Weft.storage.models import Memory, MemoryType


@pytest.fixture
def seeded_db(db_session):
    import uuid
    from sqlalchemy import select
    
    # Seed some memory types and memories for the test
    pref_type = db_session.scalars(select(MemoryType).where(MemoryType.name == "Preference")).first()
    if not pref_type:
        pref_type = MemoryType(id=str(uuid.uuid4()), name="Preference", description="User preferences")
        db_session.add(pref_type)
        
    goal_type = db_session.scalars(select(MemoryType).where(MemoryType.name == "Goal")).first()
    if not goal_type:
        goal_type = MemoryType(id=str(uuid.uuid4()), name="Goal", description="User goals")
        db_session.add(goal_type)
        
    db_session.commit()

    mem1 = Memory(id=str(uuid.uuid4()), type_id=pref_type.id, value={"target": "Prefers concise answers"}, status="active")
    mem2 = Memory(id=str(uuid.uuid4()), type_id=goal_type.id, value={"target": "Learn rust programming"}, status="active")
    db_session.add_all([mem1, mem2])
    db_session.commit()
    return db_session


@pytest.mark.integration
@patch("Weft.core.context_assembler.SessionLocal")
def test_assemble_context(mock_session_local, seeded_db):
    mock_session_local.return_value = seeded_db

    query = "How do I learn rust?"
    result = assemble_context(query)

    # Assertions
    assert "Prefers concise answers" in result  # Preference always injected
    assert "Learn rust programming" in result  # Relevant memory injected
    assert "How do I learn rust?" in result  # Query is included
