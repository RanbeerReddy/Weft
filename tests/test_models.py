import uuid

import pytest

from Weft.storage.models import (
    Conversation,
    Memory,
    MemoryEvidence,
    MemoryType,
    Message,
)


@pytest.mark.unit
def test_conversation_model_creation():
    convo_id = str(uuid.uuid4())
    convo = Conversation(id=convo_id, title="Test Convo")
    assert convo.title == "Test Convo"
    assert convo.id == convo_id


@pytest.mark.unit
def test_message_model_creation():
    msg_id = str(uuid.uuid4())
    msg = Message(
        id=msg_id, role="user", content="Hello", conversation_id=str(uuid.uuid4())
    )
    assert msg.role == "user"
    assert msg.content == "Hello"


@pytest.mark.unit
def test_memory_models_creation():
    type_id = str(uuid.uuid4())
    mem_type = MemoryType(id=type_id, name="Preference", description="User preference")
    assert mem_type.name == "Preference"

    mem_id = str(uuid.uuid4())
    memory = Memory(
        id=mem_id, type_id=type_id, value={"target": "Python"}, status="active"
    )
    assert memory.value == {"target": "Python"}
    assert memory.status == "active"

    evidence_id = str(uuid.uuid4())
    evidence = MemoryEvidence(
        id=evidence_id,
        memory_id=mem_id,
        message_id=str(uuid.uuid4()),
        confidence_score=0.95,
    )
    assert evidence.memory_id == mem_id
    assert evidence.confidence_score == 0.95
