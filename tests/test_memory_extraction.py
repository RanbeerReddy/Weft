import pytest
from Weft.storage.extract_memories import deterministic_extract

@pytest.mark.unit
def test_extract_deterministic():
    memories = deterministic_extract("my goal is to learn rust")
    
    assert len(memories) == 1
    assert memories[0]["type_name"] == "Goal"
    assert memories[0]["value"]["target"] == "to learn rust"
