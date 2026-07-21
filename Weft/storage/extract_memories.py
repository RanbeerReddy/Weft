import json
import uuid
import re
from datetime import datetime
from Weft.storage.database import SessionLocal
from Weft.storage.models import Message, Memory, MemoryEvidence, MemoryType
from sqlalchemy import select, String

# Dummy deterministic rule-based extractor
def deterministic_extract(message_content):
    memories = []
    text = message_content.lower()
    
    # Example rules from the prompt/design
    if "i am preparing for" in text or "my goal is" in text:
        match = re.search(r'(?:i am preparing for|my goal is)\s+(.*)', text)
        if match:
            memories.append({
                "type_name": "Goal",
                "value": {"target": match.group(1).strip()},
                "confidence": 0.9,
                "reasoning": "Matched deterministic goal regex."
            })
            
    if "i got selected for" in text:
        match = re.search(r'i got selected for\s+(.*)', text)
        if match:
            memories.append({
                "type_name": "Experience",
                "value": {"event": match.group(1).strip()},
                "confidence": 0.85,
                "reasoning": "Matched deterministic experience regex."
            })
            
    return memories

def mock_llm_extract(message_content):
    # Stub for future LLM integration
    return []

def extract_memories():
    print("Connecting to database...")
    db = SessionLocal()
    try:
        # Ensure base memory types exist
        types = ["Goal", "Project", "Experience", "Preference", "Skill"]
        type_map = {}
        for t_name in types:
            t = db.scalars(select(MemoryType).where(MemoryType.name == t_name)).first()
            if not t:
                t = MemoryType(id=str(uuid.uuid4()), name=t_name, description=f"Extracted {t_name}")
                db.add(t)
            type_map[t_name] = t
        db.commit()

        print("Fetching messages from database...")
        messages = db.scalars(select(Message)).all()
        print(f"Found {len(messages)} messages to process.")
        
        extracted_count = 0
        for i, msg in enumerate(messages, 1):
            if not msg.content:
                continue
            
            # 1. Deterministic
            extracted = deterministic_extract(msg.content)
            # 2. LLM (mock)
            extracted.extend(mock_llm_extract(msg.content))
            
            for ext in extracted:
                type_obj = type_map.get(ext["type_name"])
                if not type_obj:
                    continue
                
                # Simple deduplication check based on stringified value
                val_str = json.dumps(ext["value"])
                # Note: this is a naive dedup for phase 2.
                # In production, this would use pgvector on memory embeddings.
                existing_mem = db.scalars(
                    select(Memory).where(Memory.value.cast(String) == val_str)
                ).first()
                
                if existing_mem:
                    # Update evidence for existing memory
                    evidence = MemoryEvidence(
                        id=str(uuid.uuid4()),
                        memory_id=existing_mem.id,
                        message_id=msg.id,
                        extracted_at=datetime.utcnow(),
                        confidence_score=ext["confidence"],
                        reasoning=ext["reasoning"] + " (Deduplicated)"
                    )
                    db.add(evidence)
                else:
                    # Create new memory
                    mem_id = str(uuid.uuid4())
                    memory = Memory(
                        id=mem_id,
                        type_id=type_obj.id,
                        value=ext["value"],
                        status="active",
                        create_time=datetime.utcnow(),
                        update_time=datetime.utcnow()
                    )
                    
                    evidence = MemoryEvidence(
                        id=str(uuid.uuid4()),
                        memory_id=mem_id,
                        message_id=msg.id,
                        extracted_at=datetime.utcnow(),
                        confidence_score=ext["confidence"],
                        reasoning=ext["reasoning"]
                    )
                    
                    db.add(memory)
                    db.add(evidence)
                
                extracted_count += 1
                
        db.commit()
        print(f"\nSuccessfully extracted {extracted_count} memories.")
    except Exception as e:
        db.rollback()
        print(f"[ERROR] Extraction failed: {str(e)}")
    finally:
        db.close()

if __name__ == "__main__":
    extract_memories()
