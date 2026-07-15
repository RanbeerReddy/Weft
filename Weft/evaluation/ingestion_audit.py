"""Phase 1 — Ingestion Audit.

Traces the complete ingestion pipeline and verifies every stage:

    Conversation JSON → Messages → Chunks → Embeddings → pgvector

Detects:
    - Skipped conversations
    - Skipped messages
    - Empty chunks
    - Chunk generation failures (messages with no chunks)
    - Embedding failures (chunks with no embeddings)
    - Database insertion failures
    - Duplicate chunks
    - Orphaned embeddings
"""

import json
import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Set

from sqlalchemy import select, func, text

from Weft.storage.database import SessionLocal
from Weft.storage.models import Conversation, Message, Chunk, Embedding


def count_json_conversations(json_path: str) -> Dict[str, Any]:
    """Count conversations and messages in the source JSON file."""
    path = Path(json_path)
    if not path.exists():
        return {"error": f"File not found: {json_path}"}

    with open(path, "r", encoding="utf-8") as f:
        conversations = json.load(f)

    total_conversations = len(conversations)
    total_messages = 0
    empty_content_messages = 0
    conversations_with_no_messages = 0
    conversation_ids = set()

    for convo in conversations:
        conversation_ids.add(convo.get("id"))
        mapping = convo.get("mapping", {})
        msg_count = 0
        for node_id, node in mapping.items():
            message = node.get("message")
            if not message:
                continue
            msg_count += 1
            total_messages += 1

            # Check for empty content
            content = message.get("content", {})
            parts = content.get("parts", []) if content else []
            text_parts = [str(p) for p in parts if p]
            combined = "\n".join(text_parts).strip()
            if not combined:
                empty_content_messages += 1

        if msg_count == 0:
            conversations_with_no_messages += 1

    return {
        "total_conversations": total_conversations,
        "total_messages_in_mapping": total_messages,
        "empty_content_messages": empty_content_messages,
        "conversations_with_no_messages": conversations_with_no_messages,
        "conversation_ids": conversation_ids,
    }


def count_db_records(db) -> Dict[str, int]:
    """Count records at each pipeline stage in the database."""
    convos = db.scalar(select(func.count(Conversation.id)))
    msgs = db.scalar(select(func.count(Message.id)))
    chunks = db.scalar(select(func.count(Chunk.id)))
    embeds = db.scalar(select(func.count(Embedding.id)))

    return {
        "conversations": convos,
        "messages": msgs,
        "chunks": chunks,
        "embeddings": embeds,
    }


def find_skipped_conversations(db, json_ids: Set[str]) -> List[str]:
    """Find conversation IDs in JSON but not in DB."""
    db_ids_rows = db.execute(select(Conversation.id)).fetchall()
    db_ids = {row[0] for row in db_ids_rows}
    return sorted(json_ids - db_ids)


def find_messages_without_chunks(db) -> List[Dict[str, Any]]:
    """Find messages that have content but no chunks."""
    # Messages with non-empty content that have no matching chunk
    stmt = (
        select(Message.id, Message.conversation_id, Message.role)
        .outerjoin(Chunk, Chunk.message_id == Message.id)
        .where(Message.content.isnot(None))
        .where(Message.content != "")
        .where(Chunk.id.is_(None))
    )
    results = db.execute(stmt).fetchall()
    return [
        {
            "message_id": r[0],
            "conversation_id": r[1],
            "role": r[2],
        }
        for r in results
    ]


def find_chunks_without_embeddings(db) -> List[Dict[str, Any]]:
    """Find chunks that have no corresponding embedding."""
    stmt = (
        select(Chunk.id, Chunk.conversation_id, Chunk.message_id)
        .outerjoin(Embedding, Embedding.chunk_order == Chunk.id)
        .where(Embedding.id.is_(None))
    )
    results = db.execute(stmt).fetchall()
    return [
        {
            "chunk_id": r[0],
            "conversation_id": r[1],
            "message_id": r[2],
        }
        for r in results
    ]


def find_empty_chunks(db) -> List[Dict[str, Any]]:
    """Find chunks with empty or whitespace-only text."""
    stmt = select(Chunk.id, Chunk.conversation_id, Chunk.message_id).where(
        (Chunk.chunk_text.is_(None)) | (func.trim(Chunk.chunk_text) == "")
    )
    results = db.execute(stmt).fetchall()
    return [
        {
            "chunk_id": r[0],
            "conversation_id": r[1],
            "message_id": r[2],
        }
        for r in results
    ]


def find_duplicate_chunks(db) -> List[Dict[str, Any]]:
    """Find duplicate chunks (same message_id + same chunk_text)."""
    stmt = (
        select(
            Chunk.message_id,
            Chunk.chunk_text,
            func.count(Chunk.id).label("count"),
        )
        .group_by(Chunk.message_id, Chunk.chunk_text)
        .having(func.count(Chunk.id) > 1)
    )
    results = db.execute(stmt).fetchall()
    return [
        {
            "message_id": r[0],
            "chunk_text_preview": r[1][:80] if r[1] else "",
            "duplicate_count": r[2],
        }
        for r in results
    ]


def find_orphaned_embeddings(db) -> int:
    """Find embeddings whose chunk_order doesn't match any chunk.id."""
    stmt = (
        select(func.count(Embedding.id))
        .outerjoin(Chunk, Chunk.id == Embedding.chunk_order)
        .where(Chunk.id.is_(None))
    )
    return db.scalar(stmt)


def find_messages_with_empty_content(db) -> int:
    """Count messages stored with NULL or empty content."""
    stmt = select(func.count(Message.id)).where(
        (Message.content.is_(None)) | (Message.content == "")
    )
    return db.scalar(stmt)


def check_pgvector_index(db) -> List[str]:
    """Check if there's an index on the embedding_vector column."""
    stmt = text(
        """
        SELECT indexname, indexdef
        FROM pg_indexes
        WHERE tablename = 'embeddings'
    """
    )
    results = db.execute(stmt).fetchall()
    return [{"name": r[0], "definition": r[1]} for r in results]


def run_audit(json_path: str = "conversations.json") -> Dict[str, Any]:
    """Run the complete ingestion audit."""
    print("=" * 70)
    print("PHASE 1 — INGESTION AUDIT")
    print("=" * 70)

    report = {
        "timestamp": datetime.now().isoformat(),
        "source_file": json_path,
    }

    # 1. Count JSON source
    print("\n[1/7] Analyzing source JSON...")
    json_stats = count_json_conversations(json_path)
    if "error" in json_stats:
        print(f"  [!] {json_stats['error']}")
        report["json_error"] = json_stats["error"]
        json_ids = set()
    else:
        json_ids = json_stats.pop("conversation_ids")
        report["json_source"] = json_stats
        print(f"  Conversations in JSON: {json_stats['total_conversations']}")
        print(f"  Messages in JSON:      {json_stats['total_messages_in_mapping']}")
        print(f"  Empty content msgs:    {json_stats['empty_content_messages']}")

    # 2. Count DB records
    db = SessionLocal()
    try:
        print("\n[2/7] Counting database records...")
        db_counts = count_db_records(db)
        report["db_counts"] = db_counts
        print(f"  Conversations: {db_counts['conversations']}")
        print(f"  Messages:      {db_counts['messages']}")
        print(f"  Chunks:        {db_counts['chunks']}")
        print(f"  Embeddings:    {db_counts['embeddings']}")

        # 3. Skipped conversations
        print("\n[3/7] Finding skipped conversations...")
        skipped = find_skipped_conversations(db, json_ids)
        report["skipped_conversations"] = {
            "count": len(skipped),
            "ids": skipped[:20],  # first 20 only
        }
        print(f"  Skipped conversations: {len(skipped)}")

        # 4. Messages without chunks
        print("\n[4/7] Finding messages without chunks...")
        no_chunks = find_messages_without_chunks(db)
        report["messages_without_chunks"] = {
            "count": len(no_chunks),
            "samples": no_chunks[:10],
        }
        print(f"  Messages with content but no chunks: {len(no_chunks)}")

        # 5. Chunks without embeddings
        print("\n[5/7] Finding chunks without embeddings...")
        no_embeds = find_chunks_without_embeddings(db)
        report["chunks_without_embeddings"] = {
            "count": len(no_embeds),
            "samples": no_embeds[:10],
        }
        print(f"  Chunks with no embeddings: {len(no_embeds)}")

        # 6. Data quality
        print("\n[6/7] Checking data quality...")
        empty_chunks = find_empty_chunks(db)
        duplicates = find_duplicate_chunks(db)
        orphans = find_orphaned_embeddings(db)
        empty_msgs = find_messages_with_empty_content(db)

        report["data_quality"] = {
            "empty_chunks": {"count": len(empty_chunks), "samples": empty_chunks[:5]},
            "duplicate_chunks": {
                "count": len(duplicates),
                "total_excess": sum(d["duplicate_count"] - 1 for d in duplicates),
                "samples": duplicates[:5],
            },
            "orphaned_embeddings": orphans,
            "empty_content_messages_in_db": empty_msgs,
        }
        print(f"  Empty chunks:           {len(empty_chunks)}")
        print(f"  Duplicate chunk groups:  {len(duplicates)}")
        print(f"  Orphaned embeddings:    {orphans}")
        print(f"  Empty-content messages: {empty_msgs}")

        # 7. Index check
        print("\n[7/7] Checking pgvector indexes...")
        indexes = check_pgvector_index(db)
        report["indexes"] = indexes
        if indexes:
            for idx in indexes:
                print(f"  Index: {idx['name']}")
        else:
            print("  [!] No indexes found on embeddings table")

        # Summary
        print("\n" + "=" * 70)
        print("INGESTION AUDIT SUMMARY")
        print("=" * 70)

        json_total = json_stats.get("total_conversations", "?")
        db_total = db_counts["conversations"]
        print(f"\n  Pipeline Completeness:")
        print(f"    JSON → DB Conversations: {json_total} → {db_total}")
        print(f"    DB Messages → Chunks:    {db_counts['messages']} → {db_counts['chunks']}")
        print(f"    DB Chunks → Embeddings:  {db_counts['chunks']} → {db_counts['embeddings']}")

        if db_counts["chunks"] > 0:
            embed_rate = db_counts["embeddings"] / db_counts["chunks"] * 100
            print(f"    Embedding coverage:      {embed_rate:.1f}%")

        issues = []
        if len(skipped) > 0:
            issues.append(f"{len(skipped)} conversations skipped")
        if len(no_chunks) > 0:
            issues.append(f"{len(no_chunks)} messages have no chunks")
        if len(no_embeds) > 0:
            issues.append(f"{len(no_embeds)} chunks have no embeddings")
        if len(empty_chunks) > 0:
            issues.append(f"{len(empty_chunks)} empty chunks")
        if len(duplicates) > 0:
            issues.append(f"{len(duplicates)} duplicate chunk groups")

        if issues:
            print(f"\n  Issues Found ({len(issues)}):")
            for issue in issues:
                print(f"    ⚠  {issue}")
        else:
            print("\n  ✓ No issues detected")

        report["issues_summary"] = issues

    finally:
        db.close()

    return report


def main():
    parser = argparse.ArgumentParser(description="Weft Ingestion Audit")
    parser.add_argument(
        "--json-path",
        type=str,
        default="conversations.json",
        help="Path to source conversations.json",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="ingestion_audit_report.json",
        help="Output report file",
    )
    args = parser.parse_args()

    report = run_audit(args.json_path)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n[+] Report saved to {output_path}")


if __name__ == "__main__":
    main()
