"""Phase 2 — Corpus Validation.

Given a list of expected phrases (from benchmark queries), searches the
ENTIRE database to verify each phrase exists.

For each phrase reports:
    - Exists? (yes/no)
    - How many chunks contain it?
    - Conversation IDs
    - Message IDs
    - Whether it exists in raw messages but was lost during chunking

Automatically flags benchmark queries whose expected data does not exist
in the corpus, preventing invalid benchmark failures from being blamed
on the retrieval system.
"""

import json
import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

from sqlalchemy import select, func

from Weft.storage.database import SessionLocal
from Weft.storage.models import Conversation, Message, Chunk


def search_phrase_in_chunks(db, phrase: str) -> Dict[str, Any]:
    """Search for a phrase across all chunks (case-insensitive)."""
    pattern = f"%{phrase}%"

    # Find matching chunks
    stmt = (
        select(
            Chunk.id,
            Chunk.conversation_id,
            Chunk.message_id,
            Chunk.chunk_text,
        )
        .where(func.lower(Chunk.chunk_text).contains(phrase.lower()))
    )
    results = db.execute(stmt).fetchall()

    conversation_ids = set()
    message_ids = set()
    chunk_previews = []

    for r in results:
        conversation_ids.add(r.conversation_id)
        message_ids.add(r.message_id)
        chunk_previews.append({
            "chunk_id": r.id,
            "conversation_id": r.conversation_id,
            "message_id": r.message_id,
            "preview": r.chunk_text[:150] if r.chunk_text else "",
        })

    return {
        "exists_in_chunks": len(results) > 0,
        "chunk_count": len(results),
        "conversation_ids": sorted(conversation_ids),
        "message_ids": sorted(message_ids),
        "chunk_previews": chunk_previews[:5],  # first 5 only
    }


def search_phrase_in_messages(db, phrase: str) -> Dict[str, Any]:
    """Search for a phrase across all raw messages (case-insensitive)."""
    stmt = (
        select(
            Message.id,
            Message.conversation_id,
            Message.role,
            Message.content,
        )
        .where(func.lower(Message.content).contains(phrase.lower()))
    )
    results = db.execute(stmt).fetchall()

    conversation_ids = set()
    message_ids = set()
    message_previews = []

    for r in results:
        conversation_ids.add(r.conversation_id)
        message_ids.add(r.id)
        # Find the phrase in context
        content_lower = r.content.lower()
        idx = content_lower.find(phrase.lower())
        start = max(0, idx - 50)
        end = min(len(r.content), idx + len(phrase) + 50)
        context = r.content[start:end]

        message_previews.append({
            "message_id": r.id,
            "conversation_id": r.conversation_id,
            "role": r.role,
            "context": f"...{context}...",
        })

    return {
        "exists_in_messages": len(results) > 0,
        "message_count": len(results),
        "conversation_ids": sorted(conversation_ids),
        "message_ids": sorted(message_ids),
        "message_previews": message_previews[:5],
    }


def get_conversation_title(db, conversation_id: str) -> Optional[str]:
    """Get conversation title by ID."""
    stmt = select(Conversation.title).where(Conversation.id == conversation_id)
    result = db.execute(stmt).fetchone()
    return result[0] if result else None


def validate_benchmark_phrases(
    db,
    benchmark_queries: List[Dict[str, Any]],
    phrase_key: str = "expected_phrase",
) -> Dict[str, Any]:
    """Validate all benchmark phrases against the corpus.

    Args:
        db: Database session
        benchmark_queries: List of query dicts from benchmark JSON
        phrase_key: Key containing the expected phrase

    Returns:
        Validation report with per-phrase results
    """
    results = []
    valid_count = 0
    invalid_count = 0
    lost_in_chunking = 0

    for i, query_dict in enumerate(benchmark_queries, start=1):
        phrase = query_dict.get(phrase_key, "")
        query = query_dict.get("query", "")
        query_type = query_dict.get("query_type", "")

        if not phrase:
            continue

        print(f"  [{i}/{len(benchmark_queries)}] Checking: '{phrase}'")

        # Search in chunks
        chunk_result = search_phrase_in_chunks(db, phrase)

        # Search in messages
        msg_result = search_phrase_in_messages(db, phrase)

        # Enrich with conversation titles
        convo_titles = {}
        all_convo_ids = set(chunk_result["conversation_ids"]) | set(
            msg_result["conversation_ids"]
        )
        for cid in all_convo_ids:
            title = get_conversation_title(db, cid)
            if title:
                convo_titles[cid] = title

        # Determine status
        if chunk_result["exists_in_chunks"]:
            status = "VALID"
            valid_count += 1
        elif msg_result["exists_in_messages"]:
            status = "LOST_IN_CHUNKING"
            lost_in_chunking += 1
            invalid_count += 1
        else:
            status = "DATA_MISSING"
            invalid_count += 1

        result = {
            "phrase": phrase,
            "query": query,
            "query_type": query_type,
            "status": status,
            "in_chunks": chunk_result,
            "in_messages": msg_result,
            "conversation_titles": convo_titles,
        }
        results.append(result)

        # Print summary
        if status == "VALID":
            print(f"    ✓ Found in {chunk_result['chunk_count']} chunks")
        elif status == "LOST_IN_CHUNKING":
            print(f"    ⚠ In messages but LOST during chunking!")
        else:
            print(f"    ✗ NOT FOUND anywhere in database")

    return {
        "total_phrases": len(results),
        "valid": valid_count,
        "invalid": invalid_count,
        "lost_in_chunking": lost_in_chunking,
        "data_missing": invalid_count - lost_in_chunking,
        "per_phrase": results,
    }


def run_validation(
    memory_queries_path: str = None,
    test_queries_path: str = None,
) -> Dict[str, Any]:
    """Run corpus validation against all benchmark files."""
    print("=" * 70)
    print("PHASE 2 — CORPUS VALIDATION")
    print("=" * 70)

    report = {"timestamp": datetime.now().isoformat()}

    # Load memory queries
    if memory_queries_path is None:
        memory_queries_path = str(
            Path(__file__).parent / "memory_queries.json"
        )

    memory_queries = []
    mq_path = Path(memory_queries_path)
    if mq_path.exists():
        with open(mq_path, "r", encoding="utf-8") as f:
            memory_queries = json.load(f)
        print(f"\n[+] Loaded {len(memory_queries)} memory queries")
    else:
        print(f"[!] Memory queries not found: {memory_queries_path}")

    # Load test queries
    if test_queries_path is None:
        test_queries_path = str(Path(__file__).parent / "test_queries.json")

    test_queries = []
    tq_path = Path(test_queries_path)
    if tq_path.exists():
        with open(tq_path, "r", encoding="utf-8") as f:
            test_queries = json.load(f)
        print(f"[+] Loaded {len(test_queries)} test queries")
    else:
        print(f"[!] Test queries not found: {test_queries_path}")

    db = SessionLocal()
    try:
        # Validate memory queries (phrase-based)
        if memory_queries:
            print(f"\n--- Validating Memory Queries (phrase-based) ---")
            report["memory_queries"] = validate_benchmark_phrases(
                db, memory_queries, phrase_key="expected_phrase"
            )

        # Validate test queries (keyword-based)
        if test_queries:
            print(f"\n--- Validating Test Queries (keyword-based) ---")
            # For keyword-based queries, check each keyword
            keyword_results = []
            for tq in test_queries:
                keywords = tq.get("expected_keywords", [])
                for kw in keywords:
                    kw_entry = {
                        "query": tq["query"],
                        "expected_phrase": kw,
                        "query_type": "keyword",
                    }
                    keyword_results.append(kw_entry)

            # Deduplicate phrases
            seen = set()
            unique_kw = []
            for kr in keyword_results:
                if kr["expected_phrase"].lower() not in seen:
                    seen.add(kr["expected_phrase"].lower())
                    unique_kw.append(kr)

            report["test_queries_keywords"] = validate_benchmark_phrases(
                db, unique_kw, phrase_key="expected_phrase"
            )

        # Print summary
        print("\n" + "=" * 70)
        print("CORPUS VALIDATION SUMMARY")
        print("=" * 70)

        if "memory_queries" in report:
            mq = report["memory_queries"]
            print(f"\n  Memory Queries:")
            print(f"    Total phrases:     {mq['total_phrases']}")
            print(f"    Valid (in chunks): {mq['valid']}")
            print(f"    Lost in chunking: {mq['lost_in_chunking']}")
            print(f"    Data missing:     {mq['data_missing']}")

            if mq["invalid"] > 0:
                print(f"\n  Invalid Benchmark Queries:")
                for p in mq["per_phrase"]:
                    if p["status"] != "VALID":
                        print(f"    ✗ '{p['phrase']}' — {p['status']}")
                        print(f"      Query: {p['query']}")

        if "test_queries_keywords" in report:
            tq = report["test_queries_keywords"]
            print(f"\n  Test Query Keywords:")
            print(f"    Total keywords:    {tq['total_phrases']}")
            print(f"    Valid (in chunks): {tq['valid']}")
            print(f"    Missing:          {tq['data_missing']}")

    finally:
        db.close()

    return report


def main():
    parser = argparse.ArgumentParser(description="Weft Corpus Validation")
    parser.add_argument(
        "--output",
        type=str,
        default="corpus_validation_report.json",
        help="Output report file",
    )
    args = parser.parse_args()

    report = run_validation()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n[+] Report saved to {output_path}")


if __name__ == "__main__":
    main()
