"""Phase 5 — Chunk Analysis.

For every failed retrieval where data exists:

    Display:
        Original message → Generated chunks → Stored chunks → Retrieved chunks

    Determine whether chunking destroyed context:
        - Phrase split across chunks
        - Important sentence removed
        - Metadata lost
        - Very short or very long chunks
        - Context isolation
"""

import json
import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

from sqlalchemy import select, func

from Weft.storage.database import SessionLocal
from Weft.storage.models import Message, Chunk, Conversation
from Weft.evaluation.core.memory_metrics import MemoryMetricsCalculator

# Import the splitter to simulate chunk generation
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Same splitter config as production
SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    length_function=len,
)


def find_messages_containing_phrase(db, phrase: str) -> List[Dict[str, Any]]:
    """Find all messages that contain the phrase."""
    stmt = (
        select(
            Message.id,
            Message.conversation_id,
            Message.role,
            Message.content,
            Message.create_time,
            Conversation.title,
        )
        .join(Conversation, Conversation.id == Message.conversation_id)
        .where(func.lower(Message.content).contains(phrase.lower()))
    )
    results = db.execute(stmt).fetchall()
    return [
        {
            "message_id": r.id,
            "conversation_id": r.conversation_id,
            "role": r.role,
            "content": r.content,
            "create_time": str(r.create_time) if r.create_time else None,
            "conversation_title": r.title,
        }
        for r in results
    ]


def get_stored_chunks_for_message(db, message_id: str) -> List[Dict[str, Any]]:
    """Get all chunks stored in DB for a message."""
    stmt = (
        select(Chunk.id, Chunk.chunk_order, Chunk.chunk_text)
        .where(Chunk.message_id == message_id)
        .order_by(Chunk.chunk_order)
    )
    results = db.execute(stmt).fetchall()
    return [
        {
            "chunk_id": r.id,
            "chunk_order": r.chunk_order,
            "chunk_text": r.chunk_text,
            "length": len(r.chunk_text) if r.chunk_text else 0,
        }
        for r in results
    ]


def simulate_chunking(content: str) -> List[str]:
    """Re-run the splitter on the content to see what chunks WOULD be generated."""
    if not content:
        return []
    return SPLITTER.split_text(content)


def analyze_chunking_for_phrase(
    db, phrase: str, query: str, query_type: str = None
) -> Dict[str, Any]:
    """Analyze how chunking affects a specific phrase.

    Args:
        db: Database session
        phrase: The expected phrase
        query: The original query
        query_type: Optional type label

    Returns:
        Detailed chunking analysis
    """
    result = {
        "query": query,
        "expected_phrase": phrase,
        "query_type": query_type,
    }

    # 1. Find messages containing the phrase
    messages = find_messages_containing_phrase(db, phrase)
    if not messages:
        result["status"] = "PHRASE_NOT_IN_MESSAGES"
        result["explanation"] = f"Phrase '{phrase}' not found in any message"
        return result

    result["messages_with_phrase"] = len(messages)
    result["message_analyses"] = []

    for msg in messages:
        msg_analysis = {
            "message_id": msg["message_id"],
            "conversation_id": msg["conversation_id"],
            "conversation_title": msg["conversation_title"],
            "role": msg["role"],
            "message_length": len(msg["content"]) if msg["content"] else 0,
        }

        content = msg["content"] or ""

        # 2. Simulate chunking
        simulated_chunks = simulate_chunking(content)
        msg_analysis["simulated_chunk_count"] = len(simulated_chunks)

        # Check which simulated chunks contain the phrase
        sim_containing = []
        sim_split = False
        for idx, sc in enumerate(simulated_chunks):
            contains = MemoryMetricsCalculator.phrase_in_text(phrase, sc)
            sim_containing.append({
                "chunk_index": idx,
                "contains_phrase": contains,
                "length": len(sc),
                "preview": sc[:100],
            })

        phrase_in_any_sim = any(s["contains_phrase"] for s in sim_containing)

        # Check if phrase is split across chunk boundaries
        if not phrase_in_any_sim and phrase.lower() in content.lower():
            sim_split = True

        msg_analysis["simulated_chunks"] = sim_containing
        msg_analysis["phrase_in_simulated_chunks"] = phrase_in_any_sim
        msg_analysis["phrase_split_across_chunks"] = sim_split

        # 3. Get stored chunks from DB
        stored_chunks = get_stored_chunks_for_message(db, msg["message_id"])
        msg_analysis["stored_chunk_count"] = len(stored_chunks)

        stored_containing = []
        for sc in stored_chunks:
            contains = MemoryMetricsCalculator.phrase_in_text(
                phrase, sc["chunk_text"]
            )
            stored_containing.append({
                "chunk_id": sc["chunk_id"],
                "chunk_order": sc["chunk_order"],
                "contains_phrase": contains,
                "length": sc["length"],
                "preview": sc["chunk_text"][:100],
            })

        phrase_in_stored = any(s["contains_phrase"] for s in stored_containing)
        msg_analysis["stored_chunks"] = stored_containing
        msg_analysis["phrase_in_stored_chunks"] = phrase_in_stored

        # 4. Detect issues
        issues = []

        if sim_split:
            issues.append("PHRASE_SPLIT_ACROSS_CHUNKS")

        if not phrase_in_stored and phrase_in_any_sim:
            issues.append("PHRASE_IN_SIMULATED_BUT_NOT_STORED")

        if not phrase_in_stored and not phrase_in_any_sim and phrase.lower() in content.lower():
            issues.append("PHRASE_IN_MESSAGE_BUT_NOT_ANY_CHUNK")

        if len(stored_chunks) == 0 and content.strip():
            issues.append("MESSAGE_HAS_CONTENT_BUT_NO_CHUNKS")

        # Check for very short chunks (< 50 chars) that lack context
        short_chunks = [s for s in stored_containing if s["length"] < 50]
        if short_chunks:
            issues.append(f"SHORT_CHUNKS_LACKING_CONTEXT ({len(short_chunks)} chunks < 50 chars)")

        # Check if message is so long it produces many chunks
        if len(stored_chunks) > 10:
            issues.append(f"CHUNK_EXPLOSION ({len(stored_chunks)} chunks from one message)")

        # Check if there are no stored chunks but simulated chunks exist
        if len(stored_chunks) != len(simulated_chunks):
            issues.append(
                f"CHUNK_COUNT_MISMATCH (stored={len(stored_chunks)}, "
                f"simulated={len(simulated_chunks)})"
            )

        msg_analysis["issues"] = issues
        result["message_analyses"].append(msg_analysis)

    # Aggregate issues
    all_issues = []
    for ma in result["message_analyses"]:
        all_issues.extend(ma["issues"])

    result["total_issues"] = len(all_issues)
    result["unique_issues"] = list(set(all_issues))
    result["status"] = "ANALYZED"

    return result


def run_chunk_analysis(
    failure_report_path: str = None,
    memory_queries_path: str = None,
) -> Dict[str, Any]:
    """Run chunk analysis for all relevant queries."""
    print("=" * 70)
    print("PHASE 5 — CHUNK ANALYSIS")
    print("=" * 70)

    report = {"timestamp": datetime.now().isoformat()}

    # Determine which queries to analyze
    queries_to_analyze = []

    if failure_report_path is None:
        failure_report_path = "failure_analysis_report.json"

    fa_path = Path(failure_report_path)
    if fa_path.exists():
        with open(fa_path, "r", encoding="utf-8") as f:
            fa_data = json.load(f)
        # Analyze ALL queries, not just failures, to understand chunking patterns
        for c in fa_data.get("classifications", []):
            queries_to_analyze.append(c)
        print(f"[+] Analyzing chunking for {len(queries_to_analyze)} queries")
    else:
        print("[!] No failure report found — loading all memory queries")
        if memory_queries_path is None:
            memory_queries_path = str(Path(__file__).parent / "memory_queries.json")
        with open(memory_queries_path, "r", encoding="utf-8") as f:
            queries = json.load(f)
        for q in queries:
            queries_to_analyze.append({
                "query": q["query"],
                "expected_phrase": q["expected_phrase"],
                "query_type": q.get("query_type"),
                "category": "UNKNOWN",
            })

    db = SessionLocal()
    try:
        analyses = []
        issue_summary = {}

        for i, q in enumerate(queries_to_analyze, start=1):
            query_text = q.get("query", "")
            expected = q.get("expected_phrase", "")
            qtype = q.get("query_type", "")
            category = q.get("category", "?")

            print(f"\n[{i}/{len(queries_to_analyze)}] [{category}] {query_text}")
            print(f"  Phrase: {expected}")

            analysis = analyze_chunking_for_phrase(db, expected, query_text, qtype)
            analysis["failure_category"] = category
            analyses.append(analysis)

            for issue in analysis.get("unique_issues", []):
                base_issue = issue.split(" (")[0]  # remove counts
                issue_summary[base_issue] = issue_summary.get(base_issue, 0) + 1

            if analysis.get("unique_issues"):
                for issue in analysis["unique_issues"]:
                    print(f"  ⚠ {issue}")
            elif analysis["status"] == "ANALYZED":
                print(f"  ✓ No chunking issues detected")

        report["analyses"] = analyses
        report["issue_summary"] = issue_summary

        # Overall chunk statistics
        print("\n[*] Computing overall chunk statistics...")
        total_chunks = db.scalar(select(func.count(Chunk.id)))
        avg_length = db.scalar(select(func.avg(func.length(Chunk.chunk_text))))
        min_length = db.scalar(select(func.min(func.length(Chunk.chunk_text))))
        max_length = db.scalar(select(func.max(func.length(Chunk.chunk_text))))

        report["chunk_statistics"] = {
            "total_chunks": total_chunks,
            "avg_chunk_length": round(float(avg_length), 1) if avg_length else 0,
            "min_chunk_length": min_length,
            "max_chunk_length": max_length,
        }

        # Print summary
        print("\n" + "=" * 70)
        print("CHUNK ANALYSIS SUMMARY")
        print("=" * 70)
        print(f"\n  Total queries analyzed: {len(analyses)}")
        print(f"\n  Chunk Statistics:")
        print(f"    Total chunks:     {total_chunks}")
        print(f"    Avg chunk length: {report['chunk_statistics']['avg_chunk_length']}")
        print(f"    Min chunk length: {min_length}")
        print(f"    Max chunk length: {max_length}")

        if issue_summary:
            print(f"\n  Chunking Issues Found:")
            for issue, count in sorted(issue_summary.items(), key=lambda x: -x[1]):
                print(f"    {issue}: {count}")
        else:
            print(f"\n  ✓ No chunking issues detected")

    finally:
        db.close()

    return report


def main():
    parser = argparse.ArgumentParser(description="Weft Chunk Analysis")
    parser.add_argument(
        "--output",
        type=str,
        default="chunk_analysis_report.json",
        help="Output report file",
    )
    args = parser.parse_args()

    report = run_chunk_analysis()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n[+] Report saved to {output_path}")


if __name__ == "__main__":
    main()
