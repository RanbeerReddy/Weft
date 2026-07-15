"""Phase 3 — Retrieval Failure Analysis.

For every failed benchmark query, classifies the failure into exactly
one category:

    Category A — Data Missing
        The expected answer never entered the database.

    Category B — Embedding Miss
        Data exists in chunks but embedding search never retrieves it
        (not found even in top-100).

    Category C — Ranking Issue
        Correct chunk is retrieved but ranked poorly (found in top-100
        but not in top-10).

    Category D — Eval Mismatch
        Correct chunk is retrieved and ranked well (top-10) but the
        benchmark evaluation incorrectly reports it as a failure.

Generates per-query classification and aggregate statistics.
"""

import json
import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

from sentence_transformers import SentenceTransformer
from sqlalchemy import select, func

from Weft.storage.database import SessionLocal
from Weft.storage.models import Embedding, Chunk, Message, Conversation
from Weft.evaluation.memory_metrics import MemoryMetricsCalculator


MODEL = None


def get_model() -> SentenceTransformer:
    global MODEL
    if MODEL is None:
        print("[*] Loading embedding model: BAAI/bge-small-en-v1.5")
        MODEL = SentenceTransformer("BAAI/bge-small-en-v1.5")
    return MODEL


def phrase_exists_in_db(db, phrase: str) -> bool:
    """Check if phrase exists anywhere in chunks table."""
    stmt = (
        select(func.count(Chunk.id))
        .where(func.lower(Chunk.chunk_text).contains(phrase.lower()))
    )
    count = db.scalar(stmt)
    return count > 0


def retrieve_top_k(db, query: str, k: int = 100) -> List[Dict[str, Any]]:
    """Retrieve top-k results for a query."""
    model = get_model()
    vector = model.encode(query, normalize_embeddings=True).tolist()
    distance_attr = Embedding.embedding_vector.cosine_distance(vector)

    stmt = (
        select(
            Embedding.conversation_id,
            Embedding.message_id,
            Embedding.chunk_order,
            Chunk.chunk_text,
            distance_attr.label("distance"),
            Conversation.title,
            Message.role,
        )
        .join(Chunk, Chunk.id == Embedding.chunk_order)
        .join(Message, Message.id == Embedding.message_id)
        .join(Conversation, Conversation.id == Embedding.conversation_id)
        .order_by("distance")
        .limit(k)
    )

    results = db.execute(stmt).fetchall()

    return [
        {
            "rank": rank,
            "distance": float(row.distance),
            "chunk_text": row.chunk_text,
            "conversation_id": row.conversation_id,
            "conversation_title": row.title,
            "message_id": row.message_id,
            "role": row.role,
        }
        for rank, row in enumerate(results, start=1)
    ]


def classify_failure(
    db, query: str, expected_phrase: str, query_type: str = None
) -> Dict[str, Any]:
    """Classify a query failure into category A, B, C, or D.

    Args:
        db: Database session
        query: The search query
        expected_phrase: The phrase expected in results
        query_type: Optional type label

    Returns:
        Classification result dict
    """
    result = {
        "query": query,
        "expected_phrase": expected_phrase,
        "query_type": query_type,
    }

    # Step 1: Does the phrase exist in the database at all?
    exists = phrase_exists_in_db(db, expected_phrase)
    result["phrase_exists_in_db"] = exists

    if not exists:
        result["category"] = "A"
        result["category_label"] = "Data Missing"
        result["explanation"] = (
            f"Phrase '{expected_phrase}' does not exist anywhere in the "
            f"chunks table. The expected data was never ingested."
        )
        return result

    # Step 2: Retrieve top-100 and check if phrase appears
    retrieved = retrieve_top_k(db, query, k=100)
    result["top_100_count"] = len(retrieved)

    # Find where phrase appears in results
    phrase_rank = None
    for r in retrieved:
        if MemoryMetricsCalculator.phrase_in_text(expected_phrase, r["chunk_text"]):
            phrase_rank = r["rank"]
            break

    result["phrase_rank_in_top_100"] = phrase_rank

    if phrase_rank is None:
        result["category"] = "B"
        result["category_label"] = "Embedding Miss"
        result["explanation"] = (
            f"Phrase '{expected_phrase}' exists in the database but was "
            f"not found in the top-100 vector search results. The embedding "
            f"model fails to associate this query with the correct chunk."
        )
        # Add what WAS retrieved
        result["top_3_retrieved"] = [
            {
                "rank": r["rank"],
                "distance": r["distance"],
                "preview": r["chunk_text"][:100],
                "conversation_title": r["conversation_title"],
            }
            for r in retrieved[:3]
        ]
        return result

    if phrase_rank > 10:
        result["category"] = "C"
        result["category_label"] = "Ranking Issue"
        result["explanation"] = (
            f"Phrase '{expected_phrase}' found at rank {phrase_rank} — "
            f"exists in top-100 but not in top-10. The embedding is somewhat "
            f"related but ranking is poor."
        )
        result["top_3_retrieved"] = [
            {
                "rank": r["rank"],
                "distance": r["distance"],
                "preview": r["chunk_text"][:100],
                "conversation_title": r["conversation_title"],
            }
            for r in retrieved[:3]
        ]
        # Find the correct chunk details
        for r in retrieved:
            if r["rank"] == phrase_rank:
                result["correct_chunk"] = {
                    "rank": r["rank"],
                    "distance": r["distance"],
                    "preview": r["chunk_text"][:150],
                }
                break
        return result

    # Phrase is in top-10, so it's a possible eval mismatch
    result["category"] = "D"
    result["category_label"] = "Eval Mismatch"
    result["explanation"] = (
        f"Phrase '{expected_phrase}' found at rank {phrase_rank} (within "
        f"top-10). The benchmark may have incorrectly evaluated this as a "
        f"failure, or a previous run had different data."
    )
    return result


def run_failure_analysis(
    memory_queries_path: str = None,
    eval_results_path: str = None,
) -> Dict[str, Any]:
    """Run failure analysis on all benchmark queries."""
    print("=" * 70)
    print("PHASE 3 — RETRIEVAL FAILURE ANALYSIS")
    print("=" * 70)

    report = {"timestamp": datetime.now().isoformat()}

    # Load memory queries
    if memory_queries_path is None:
        memory_queries_path = str(Path(__file__).parent / "memory_queries.json")

    with open(memory_queries_path, "r", encoding="utf-8") as f:
        queries = json.load(f)
    print(f"\n[+] Loaded {len(queries)} memory queries")

    # Load previous eval results to identify failures
    if eval_results_path is None:
        eval_results_path = "memory_eval_results.json"

    failed_queries = set()
    eval_path = Path(eval_results_path)
    if eval_path.exists():
        with open(eval_path, "r", encoding="utf-8") as f:
            eval_data = json.load(f)
        for pq in eval_data.get("per_query", []):
            if not pq.get("memory_hit_at_10", False):
                failed_queries.add(pq["query"])
        print(f"[+] Identified {len(failed_queries)} failed queries from previous eval")
    else:
        print("[!] No previous eval results found — analyzing ALL queries")
        failed_queries = {q["query"] for q in queries}

    db = SessionLocal()
    try:
        classifications = []
        category_counts = {"A": 0, "B": 0, "C": 0, "D": 0}

        # Classify ALL queries (not just failed ones)
        for i, q in enumerate(queries, start=1):
            query_text = q.get("query", "")
            expected = q.get("expected_phrase", "")
            qtype = q.get("query_type", "")
            is_failed = query_text in failed_queries

            print(f"\n[{i}/{len(queries)}] {'FAILED' if is_failed else 'PASSED'}: {query_text}")
            print(f"  Expected: {expected}")

            classification = classify_failure(db, query_text, expected, qtype)
            classification["was_previously_failed"] = is_failed
            classifications.append(classification)

            cat = classification["category"]
            if is_failed:
                category_counts[cat] += 1
            print(f"  Category: {cat} — {classification['category_label']}")

        report["classifications"] = classifications
        report["failed_query_categories"] = category_counts
        report["total_failed"] = sum(category_counts.values())

        # Print summary
        print("\n" + "=" * 70)
        print("FAILURE ANALYSIS SUMMARY")
        print("=" * 70)
        print(f"\n  Total queries analyzed: {len(queries)}")
        print(f"  Previously failed:     {len(failed_queries)}")
        print(f"\n  Failure Categories (for previously failed queries):")
        print(f"    A — Data Missing:    {category_counts['A']}")
        print(f"    B — Embedding Miss:  {category_counts['B']}")
        print(f"    C — Ranking Issue:   {category_counts['C']}")
        print(f"    D — Eval Mismatch:   {category_counts['D']}")

        if category_counts["A"] > 0:
            print(f"\n  ⚠  {category_counts['A']} failures are INVALID benchmarks "
                  f"(data never existed)")
            print(f"     These queries should be excluded or the data should be ingested")

    finally:
        db.close()

    return report


def main():
    parser = argparse.ArgumentParser(description="Weft Failure Analysis")
    parser.add_argument(
        "--output",
        type=str,
        default="failure_analysis_report.json",
        help="Output report file",
    )
    args = parser.parse_args()

    report = run_failure_analysis()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n[+] Report saved to {output_path}")


if __name__ == "__main__":
    main()
