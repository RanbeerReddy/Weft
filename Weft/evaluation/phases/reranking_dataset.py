"""Phase 7 — Reranking Readiness.

Does NOT implement reranking.

Instead, prepares the infrastructure by storing for each benchmark query:
    - query
    - top 100 retrieved chunks (with text, score, metadata)
    - correct chunk (if known from corpus validation)
    - correct chunk rank in current vector search
    - distance comparisons

This dataset will later be used to evaluate multiple rerankers
(e.g., cross-encoder/ms-marco-MiniLM-L-6-v2, BAAI/bge-reranker-base).
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from sentence_transformers import SentenceTransformer
from sqlalchemy import func, select

from Weft.evaluation.core.memory_metrics import MemoryMetricsCalculator
from Weft.storage.database import SessionLocal
from Weft.storage.models import Chunk, Conversation, Embedding, Message

MODEL: Optional[SentenceTransformer] = None


def get_model() -> SentenceTransformer:
    global MODEL
    if MODEL is None:
        print("[*] Loading embedding model: BAAI/bge-small-en-v1.5")
        MODEL = SentenceTransformer("BAAI/bge-small-en-v1.5")
    return MODEL


def retrieve_top_100_with_metadata(db, query: str) -> List[Dict[str, Any]]:
    """Retrieve top-100 chunks with full metadata for reranking preparation."""
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
            Message.create_time,
        )
        .join(Chunk, Chunk.id == Embedding.chunk_order)
        .join(Message, Message.id == Embedding.message_id)
        .join(Conversation, Conversation.id == Embedding.conversation_id)
        .order_by("distance")
        .limit(100)
    )

    results = db.execute(stmt).fetchall()
    return [
        {
            "rank": rank,
            "distance": round(float(row.distance), 6),
            "chunk_text": row.chunk_text,
            "chunk_id": row.chunk_order,
            "conversation_id": row.conversation_id,
            "conversation_title": row.title,
            "message_id": row.message_id,
            "message_role": row.role,
            "message_timestamp": str(row.create_time) if row.create_time else None,
        }
        for rank, row in enumerate(results, start=1)
    ]


def find_correct_chunk_info(db, phrase: str) -> Optional[Dict[str, Any]]:
    """Find the best correct chunk (containing the phrase) with metadata."""
    stmt = (
        select(
            Chunk.id,
            Chunk.conversation_id,
            Chunk.message_id,
            Chunk.chunk_text,
            Conversation.title,
            Message.role,
        )
        .join(Message, Message.id == Chunk.message_id)
        .join(Conversation, Conversation.id == Chunk.conversation_id)
        .where(func.lower(Chunk.chunk_text).contains(phrase.lower()))
    )
    results = db.execute(stmt).fetchall()
    if not results:
        return None

    # Pick the one with shortest text (most specific match)
    best = min(results, key=lambda r: len(r.chunk_text))
    return {
        "chunk_id": best.id,
        "conversation_id": best.conversation_id,
        "message_id": best.message_id,
        "chunk_text": best.chunk_text,
        "conversation_title": best.title,
        "message_role": best.role,
        "total_matching_chunks": len(results),
    }


def run_reranking_preparation(  # noqa: C901
    memory_queries_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Prepare reranking dataset for all benchmark queries."""
    print("=" * 70)
    print("PHASE 7 — RERANKING READINESS")
    print("=" * 70)

    report: Dict[str, Any] = {"timestamp": datetime.now().isoformat()}

    # Load queries
    if memory_queries_path is None:
        memory_queries_path = str(
            Path(__file__).parent.parent / "data" / "memory_queries.json"
        )

    with open(
        "Weft/evaluation/reports/search_experiments_report.json", "r", encoding="utf-8"
    ) as f:
        queries = json.load(f)
    print(f"[+] Loaded {len(queries)} benchmark queries")

    db = SessionLocal()
    try:
        candidates = []
        stats = {
            "total_queries": len(queries),
            "queries_with_correct_chunk": 0,
            "correct_in_top_10": 0,
            "correct_in_top_50": 0,
            "correct_in_top_100": 0,
            "correct_not_found": 0,
        }

        for i, q in enumerate(queries, start=1):
            query_text = q.get("query", "")
            expected = q.get("expected_phrase", "")
            qtype = q.get("query_type", "")

            print(f"[{i}/{len(queries)}] {query_text}")

            # Get top 100
            top_100 = retrieve_top_100_with_metadata(db, query_text)

            # Find correct chunk
            correct = find_correct_chunk_info(db, expected)

            # Determine correct chunk rank in retrieval
            correct_rank = None
            correct_distance = None
            if correct:
                stats["queries_with_correct_chunk"] += 1
                for r in top_100:
                    if MemoryMetricsCalculator.phrase_in_text(
                        expected, r["chunk_text"]
                    ):
                        correct_rank = r["rank"]
                        correct_distance = r["distance"]
                        break

                if correct_rank and correct_rank <= 10:
                    stats["correct_in_top_10"] += 1
                if correct_rank and correct_rank <= 50:
                    stats["correct_in_top_50"] += 1
                if correct_rank and correct_rank <= 100:
                    stats["correct_in_top_100"] += 1
                if correct_rank is None:
                    stats["correct_not_found"] += 1

            # Mark phrase presence in each candidate
            for r in top_100:
                r["phrase_found"] = MemoryMetricsCalculator.phrase_in_text(
                    expected, r["chunk_text"]
                )

            candidate = {
                "query": query_text,
                "expected_phrase": expected,
                "query_type": qtype,
                "correct_chunk": correct,
                "correct_chunk_rank": correct_rank,
                "correct_chunk_distance": correct_distance,
                "top_1_distance": top_100[0]["distance"] if top_100 else None,
                "distance_gap": (
                    round(correct_distance - top_100[0]["distance"], 6)
                    if correct_distance and top_100
                    else None
                ),
                "top_100_candidates": top_100,
            }
            candidates.append(candidate)

            if correct_rank:
                print(
                    f"  Correct chunk at rank {correct_rank} (dist={correct_distance:.4f})"  # noqa: E501
                )
            elif correct:
                print("  Correct chunk exists but NOT in top-100")
            else:
                print("  No correct chunk found in DB")

        report["candidates"] = candidates
        report["statistics"] = stats

        # Print summary
        print("\n" + "=" * 70)
        print("RERANKING READINESS SUMMARY")
        print("=" * 70)
        print(f"\n  Total queries:              {stats['total_queries']}")
        print(f"  Queries with correct chunk: {stats['queries_with_correct_chunk']}")
        print(f"  Correct in top-10:          {stats['correct_in_top_10']}")
        print(f"  Correct in top-50:          {stats['correct_in_top_50']}")
        print(f"  Correct in top-100:         {stats['correct_in_top_100']}")
        print(f"  Correct not in top-100:     {stats['correct_not_found']}")

        if stats["queries_with_correct_chunk"] > 0:
            reranking_potential = (
                stats["correct_in_top_100"] - stats["correct_in_top_10"]
            )
            print(
                f"\n  Reranking potential: {reranking_potential} queries could improve"
            )
            print("  (chunks in top-100 but not top-10 → reranker could promote them)")

    finally:
        db.close()

    return report


def main():
    parser = argparse.ArgumentParser(description="Weft Reranking Dataset Preparation")
    parser.add_argument(
        "--output",
        type=str,
        default="Weft/evaluation/data/reranking_dataset.json",
        help="Output dataset file",
    )
    args = parser.parse_args()

    report = run_reranking_preparation()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n[+] Dataset saved to {output_path}")


if __name__ == "__main__":
    main()
