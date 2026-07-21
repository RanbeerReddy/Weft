"""Memory retrieval evaluation pipeline.

Evaluates whether the system retrieves the CORRECT MEMORY (phrase-based)
rather than just topically related content.

Implements:
- STEP 3: MemoryHit@k metrics and MRR
- STEP 4: Failure analysis
- STEP 5: Enhanced diagnostics (conversation title, role, timestamp)
- STEP 6: Candidate recall @10, @20, @50
- STEP 7: Reranking candidates preparation
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from sentence_transformers import SentenceTransformer
from sqlalchemy import select

from Weft.evaluation.core.memory_metrics import (
    MemoryEvaluationSummary,
    MemoryMetricsCalculator,
    MemoryQueryMetrics,
    MemoryRetrievalResult,
)
from Weft.utils.exceptions import WeftException
from Weft.storage.database import SessionLocal
from Weft.storage.models import Chunk, Conversation, Embedding, Message

# Model loaded at module level
MODEL = None


def get_model() -> SentenceTransformer:
    """Lazily load embedding model."""
    global MODEL
    if MODEL is None:
        print("[*] Loading embedding model: BAAI/bge-small-en-v1.5")
        MODEL = SentenceTransformer("BAAI/bge-small-en-v1.5")
    return MODEL


def retrieve_top_50_with_metadata(query: str) -> List[MemoryRetrievalResult]:
    """Retrieve top-50 chunks for a query WITH metadata (conversation, message, timestamp).

    This retrieves more results for candidate recall analysis and includes
    conversation title, message role, and timestamp for diagnostics.

    Args:
        query: Search query string

    Returns:
        List of MemoryRetrievalResult objects with full metadata, ranked by distance
    """
    model = get_model()
    db = SessionLocal()

    try:
        # 1. Encode query
        vector_embedding = model.encode(query, normalize_embeddings=True).tolist()

        # 2. Build query with JOIN for metadata
        distance_attr = Embedding.embedding_vector.cosine_distance(vector_embedding)

        stmt = (
            select(
                Embedding.conversation_id,
                Embedding.message_id,
                Embedding.chunk_order,
                Chunk.chunk_text,
                distance_attr.label("distance"),
                Conversation.title,  # STEP 5: Get conversation title
                Message.role,  # STEP 5: Get message role
                Message.create_time,  # STEP 5: Get timestamp
            )
            .join(Chunk, Chunk.id == Embedding.chunk_order)
            .join(Message, Message.id == Embedding.message_id)
            .join(Conversation, Conversation.id == Embedding.conversation_id)
            .order_by("distance")
            .limit(50)  # STEP 6: Get top-50 for candidate recall
        )

        # 3. Execute
        results = db.execute(stmt).fetchall()

        # 4. Convert to MemoryRetrievalResult objects
        retrieved = []
        for rank, row in enumerate(results, start=1):
            # Format timestamp
            timestamp = None
            if row.create_time:
                try:
                    ts = row.create_time
                    if isinstance(ts, (int, float)):
                        # Unix timestamp - convert to ISO format
                        timestamp = datetime.fromtimestamp(ts).isoformat()
                    else:
                        timestamp = str(ts)
                except Exception:
                    raise WeftException("An error occurred", None)

            retrieved.append(
                MemoryRetrievalResult(
                    rank=rank,
                    distance=float(row.distance),
                    chunk_text=row.chunk_text,
                    conversation_id=row.conversation_id,
                    conversation_title=row.title,  # STEP 5
                    message_id=row.message_id,
                    message_role=row.role,  # STEP 5
                    message_timestamp=timestamp,  # STEP 5
                )
            )

        return retrieved

    except Exception as e:
        raise WeftException(str(e), e) from e
        return []
    finally:
        db.close()


def load_memory_queries(queries_file: str = None) -> List[Dict[str, Any]]:
    """Load memory queries from JSON file.

    Args:
        queries_file: Path to memory_queries.json. If None, uses default location.

    Returns:
        List of query dicts with 'query' and 'expected_phrase'
    """
    if queries_file is None:
        queries_file = Path(__file__).parent / "memory_queries.json"

    queries_file = Path(queries_file)

    if not queries_file.exists():
        print(f"[!] Queries file not found: {queries_file}")
        return []

    try:
        with open(queries_file, "r", encoding="utf-8") as f:
            queries = json.load(f)
        print(f"[+] Loaded {len(queries)} memory queries")
        return queries
    except Exception as e:
        raise WeftException(str(e), e) from e
        return []


def evaluate_all_memory_queries(
    queries: List[Dict[str, Any]], verbose: bool = False, max_results: int = 50
) -> Tuple[MemoryEvaluationSummary, List[Dict]]:
    """Evaluate all queries using memory-based metrics.

    Args:
        queries: List of query dicts with 'query' and 'expected_phrase'
        verbose: If True, print detailed results
        max_results: Number of top results to retrieve and analyze

    Returns:
        Tuple of (MemoryEvaluationSummary, list of reranking candidates)
    """
    query_metrics = []
    reranking_candidates = []

    for i, query_dict in enumerate(queries, start=1):
        query_text = query_dict.get("query", "")
        expected_phrase = query_dict.get("expected_phrase", "")
        query_type = query_dict.get("query_type")

        if not query_text or not expected_phrase:
            print(f"[!] Query {i}: missing 'query' or 'expected_phrase'")
            continue

        print(
            f"[{i}/{len(queries)}] {query_text[:60]}... → expecting: {expected_phrase}"
        )

        # STEP 5: Retrieve with metadata
        retrieved = retrieve_top_50_with_metadata(query_text)

        # Evaluate query
        metrics = MemoryMetricsCalculator.evaluate_memory_query(
            query_text, expected_phrase, retrieved, query_type
        )

        query_metrics.append(metrics)

        # STEP 7: Prepare reranking candidates (top 50 with all metadata)
        reranking_candidate = {
            "query": query_text,
            "expected_phrase": expected_phrase,
            "query_type": query_type,
            "top_50_candidates": [
                {
                    "rank": r.rank,
                    "distance": r.distance,
                    "chunk_text": r.chunk_text[:200],  # Preview only
                    "chunk_text_full": r.chunk_text,  # Full text
                    "conversation_id": r.conversation_id,
                    "conversation_title": r.conversation_title,
                    "message_id": r.message_id,
                    "message_role": r.message_role,
                    "message_timestamp": r.message_timestamp,
                    "phrase_found": MemoryMetricsCalculator.phrase_in_text(
                        expected_phrase, r.chunk_text
                    ),
                }
                for r in retrieved
            ],
        }
        reranking_candidates.append(reranking_candidate)

        # Print results
        if verbose:
            _print_query_details(metrics)

    # Aggregate metrics
    summary = MemoryEvaluationSummary(
        total_queries=len(query_metrics), per_query_metrics=query_metrics
    )

    if query_metrics:
        summary.queries_with_hits_at_1 = sum(
            1 for m in query_metrics if m.memory_hit_at_1
        )
        summary.queries_with_hits_at_3 = sum(
            1 for m in query_metrics if m.memory_hit_at_3
        )
        summary.queries_with_hits_at_5 = sum(
            1 for m in query_metrics if m.memory_hit_at_5
        )
        summary.queries_with_hits_at_10 = sum(
            1 for m in query_metrics if m.memory_hit_at_10
        )

        summary.queries_with_candidate_recall_at_10 = sum(
            1 for m in query_metrics if m.candidate_recall_at_10
        )
        summary.queries_with_candidate_recall_at_20 = sum(
            1 for m in query_metrics if m.candidate_recall_at_20
        )
        summary.queries_with_candidate_recall_at_50 = sum(
            1 for m in query_metrics if m.candidate_recall_at_50
        )

        summary.avg_phrase_mrr = sum(m.phrase_mrr for m in query_metrics) / len(
            query_metrics
        )

        summary.compute_rates()

    return summary, reranking_candidates


def _print_query_details(metrics: MemoryQueryMetrics):
    """Print detailed results for a single query (STEP 4: Failure Analysis)."""
    print(f"\n  Query: {metrics.query}")
    print(f"  Expected phrase: {metrics.expected_phrase}")
    print(f"  Type: {metrics.query_type}")
    print(f"\n  Memory Hits:")
    print(f"    Hit@1:  {metrics.memory_hit_at_1}")
    print(f"    Hit@3:  {metrics.memory_hit_at_3}")
    print(f"    Hit@5:  {metrics.memory_hit_at_5}")
    print(f"    Hit@10: {metrics.memory_hit_at_10}")
    print(f"\n  Candidate Recall:")
    print(f"    Recall@10: {metrics.candidate_recall_at_10}")
    print(f"    Recall@20: {metrics.candidate_recall_at_20}")
    print(f"    Recall@50: {metrics.candidate_recall_at_50}")
    print(f"\n  Phrase MRR: {metrics.phrase_mrr:.4f}")
    print(f"  Phrase found at rank: {metrics.phrase_rank}")

    if metrics.retrieved_chunks:
        print(f"\n  Top 3 Retrieved Chunks:")
        for chunk in metrics.retrieved_chunks[:3]:
            status = (
                "✓ HAS PHRASE"
                if MemoryMetricsCalculator.phrase_in_text(
                    metrics.expected_phrase, chunk.chunk_text
                )
                else "✗ NO PHRASE"
            )

            print(f"\n    Rank {chunk.rank}: Distance={chunk.distance:.4f} [{status}]")
            print(f"    Conversation: {chunk.conversation_title}")
            print(f"    Role: {chunk.message_role} | Time: {chunk.message_timestamp}")
            print(f"    Preview: {chunk.chunk_text[:150]}...")


def format_memory_report(
    summary: MemoryEvaluationSummary, verbose: bool = False
) -> str:
    """Format human-readable memory evaluation report.

    Args:
        summary: MemoryEvaluationSummary object
        verbose: Include per-query details

    Returns:
        Formatted report string
    """
    lines = []
    lines.append("\n" + "=" * 70)
    lines.append("MEMORY RETRIEVAL EVALUATION REPORT")
    lines.append("=" * 70)

    # Summary metrics
    lines.append("\nMEMORY HIT RATES:")
    lines.append(
        f"  Hit@1:  {summary.memory_hit_at_1_rate*100:5.1f}%  ({summary.queries_with_hits_at_1}/{summary.total_queries})"
    )
    lines.append(
        f"  Hit@3:  {summary.memory_hit_at_3_rate*100:5.1f}%  ({summary.queries_with_hits_at_3}/{summary.total_queries})"
    )
    lines.append(
        f"  Hit@5:  {summary.memory_hit_at_5_rate*100:5.1f}%  ({summary.queries_with_hits_at_5}/{summary.total_queries})"
    )
    lines.append(
        f"  Hit@10: {summary.memory_hit_at_10_rate*100:5.1f}%  ({summary.queries_with_hits_at_10}/{summary.total_queries})"
    )

    lines.append("\nCANDIDATE RECALL (phrase found anywhere in range):")
    lines.append(
        f"  Recall@10: {summary.candidate_recall_at_10_rate*100:5.1f}%  ({summary.queries_with_candidate_recall_at_10}/{summary.total_queries})"
    )
    lines.append(
        f"  Recall@20: {summary.candidate_recall_at_20_rate*100:5.1f}%  ({summary.queries_with_candidate_recall_at_20}/{summary.total_queries})"
    )
    lines.append(
        f"  Recall@50: {summary.candidate_recall_at_50_rate*100:5.1f}%  ({summary.queries_with_candidate_recall_at_50}/{summary.total_queries})"
    )

    lines.append(f"\nMEAN RECIPROCAL RANK (MRR): {summary.avg_phrase_mrr:.4f}")

    # Diagnostic interpretation
    lines.append("\n" + "-" * 70)
    lines.append("DIAGNOSTIC INTERPRETATION:")
    lines.append("-" * 70)

    if summary.candidate_recall_at_50_rate < 0.5:
        lines.append(
            "\n[!] CRITICAL: Only {:.0f}% of correct memories found in top-50".format(
                summary.candidate_recall_at_50_rate * 100
            )
        )
        lines.append("    → EMBEDDING MODEL or CHUNKING is failing")
        lines.append(
            "    → Consider: different embedding model, different chunking strategy"
        )
    elif summary.candidate_recall_at_50_rate > 0.9:
        if summary.memory_hit_at_10_rate < 0.7:
            lines.append(
                "\n[!] IMPORTANT: Correct memories found in top-50 but not top-10"
            )
            lines.append("    → RANKING is failing")
            lines.append("    → Consider: reranking, secondary scoring, filtering")
        if summary.memory_hit_at_3_rate < 0.4:
            lines.append(
                "\n[!] IMPORTANT: Correct memories found in top-10 but not top-3"
            )
            lines.append("    → TOP-K RANKING needs improvement")
            lines.append("    → Reranker could significantly help")

    if summary.memory_hit_at_1_rate > 0.7:
        lines.append("\n[✓] GOOD: Most queries retrieve correct memory in first result")

    # Best/Worst queries
    if summary.per_query_metrics:
        lines.append("\n" + "-" * 70)
        lines.append("BEST PERFORMING QUERIES (hit in top-3):")
        lines.append("-" * 70)

        best = [m for m in summary.per_query_metrics if m.memory_hit_at_3][:3]
        for m in best:
            lines.append(f"  • {m.query[:50]}...")
            lines.append(
                f"    Expected: {m.expected_phrase} | Found at rank: {m.phrase_rank} | MRR: {m.phrase_mrr:.4f}"
            )

        lines.append("\n" + "-" * 70)
        lines.append("WORST PERFORMING QUERIES (no hit in top-10):")
        lines.append("-" * 70)

        worst = [m for m in summary.per_query_metrics if not m.memory_hit_at_10][:3]
        if worst:
            for m in worst:
                lines.append(f"  • {m.query[:50]}...")
                lines.append(f"    Expected: {m.expected_phrase}")
                if m.candidate_recall_at_50:
                    lines.append(f"    → Found at rank {m.phrase_rank} (ranking issue)")
                else:
                    lines.append(f"    → NOT in top-50 (embedding/chunking issue)")
        else:
            lines.append("  [✓] All queries retrieved correct memory in top-10")

    lines.append("\n" + "=" * 70)

    return "\n".join(lines)


def format_memory_json_report(summary: MemoryEvaluationSummary) -> Dict:
    """Convert memory evaluation to JSON-serializable dict.

    Args:
        summary: MemoryEvaluationSummary

    Returns:
        Dict ready for JSON serialization
    """
    return {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total_queries": summary.total_queries,
            "memory_hit_at_1_rate": summary.memory_hit_at_1_rate,
            "memory_hit_at_3_rate": summary.memory_hit_at_3_rate,
            "memory_hit_at_5_rate": summary.memory_hit_at_5_rate,
            "memory_hit_at_10_rate": summary.memory_hit_at_10_rate,
            "candidate_recall_at_10_rate": summary.candidate_recall_at_10_rate,
            "candidate_recall_at_20_rate": summary.candidate_recall_at_20_rate,
            "candidate_recall_at_50_rate": summary.candidate_recall_at_50_rate,
            "avg_phrase_mrr": summary.avg_phrase_mrr,
        },
        "per_query": [
            {
                "query": m.query,
                "expected_phrase": m.expected_phrase,
                "query_type": m.query_type,
                "memory_hit_at_1": m.memory_hit_at_1,
                "memory_hit_at_3": m.memory_hit_at_3,
                "memory_hit_at_5": m.memory_hit_at_5,
                "memory_hit_at_10": m.memory_hit_at_10,
                "candidate_recall_at_10": m.candidate_recall_at_10,
                "candidate_recall_at_20": m.candidate_recall_at_20,
                "candidate_recall_at_50": m.candidate_recall_at_50,
                "phrase_mrr": m.phrase_mrr,
                "phrase_rank": m.phrase_rank,
            }
            for m in (summary.per_query_metrics or [])
        ],
    }


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Memory Retrieval Evaluation")
    parser.add_argument(
        "--queries",
        type=str,
        default=None,
        help="Path to memory_queries.json file (default: Weft/evaluation/memory_queries.json)",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print detailed per-query results"
    )
    parser.add_argument(
        "--json-output", type=str, default=None, help="Save JSON report to file"
    )
    parser.add_argument(
        "--reranking-output",
        type=str,
        default=None,
        help="Save reranking candidates to file (default: reranking_candidates.json)",
    )

    args = parser.parse_args()

    # Load queries
    queries = load_memory_queries(args.queries)
    if not queries:
        print("[!] No queries loaded, exiting")
        sys.exit(1)

    # Evaluate
    print("\n[*] Starting memory retrieval evaluation...")
    t0 = time.time()

    summary, reranking_candidates = evaluate_all_memory_queries(
        queries, verbose=args.verbose
    )

    elapsed = time.time() - t0
    print(f"\n[+] Evaluation complete in {elapsed:.2f}s")

    # Print report
    report = format_memory_report(summary, verbose=args.verbose)
    print(report)

    # Save JSON report if requested
    if args.json_output:
        json_report = format_memory_json_report(summary)
        output_path = Path(args.json_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(json_report, f, indent=2)
        print(f"\n[+] JSON report saved to {output_path}")

    # Save reranking candidates if requested (or default location)
    reranking_path = args.reranking_output or "reranking_candidates.json"
    reranking_path = Path(reranking_path)
    reranking_path.parent.mkdir(parents=True, exist_ok=True)

    with open(reranking_path, "w", encoding="utf-8") as f:
        json.dump(reranking_candidates, f, indent=2)
    print(f"[+] Reranking candidates saved to {reranking_path}")


if __name__ == "__main__":
    main()
