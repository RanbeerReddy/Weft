"""Memory Retrieval Evaluation Pipeline.

Evaluates the actual memory retrieval engine using ground-truth IDs.
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from sentence_transformers import SentenceTransformer
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from Weft.config.settings import settings
from Weft.evaluation.core.memory_metrics import (
    MemoryEvaluationSummary,
    MemoryMetricsCalculator,
    MemoryRetrievalResult,
)
from Weft.storage.database import SessionLocal
from Weft.storage.models import Memory, MemoryType
from Weft.utils.exceptions import WeftException
from Weft.utils.logger import logger

# Initialize vector retriever model at module level
MODEL = None


def get_model() -> SentenceTransformer:
    """Lazily load SentenceTransformer."""
    global MODEL
    if MODEL is None:
        MODEL = SentenceTransformer(settings.EMBEDDING_MODEL)
    return MODEL


def retrieve_top_k_memories(
    db: Session, query: str, k: int = 50
) -> List[MemoryRetrievalResult]:
    """Retrieve top-k memories for a query exactly as the engine does.

    Args:
        db: Database session.
        query: Search query string.
        k: Max results to return for evaluation (typically we need more than engine's 5).

    Returns:
        List of MemoryRetrievalResult objects, ranked by distance.
    """
    model = get_model()
    query_embedding = model.encode(query, normalize_embeddings=True).tolist()

    # Query exactly like context_assembler / search_memories but with higher limit
    # and we select the distance to return it
    results = db.execute(
        select(
            Memory,
            Memory.embedding_vector.l2_distance(query_embedding).label("dist"),
        )
        .where(Memory.status == "active")
        .where(Memory.embedding_vector.is_not(None))
        .order_by("dist")
        .limit(k)
    ).all()

    retrieved = []
    seen_ids = set()
    rank = 1

    for row in results:
        mem, dist = row[0], row[1]

        if mem.id in seen_ids:
            continue

        # Weft threshold for memory retrieval is 1.0 (L2 distance)
        # But for evaluation, we want to see ranking even if > 1.0,
        # but we flag it if it wouldn't be a hit.
        # Actually, let's just return all of them so we can compute MRR and candidate recall.

        # Get metadata
        convo_id = mem.conversation_id if hasattr(mem, "conversation_id") else None
        msg_id = mem.message_id if hasattr(mem, "message_id") else None

        retrieved.append(
            MemoryRetrievalResult(
                rank=rank,
                distance=dist,
                chunk_text=mem.value,
                memory_id=mem.id,
                conversation_id=convo_id or "unknown",
                message_id=msg_id,
            )
        )
        seen_ids.add(mem.id)
        rank += 1

    return retrieved


def load_fixtures(db: Session, fixtures_file: Optional[str] = None):
    """Load fixtures into the database."""
    resolved_path: Path
    if fixtures_file is None:
        resolved_path = Path(__file__).parent.parent / "data" / "memory_fixtures.json"
    else:
        resolved_path = Path(fixtures_file)

    if not resolved_path.exists():
        logger.error(f"Fixtures file not found: {resolved_path}")
        return []

    try:
        with open(resolved_path, "r", encoding="utf-8") as f:
            fixtures = json.load(f)

        logger.info(f"Loading {len(fixtures)} fixtures into DB")
        model = get_model()

        # Ensure memory types exist
        types = set(f.get("type_name", "Experience") for f in fixtures)
        for t in types:
            if not db.scalars(select(MemoryType).where(MemoryType.name == t)).first():
                db.add(MemoryType(id=f"type-{t.lower()}", name=t, description=t))
        db.flush()

        type_map = {t.name: t.id for t in db.scalars(select(MemoryType)).all()}

        inserted_ids = []
        for fixture in fixtures:
            # Delete if exists
            db.execute(delete(Memory).where(Memory.id == fixture["id"]))

            # Create embedding
            embedding = model.encode(
                fixture["value"], normalize_embeddings=True
            ).tolist()

            mem = Memory(
                id=fixture["id"],
                type_id=type_map.get(fixture.get("type_name", "Experience")),
                value=fixture["value"],
                status="active",
                embedding_vector=embedding,
            )
            # Add optional fields if model supports them
            if hasattr(mem, "conversation_id") and "conversation_id" in fixture:
                mem.conversation_id = fixture["conversation_id"]
            if hasattr(mem, "message_id") and "message_id" in fixture:
                mem.message_id = fixture["message_id"]

            db.add(mem)
            inserted_ids.append(fixture["id"])

        db.commit()
        return inserted_ids

    except Exception as e:
        db.rollback()
        raise WeftException(str(e), e) from e


def cleanup_fixtures(db: Session, inserted_ids: List[str]):
    """Remove fixtures from the database."""
    if not inserted_ids:
        return
    try:
        db.execute(delete(Memory).where(Memory.id.in_(inserted_ids)))
        db.commit()
        logger.info(f"Cleaned up {len(inserted_ids)} fixtures")
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to cleanup fixtures: {e}")


def load_queries(queries_file: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load benchmark queries."""
    resolved_path: Path
    if queries_file is None:
        resolved_path = (
            Path(__file__).parent.parent / "data" / "memory_benchmark_queries.json"
        )
    else:
        resolved_path = Path(queries_file)

    if not resolved_path.exists():
        logger.error(f"Queries file not found: {resolved_path}")
        return []

    try:
        with open(resolved_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        raise WeftException(str(e), e) from e


def evaluate_all_queries(
    db: Session, queries: List[Dict[str, Any]]
) -> MemoryEvaluationSummary:
    """Evaluate all queries and compute ID-based metrics."""
    query_metrics = []

    for i, query_dict in enumerate(queries, start=1):
        query_text = query_dict.get("query", "")
        expected_id = query_dict.get("expected_id")
        qtype = query_dict.get("query_type", "exact_match")

        if not query_text or not expected_id:
            logger.warning(f"Query {i}: missing 'query' or 'expected_id'")
            continue

        logger.info(
            f"[{i}/{len(queries)}] Evaluating: {query_text[:60]}... -> {expected_id}"
        )

        # Retrieve
        retrieved = retrieve_top_k_memories(db, query_text, k=50)

        # Calculate metrics using exact ID match
        metrics = MemoryMetricsCalculator.evaluate_memory_query(
            query=query_text,
            expected_id=expected_id,
            retrieved_chunks=retrieved,
            query_type=qtype,
        )

        query_metrics.append(metrics)

    # Aggregate metrics
    summary = MemoryEvaluationSummary(
        total_queries=len(query_metrics), per_query_metrics=query_metrics
    )

    for m in query_metrics:
        if m.memory_hit_at_1:
            summary.queries_with_hits_at_1 += 1
        if m.memory_hit_at_3:
            summary.queries_with_hits_at_3 += 1
        if m.memory_hit_at_5:
            summary.queries_with_hits_at_5 += 1
        if m.memory_hit_at_10:
            summary.queries_with_hits_at_10 += 1

        if m.candidate_recall_at_10:
            summary.queries_with_candidate_recall_at_10 += 1
        if m.candidate_recall_at_20:
            summary.queries_with_candidate_recall_at_20 += 1
        if m.candidate_recall_at_50:
            summary.queries_with_candidate_recall_at_50 += 1

        summary.avg_memory_mrr += m.memory_mrr

    if summary.total_queries > 0:
        summary.avg_memory_mrr /= summary.total_queries
        summary.compute_rates()

    return summary


def format_report(summary: MemoryEvaluationSummary, verbose: bool = False) -> str:
    """Generate human-readable ID-based evaluation report."""
    lines: List[str] = []
    lines.append("=" * 70)
    lines.append("MEMORY RETRIEVAL EVALUATION REPORT (ID-BASED GROUND TRUTH)")
    lines.append("=" * 70)
    lines.append("")

    lines.append(f"Timestamp:           {datetime.now().isoformat()}")
    lines.append(f"Total Queries:       {summary.total_queries}")
    lines.append("")

    lines.append("HIT RATE METRICS (Exact ID Match):")
    lines.append(
        f"  Hit@1:             {summary.memory_hit_at_1_rate:.1%} ({summary.queries_with_hits_at_1}/{summary.total_queries})"
    )
    lines.append(
        f"  Hit@3:             {summary.memory_hit_at_3_rate:.1%} ({summary.queries_with_hits_at_3}/{summary.total_queries})"
    )
    lines.append(
        f"  Hit@5:             {summary.memory_hit_at_5_rate:.1%} ({summary.queries_with_hits_at_5}/{summary.total_queries})"
    )
    lines.append(
        f"  Hit@10:            {summary.memory_hit_at_10_rate:.1%} ({summary.queries_with_hits_at_10}/{summary.total_queries})"
    )
    lines.append(f"  MRR:               {summary.avg_memory_mrr:.3f}")
    lines.append("")

    lines.append("CANDIDATE RECALL (Is it retrievable at all?):")
    lines.append(
        f"  Top 10:            {summary.candidate_recall_at_10_rate:.1%} ({summary.queries_with_candidate_recall_at_10})"
    )
    lines.append(
        f"  Top 20:            {summary.candidate_recall_at_20_rate:.1%} ({summary.queries_with_candidate_recall_at_20})"
    )
    lines.append(
        f"  Top 50:            {summary.candidate_recall_at_50_rate:.1%} ({summary.queries_with_candidate_recall_at_50})"
    )
    lines.append("")

    if verbose and summary.per_query_metrics:
        lines.append("=" * 70)
        lines.append("DETAILED QUERY RESULTS")
        lines.append("=" * 70)

        for i, m in enumerate(summary.per_query_metrics, start=1):
            lines.append(f"[{i}] Query: {m.query}")
            lines.append(f"    Expected ID: {m.expected_id}")
            lines.append(
                f"    Rank found: {m.memory_rank if m.memory_rank else 'Not in top 50'}"
            )
            lines.append(
                f"    Hit@1/3/5/10: {m.memory_hit_at_1}/{m.memory_hit_at_3}/{m.memory_hit_at_5}/{m.memory_hit_at_10}"
            )

            if m.retrieved_chunks:
                lines.append("    Top 3 retrieved:")
                for chunk in m.retrieved_chunks[:3]:
                    marker = (
                        "[EXPECTED]" if chunk.memory_id == m.expected_id else "[WRONG]"
                    )
                    lines.append(
                        f"      [{chunk.rank}] {marker} dist={chunk.distance:.4f} id={chunk.memory_id}: {chunk.chunk_text[:50]}..."
                    )
            lines.append("")

    return "\n".join(lines)


def main():
    """Main CLI entrypoint."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate Weft memory retrieval with ID fixtures"
    )
    parser.add_argument("--queries", type=str, default=None)
    parser.add_argument("--fixtures", type=str, default=None)
    parser.add_argument("--verbose", action="store_true", default=True)

    args = parser.parse_args()

    queries = load_queries(args.queries)
    if not queries:
        logger.warning("No queries loaded. Exiting.")
        return 1

    db = SessionLocal()
    inserted_ids = []
    try:
        # 1. Load fixtures
        inserted_ids = load_fixtures(db, args.fixtures)

        # 2. Evaluate
        logger.info("Starting memory retrieval evaluation...")
        summary = evaluate_all_queries(db, queries)

        # 3. Print report
        print("\n")
        print(format_report(summary, verbose=args.verbose))

    finally:
        # 4. Cleanup
        cleanup_fixtures(db, inserted_ids)
        db.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
