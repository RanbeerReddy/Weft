"""Phase 6 — Search Quality Experiments.

Implements 4 retrieval experiments, keeping everything else constant:

    Experiment 1: Current vector search (baseline)
    Experiment 2: Increase Top-K (10, 20, 50, 100)
    Experiment 3: Hybrid Search (BM25 via tsvector + pgvector)
    Experiment 4: Metadata filtering (conversation title, role, date)

All experiments use the same benchmark queries and produce comparable
metrics for side-by-side comparison.

NO changes are kept to production code unless benchmark improves.
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from sentence_transformers import SentenceTransformer
from sqlalchemy import select, text

from Weft.config.settings import settings
from Weft.evaluation.core.memory_metrics import (
    MemoryMetricsCalculator,
    MemoryRetrievalResult,
)
from Weft.storage.database import SessionLocal
from Weft.storage.models import Chunk, Conversation, Embedding, Message
from Weft.utils.exceptions import WeftException
from Weft.utils.logger import logger

MODEL: Optional[SentenceTransformer] = None


def get_model() -> SentenceTransformer:
    global MODEL
    if MODEL is None:
        logger.info("[*] Loading embedding model: BAAI/bge-small-en-v1.5")
        MODEL = SentenceTransformer(settings.EMBEDDING_MODEL)
    return MODEL


def vector_search(db, query: str, k: int = 10) -> List[MemoryRetrievalResult]:
    """Standard vector search (baseline)."""
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
        .limit(k)
    )

    results = db.execute(stmt).fetchall()
    return [
        MemoryRetrievalResult(
            rank=rank,
            distance=float(row.distance),
            chunk_text=row.chunk_text,
            conversation_id=row.conversation_id,
            conversation_title=row.title,
            message_id=row.message_id,
            message_role=row.role,
            message_timestamp=str(row.create_time) if row.create_time else None,
        )
        for rank, row in enumerate(results, start=1)
    ]


def hybrid_search(
    db, query: str, k: int = 10, alpha: float = 0.5
) -> List[MemoryRetrievalResult]:
    """Hybrid search: vector score + BM25 (tsvector) score fusion.

    Uses Reciprocal Rank Fusion (RRF) to combine vector and text search results.

    Args:
        db: Database session
        query: Search query
        k: Number of results
        alpha: Weight for vector score (1-alpha for text score)
    """
    model = get_model()
    vector = model.encode(query, normalize_embeddings=True).tolist()

    # Use raw SQL for hybrid search with RRF
    # Get vector results
    distance_attr = Embedding.embedding_vector.cosine_distance(vector)

    vec_stmt = (
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
        .limit(k * 3)  # Get more candidates for fusion
    )
    vec_results = db.execute(vec_stmt).fetchall()

    # Get text search results using PostgreSQL full-text search
    # Use plainto_tsquery for simple query parsing
    ts_stmt = text(
        """
        SELECT c.conversation_id, c.message_id, c.id as chunk_id,
               c.chunk_text,
               ts_rank(to_tsvector('english', c.chunk_text),
                       plainto_tsquery('english', :query)) as text_score,
               conv.title, m.role, m.create_time
        FROM chunks c
        JOIN conversations conv ON conv.id = c.conversation_id
        JOIN messages m ON m.id = c.message_id
        WHERE to_tsvector('english', c.chunk_text) @@ plainto_tsquery('english', :query)
        ORDER BY text_score DESC
        LIMIT :limit
    """
    )
    text_results = db.execute(ts_stmt, {"query": query, "limit": k * 3}).fetchall()

    # Reciprocal Rank Fusion
    rrf_scores = {}  # chunk_id -> (rrf_score, row_data)
    RRF_K = 60  # standard RRF constant

    for rank, row in enumerate(vec_results, start=1):
        chunk_id = row.chunk_order
        score = alpha * (1.0 / (RRF_K + rank))
        if chunk_id not in rrf_scores:
            rrf_scores[chunk_id] = {
                "score": score,
                "conversation_id": row.conversation_id,
                "message_id": row.message_id,
                "chunk_text": row.chunk_text,
                "distance": float(row.distance),
                "title": row.title,
                "role": row.role,
                "create_time": row.create_time,
            }
        else:
            rrf_scores[chunk_id]["score"] += score

    for rank, row in enumerate(text_results, start=1):
        chunk_id = row.chunk_id
        score = (1 - alpha) * (1.0 / (RRF_K + rank))
        if chunk_id not in rrf_scores:
            rrf_scores[chunk_id] = {
                "score": score,
                "conversation_id": row.conversation_id,
                "message_id": row.message_id,
                "chunk_text": row.chunk_text,
                "distance": 1.0,  # unknown exact vector distance
                "title": row.title,
                "role": row.role,
                "create_time": row.create_time,
            }
        else:
            rrf_scores[chunk_id]["score"] += score

    # Sort by RRF score and take top-k
    sorted_results = sorted(
        rrf_scores.values(), key=lambda x: x["score"], reverse=True
    )[:k]

    return [
        MemoryRetrievalResult(
            rank=rank,
            distance=r["distance"],
            chunk_text=r["chunk_text"],
            conversation_id=r["conversation_id"],
            conversation_title=r["title"],
            message_id=r["message_id"],
            message_role=r["role"],
            message_timestamp=str(r["create_time"]) if r["create_time"] else None,
        )
        for rank, r in enumerate(sorted_results, start=1)
    ]


def metadata_filtered_search(
    db, query: str, k: int = 10, role_filter: Optional[str] = None
) -> List[MemoryRetrievalResult]:
    """Vector search with metadata pre-filtering.

    Filters by role (user messages tend to contain personal information).
    """
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
    )

    if role_filter:
        stmt = stmt.where(Message.role == role_filter)

    stmt = stmt.order_by("distance").limit(k)

    results = db.execute(stmt).fetchall()
    return [
        MemoryRetrievalResult(
            rank=rank,
            distance=float(row.distance),
            chunk_text=row.chunk_text,
            conversation_id=row.conversation_id,
            conversation_title=row.title,
            message_id=row.message_id,
            message_role=row.role,
            message_timestamp=str(row.create_time) if row.create_time else None,
        )
        for rank, row in enumerate(results, start=1)
    ]


def evaluate_experiment(
    queries: List[Dict[str, Any]],
    search_fn,
    experiment_name: str,
    db,
    **search_kwargs,
) -> Dict[str, Any]:
    """Run a search experiment and compute metrics.

    Args:
        queries: Benchmark queries with expected_phrase
        search_fn: Search function to use
        experiment_name: Name for this experiment
        db: Database session
        **search_kwargs: Additional args for search_fn

    Returns:
        Experiment results with metrics
    """
    print(f"\n  Running: {experiment_name}")
    t0 = time.time()

    per_query = []
    hits_at = {1: 0, 3: 0, 5: 0, 10: 0}
    total_mrr = 0.0

    for q in queries:
        query_text = q.get("query", "")
        expected = q.get("expected_phrase", "")

        results = search_fn(db, query_text, **search_kwargs)

        # Evaluate
        metrics = MemoryMetricsCalculator.evaluate_memory_query(
            query_text, expected, results
        )

        if metrics.memory_hit_at_1:
            hits_at[1] += 1
        if metrics.memory_hit_at_3:
            hits_at[3] += 1
        if metrics.memory_hit_at_5:
            hits_at[5] += 1
        if metrics.memory_hit_at_10:
            hits_at[10] += 1
        total_mrr += metrics.phrase_mrr

        per_query.append(
            {
                "query": query_text,
                "expected_phrase": expected,
                "phrase_rank": metrics.phrase_rank,
                "phrase_mrr": metrics.phrase_mrr,
                "hit_at_1": metrics.memory_hit_at_1,
                "hit_at_10": metrics.memory_hit_at_10,
            }
        )

    elapsed = time.time() - t0
    n = len(queries)

    result = {
        "experiment": experiment_name,
        "total_queries": n,
        "elapsed_seconds": round(elapsed, 2),
        "hit_at_1_rate": round(hits_at[1] / n, 4) if n else 0,
        "hit_at_3_rate": round(hits_at[3] / n, 4) if n else 0,
        "hit_at_5_rate": round(hits_at[5] / n, 4) if n else 0,
        "hit_at_10_rate": round(hits_at[10] / n, 4) if n else 0,
        "mrr": round(total_mrr / n, 4) if n else 0,
        "per_query": per_query,
    }

    print(
        f"    Hit@1={result['hit_at_1_rate']:.1%}  "
        f"Hit@10={result['hit_at_10_rate']:.1%}  "
        f"MRR={result['mrr']:.4f}  "
        f"({elapsed:.1f}s)"
    )

    return result


def run_experiments(
    memory_queries_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Run all search experiments."""
    print("=" * 70)
    print("PHASE 6 — SEARCH QUALITY EXPERIMENTS")
    print("=" * 70)

    report: Dict[str, Any] = {"timestamp": datetime.now().isoformat()}

    # Load queries
    if memory_queries_path is None:
        memory_queries_path = str(Path(__file__).parent / "memory_queries.json")

    with open(memory_queries_path, "r", encoding="utf-8") as f:
        queries = json.load(f)
    logger.info(f"[+] Loaded {len(queries)} benchmark queries")

    db = SessionLocal()
    try:
        experiments = []

        # Experiment 1: Baseline (current, k=10)
        exp1 = evaluate_experiment(
            queries, vector_search, "Exp1: Vector Search (k=10)", db, k=10
        )
        experiments.append(exp1)

        # Experiment 2: Top-K variations
        for k in [20, 50, 100]:
            exp = evaluate_experiment(
                queries, vector_search, f"Exp2: Vector Search (k={k})", db, k=k
            )
            experiments.append(exp)

        # Experiment 3: Hybrid Search
        for alpha in [0.7, 0.5, 0.3]:
            try:
                exp = evaluate_experiment(
                    queries,
                    hybrid_search,
                    f"Exp3: Hybrid (alpha={alpha})",
                    db,
                    k=10,
                    alpha=alpha,
                )
                experiments.append(exp)
            except Exception as e:
                raise WeftException(str(e), e) from e
                experiments.append(
                    {
                        "experiment": f"Exp3: Hybrid (alpha={alpha})",
                        "error": str(e),
                    }
                )

        # Experiment 4: Metadata filtering (user messages only)
        exp4 = evaluate_experiment(
            queries,
            metadata_filtered_search,
            "Exp4: Vector + User-only filter",
            db,
            k=10,
            role_filter="user",
        )
        experiments.append(exp4)

        report["experiments"] = experiments

        # Print comparison table
        print("\n" + "=" * 70)
        print("EXPERIMENT COMPARISON")
        print("=" * 70)
        print(
            f"\n  {'Experiment':<40} {'Hit@1':>6} {'Hit@3':>6} {'Hit@5':>6} {'Hit@10':>7} {'MRR':>7}"  # noqa: E501
        )
        print("  " + "-" * 68)

        baseline_mrr = None
        for exp in experiments:
            if "error" in exp:
                print(f"  {exp['experiment']:<40} ERROR: {exp['error'][:20]}")
                continue
            if baseline_mrr is None:
                baseline_mrr = exp["mrr"]
            delta = (
                f" ({exp['mrr'] - baseline_mrr:+.4f})"
                if baseline_mrr and exp != experiments[0]
                else ""
            )
            print(
                f"  {exp['experiment']:<40} "
                f"{exp['hit_at_1_rate']:>5.1%} "
                f"{exp['hit_at_3_rate']:>5.1%} "
                f"{exp['hit_at_5_rate']:>5.1%} "
                f"{exp['hit_at_10_rate']:>6.1%} "
                f"{exp['mrr']:>6.4f}{delta}"
            )

        # Identify best experiment
        valid_exps = [e for e in experiments if "error" not in e]
        if valid_exps:
            best = max(valid_exps, key=lambda e: e["mrr"])
            print(f"\n  Best by MRR: {best['experiment']} (MRR={best['mrr']:.4f})")

            if best["mrr"] > (baseline_mrr or 0):
                improvement = best["mrr"] - (baseline_mrr or 0)
                print(f"  Improvement over baseline: +{improvement:.4f} MRR")
            else:
                print("  No improvement over baseline")

    finally:
        db.close()

    return report


def main():
    parser = argparse.ArgumentParser(description="Weft Search Experiments")
    parser.add_argument(
        "--output",
        type=str,
        default="search_experiments_report.json",
        help="Output report file",
    )
    args = parser.parse_args()

    report = run_experiments()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n[+] Report saved to {output_path}")


if __name__ == "__main__":
    main()
