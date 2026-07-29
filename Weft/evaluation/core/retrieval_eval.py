"""Retrieval evaluation pipeline."""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from Weft.core.retrieval import VectorRetriever
from Weft.evaluation.core.metrics import (
    EvaluationSummary,
    MetricsCalculator,
    RetrievalResult,
)
from Weft.utils.exceptions import WeftException
from Weft.utils.logger import logger

# Initialize vector retriever at module level (can be slow)
RETRIEVER = None


def get_retriever() -> VectorRetriever:
    """Lazily load vector retriever."""
    global RETRIEVER
    if RETRIEVER is None:
        RETRIEVER = VectorRetriever()
    return RETRIEVER


def retrieve_top_k(query: str, k: int = 10) -> List[RetrievalResult]:
    """Retrieve top-k chunks for a query using VectorRetriever.

    Args:
        query: Search query string.
        k: Number of results to return (default 10).

    Returns:
        List of RetrievalResult objects, ranked by distance.
    """
    retriever = get_retriever()
    return retriever.retrieve(query, k=k)


def load_test_queries(queries_file: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load test queries from JSON file.

    Args:
        queries_file: Path to test_queries.json. If None, uses default location.

    Returns:
        List of query dicts with 'query' and 'expected_keywords'.
    """
    resolved_path: Path
    if queries_file is None:
        resolved_path = Path(__file__).parent / "test_queries.json"
    else:
        resolved_path = Path(queries_file)

    if not resolved_path.exists():
        logger.error(f"Queries file not found: {resolved_path}")
        return []

    try:
        with open(resolved_path, "r", encoding="utf-8") as f:
            queries: List[Dict[str, Any]] = json.load(f)
        logger.info(f"Loaded {len(queries)} test queries")
        return queries
    except Exception as e:
        raise WeftException(str(e), e) from e


def evaluate_all_queries(queries: List[Dict[str, Any]]) -> EvaluationSummary:
    """Evaluate all queries and compute metrics.

    Args:
        queries: List of query dicts with 'query' and 'expected_keywords'.

    Returns:
        EvaluationSummary with aggregated metrics.
    """
    query_metrics = []

    for i, query_dict in enumerate(queries, start=1):
        query_text = query_dict.get("query", "")
        expected_kw = query_dict.get("expected_keywords", [])

        if not query_text:
            logger.warning(f"Query {i}: missing 'query' field")
            continue

        logger.info(f"[{i}/{len(queries)}] Evaluating: {query_text[:60]}...")

        # Retrieve top-k results
        retrieved = retrieve_top_k(query_text, k=10)

        # Calculate metrics
        metrics = MetricsCalculator.evaluate_query(
            query=query_text, expected_keywords=expected_kw, retrieved_chunks=retrieved
        )

        query_metrics.append(metrics)

    return EvaluationSummary(query_metrics)


def format_report(summary: EvaluationSummary, verbose: bool = False) -> str:
    """Generate human-readable evaluation report.

    Args:
        summary: EvaluationSummary object.
        verbose: If True, include detailed query results.

    Returns:
        Formatted report string.
    """
    lines: List[str] = []
    lines.append("=" * 70)
    lines.append("RETRIEVAL EVALUATION REPORT")
    lines.append("=" * 70)
    lines.append("")

    # Summary metrics
    lines.append(f"Timestamp:           {datetime.now().isoformat()}")
    lines.append(f"Total Queries:       {summary.total_queries}")
    lines.append(f"Passed (Hit):        {len(summary.passed_queries)}")
    lines.append(f"Failed (No Hit):     {len(summary.failed_queries)}")
    lines.append("")

    lines.append("HIT RATE METRICS:")
    lines.append(f"  Hit@1:             {summary.hit_at_1_rate:.1%}")
    lines.append(f"  Hit@3:             {summary.hit_at_3_rate:.1%}")
    lines.append(f"  Hit@5:             {summary.hit_at_5_rate:.1%}")
    lines.append(f"  Hit@10:            {summary.hit_at_10_rate:.1%}")
    lines.append("")

    lines.append("KEYWORD METRICS:")
    lines.append(f"  Avg Keyword Recall:     {summary.avg_keyword_recall:.3f}")
    lines.append(f"  Avg Keyword Precision:  {summary.avg_keyword_precision:.3f}")
    lines.append(f"  Avg MRR:                {summary.avg_mrr:.3f}")
    lines.append("")

    # Best queries
    if summary.passed_queries:
        best_by_mrr = sorted(summary.passed_queries, key=lambda m: m.mrr, reverse=True)[
            :3
        ]
        lines.append("BEST PERFORMING QUERIES (by MRR):")
        for m in best_by_mrr:
            lines.append(f"  • {m.query[:50]}")
            lines.append(
                f"    MRR: {m.mrr:.3f}, Hit@1: {m.hit_at_1}, Recall: {m.keyword_recall:.2f}"  # noqa: E501
            )
        lines.append("")

    # Worst queries
    if summary.failed_queries:
        worst = summary.failed_queries[:3]
        lines.append("WORST PERFORMING QUERIES (no hits in top-10):")
        for m in worst:
            lines.append(f"  • {m.query[:50]}")
            lines.append(f"    Expected: {', '.join(m.expected_keywords[:2])}")
            if m.retrieved_chunks:
                top_chunk = m.retrieved_chunks[0]
                lines.append(
                    f"    Top result (dist={top_chunk.distance:.3f}): {top_chunk.chunk_text[:60]}..."  # noqa: E501
                )
        lines.append("")

    # Detailed results if verbose
    if verbose:
        lines.append("=" * 70)
        lines.append("DETAILED QUERY RESULTS")
        lines.append("=" * 70)
        lines.append("")

        for i, m in enumerate(summary.query_metrics, start=1):
            lines.append(f"[{i}] Query: {m.query}")
            lines.append(f"    Expected Keywords: {', '.join(m.expected_keywords)}")
            lines.append(
                f"    Hit@1/3/5/10: {m.hit_at_1}/{m.hit_at_3}/{m.hit_at_5}/{m.hit_at_10}"  # noqa: E501
            )
            lines.append(
                f"    Recall: {m.keyword_recall:.2f}, Precision: {m.keyword_precision:.2f}, MRR: {m.mrr:.3f}"  # noqa: E501
            )

            if m.retrieved_chunks:
                lines.append("    Top 3 retrieved chunks:")
                for chunk in m.retrieved_chunks[:3]:
                    lines.append(
                        f"      [{chunk.rank}] dist={chunk.distance:.4f}: {chunk.chunk_text[:70]}..."  # noqa: E501
                    )
            lines.append("")

    lines.append("=" * 70)

    return "\n".join(lines)


def format_json_report(summary: EvaluationSummary) -> Dict[str, Any]:
    """Generate JSON-serializable report.

    Args:
        summary: EvaluationSummary object.

    Returns:
        Dictionary suitable for JSON serialization.
    """
    return {
        "timestamp": datetime.now().isoformat(),
        "summary": summary.to_dict(),
        "per_query_results": [
            {
                "query": m.query,
                "expected_keywords": m.expected_keywords,
                "hit_at_1": m.hit_at_1,
                "hit_at_3": m.hit_at_3,
                "hit_at_5": m.hit_at_5,
                "hit_at_10": m.hit_at_10,
                "keyword_recall": round(m.keyword_recall, 3),
                "keyword_precision": round(m.keyword_precision, 3),
                "mrr": round(m.mrr, 3),
                "top_results": [
                    {
                        "rank": c.rank,
                        "distance": round(c.distance, 4),
                        "chunk_text": c.chunk_text[:100],
                        "conversation_id": c.conversation_id,
                    }
                    for c in m.retrieved_chunks[:3]
                ],
            }
            for m in summary.query_metrics
        ],
    }


def main():
    """Main CLI entrypoint."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate Weft retrieval system")
    parser.add_argument(
        "--queries",
        type=str,
        default=None,
        help="Path to test_queries.json (default: ./Weft/evaluation/test_queries.json)",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print detailed results for each query"
    )
    parser.add_argument(
        "--json-output", type=str, default=None, help="Save JSON report to file"
    )

    args = parser.parse_args()

    # Load queries
    queries = load_test_queries(args.queries)
    if not queries:
        logger.warning("No queries loaded. Exiting.")
        return 1

    # Evaluate
    logger.info("Starting retrieval evaluation...")
    summary = evaluate_all_queries(queries)

    # Print report
    print("\n")
    print(format_report(summary, verbose=args.verbose))

    # Save JSON if requested
    if args.json_output:
        json_report = format_json_report(summary)
        output_path = Path(args.json_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(json_report, f, indent=2, default=str)
        logger.info(f"JSON report saved to: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
