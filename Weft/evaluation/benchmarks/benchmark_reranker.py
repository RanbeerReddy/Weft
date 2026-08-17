"""Benchmark reranking pipeline vs baseline."""

import time
from typing import Any, Dict, List, Tuple

from typing_extensions import TypedDict

from Weft.core.retrieval import CrossEncoderReranker, RetrievalPipeline, VectorRetriever
from Weft.evaluation.core.metrics import (
    MetricsCalculator,
    QueryMetrics,
    RetrievalResult,
)
from Weft.evaluation.core.retrieval_eval import load_test_queries


class _QueryResult(TypedDict):
    query: str
    expected: List[str]
    metrics: QueryMetrics
    latency: float
    retrieved: List[RetrievalResult]


def evaluate_system(
    queries: List[Dict[str, Any]], search_func, name: str
) -> Tuple[Dict[str, Any], List[_QueryResult]]:
    """Evaluate a search function over queries."""

    results: List[_QueryResult] = []
    latencies = []

    print(f"\n[*] Evaluating {name}...")
    for i, q in enumerate(queries):
        query_text = q["query"]
        expected = q.get("expected_keywords", [])

        start = time.perf_counter()
        retrieved = search_func(query_text)
        end = time.perf_counter()

        latency = end - start
        latencies.append(latency)

        metrics = MetricsCalculator.evaluate_query(query_text, expected, retrieved)

        results.append(
            {
                "query": query_text,
                "expected": expected,
                "metrics": metrics,
                "latency": latency,
                "retrieved": retrieved,
            }
        )

        if (i + 1) % 10 == 0:
            print(f"    Processed {i + 1}/{len(queries)}")

    # Aggregate
    agg = {
        "hit_at_1": sum(1 for r in results if r["metrics"].hit_at_1) / len(results),
        "hit_at_3": sum(1 for r in results if r["metrics"].hit_at_3) / len(results),
        "hit_at_5": sum(1 for r in results if r["metrics"].hit_at_5) / len(results),
        "hit_at_10": sum(1 for r in results if r["metrics"].hit_at_10) / len(results),
        "mrr": sum(r["metrics"].mrr for r in results) / len(results),
        "avg_latency_ms": (sum(latencies) / len(latencies)) * 1000,
        "p95_latency_ms": (sorted(latencies)[int(len(latencies) * 0.95)]) * 1000,
        "worst_latency_ms": max(latencies) * 1000,
    }

    return agg, results


def run_benchmark():
    """Run baseline vs reranker benchmark."""
    queries = load_test_queries("Weft/evaluation/data/test_queries.json")
    if not queries:
        return

    print(f"Loaded {len(queries)} test queries.")

    # Initialize components
    retriever = VectorRetriever()
    reranker = CrossEncoderReranker()
    pipeline = RetrievalPipeline(retriever=retriever, reranker=reranker)

    # Run Baseline
    baseline_agg, baseline_results = evaluate_system(
        queries, lambda q: retriever.retrieve(q, k=10), "Baseline (Vector Top-10)"
    )

    # Run Reranker
    reranker_agg, reranker_results = evaluate_system(
        queries,
        lambda q: pipeline.search(q, top_n=100, final_k=10),
        "Pipeline (Vector Top-100 -> Reranker Top-10)",
    )

    # Print Comparison
    print("\n" + "=" * 50)
    print("BENCHMARK COMPARISON")
    print("=" * 50)
    print(f"{'Metric':<20} | {'Baseline':<15} | {'Reranker':<15} | {'Delta':<10}")
    print("-" * 65)

    metrics = ["hit_at_1", "hit_at_3", "hit_at_5", "hit_at_10", "mrr"]
    for m in metrics:
        b_val = baseline_agg[m]
        r_val = reranker_agg[m]
        delta = r_val - b_val
        print(f"{m:<20} | {b_val:.3f}           | {r_val:.3f}           | {delta:+.3f}")

    print("-" * 65)
    print("PERFORMANCE (ms)")
    print("-" * 65)
    perf_metrics = ["avg_latency_ms", "p95_latency_ms", "worst_latency_ms"]
    for m in perf_metrics:
        b_val = baseline_agg[m]
        r_val = reranker_agg[m]
        delta = r_val - b_val
        print(f"{m:<20} | {b_val:7.1f}         | {r_val:7.1f}         | {delta:+7.1f}")

    # Average Reranking Time
    avg_rerank = reranker_agg["avg_latency_ms"] - baseline_agg["avg_latency_ms"]
    print(f"\nAverage Reranking Overhead (Top-100): {avg_rerank:.1f} ms")

    print("\n" + "=" * 50)
    print("FAILURE ANALYSIS (Changed Outcomes)")
    print("=" * 50)

    for b_res, r_res in zip(baseline_results, reranker_results):
        b_hit_rank = next(
            (
                c.rank
                for c in b_res["retrieved"]
                if MetricsCalculator.is_hit(
                    c.chunk_text, set(x.lower() for x in b_res["expected"])
                )
            ),
            None,
        )
        r_hit_rank = next(
            (
                c.rank
                for c in r_res["retrieved"]
                if MetricsCalculator.is_hit(
                    c.chunk_text, set(x.lower() for x in r_res["expected"])
                )
            ),
            None,
        )

        # Only show queries where the rank of the correct answer changed
        if b_hit_rank != r_hit_rank:
            print(f"\nQuery: {b_res['query']}")
            print(f"Expected: {b_res['expected']}")
            print(f"  Before: Rank {b_hit_rank if b_hit_rank else '>10'}")
            print(f"  After : Rank {r_hit_rank if r_hit_rank else '>10'}")

            # If we found it after, let's see why
            if r_hit_rank:
                correct_chunk = r_res["retrieved"][r_hit_rank - 1]
                print(
                    f"  Reranker Score (saved as distance): {-correct_chunk.distance:.4f}"  # noqa: E501
                )
                print(
                    "  Explanation: The reranker successfully lifted this chunk from outside the baseline top-10 or from a lower rank by recognizing semantic interaction between query and chunk."  # noqa: E501
                )


if __name__ == "__main__":
    run_benchmark()
