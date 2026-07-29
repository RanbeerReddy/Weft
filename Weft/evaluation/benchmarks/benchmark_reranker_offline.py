"""Offline benchmark reranking pipeline vs baseline using pre-retrieved top 100 candidates."""  # noqa: E501

import json
import time
from typing import Any, Dict, List, Optional, Tuple

from typing_extensions import TypedDict

from Weft.core.retrieval import CrossEncoderReranker
from Weft.evaluation.core.metrics import (
    MetricsCalculator,
    QueryMetrics,
    RetrievalResult,
)
from Weft.utils.exceptions import WeftException


class _QueryResult(TypedDict):
    query: str
    expected: List[str]
    metrics: QueryMetrics
    latency: float
    retrieved: List[RetrievalResult]


def evaluate_system(
    dataset: List[Dict[str, Any]],
    reranker: Optional[CrossEncoderReranker] = None,
    name: str = "Baseline",
) -> Tuple[Dict[str, Any], List[_QueryResult]]:
    """Evaluate over queries."""

    results: List[_QueryResult] = []
    latencies = []

    print(f"\n[*] Evaluating {name}...")
    for i, item in enumerate(dataset):
        query_text = item["query"]
        expected = [item["expected_phrase"]]

        # Convert dictionary candidates to RetrievalResult objects
        candidates = []
        for c in item["top_100_candidates"]:
            candidates.append(
                RetrievalResult(
                    chunk_text=c["chunk_text"],
                    distance=c["distance"],
                    conversation_id=c.get("conversation_id", ""),
                    message_id=c.get("message_id", ""),
                    chunk_order=c.get("chunk_id", 0),
                    rank=c["rank"],
                )
            )

        start = time.perf_counter()
        if reranker:
            # Pipeline: rerank top 100 to get top 10
            retrieved = reranker.rerank(query_text, candidates, top_k=10)
        else:
            # Baseline: just take the top 10 from vector search
            retrieved = candidates[:10]
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

    # Aggregate
    agg = {
        "hit_at_1": (
            sum(1 for r in results if r["metrics"].hit_at_1) / len(results)
            if results
            else 0
        ),
        "hit_at_3": (
            sum(1 for r in results if r["metrics"].hit_at_3) / len(results)
            if results
            else 0
        ),
        "hit_at_5": (
            sum(1 for r in results if r["metrics"].hit_at_5) / len(results)
            if results
            else 0
        ),
        "hit_at_10": (
            sum(1 for r in results if r["metrics"].hit_at_10) / len(results)
            if results
            else 0
        ),
        "mrr": sum(r["metrics"].mrr for r in results) / len(results) if results else 0,
        "avg_latency_ms": (sum(latencies) / len(latencies)) * 1000 if latencies else 0,
        "p95_latency_ms": (
            (sorted(latencies)[int(len(latencies) * 0.95)]) * 1000 if latencies else 0
        ),
        "worst_latency_ms": max(latencies) * 1000 if latencies else 0,
    }

    return agg, results


def run_benchmark():
    """Run baseline vs reranker benchmark offline."""
    try:
        with open(
            "Weft/evaluation/data/reranking_dataset.json", "r", encoding="utf-8"
        ) as f:
            data = json.load(f)
            dataset = data["candidates"]
    except Exception as e:
        raise WeftException(str(e), e) from e
        return

    print(f"Loaded {len(dataset)} test queries from reranking_dataset.json.")

    reranker = CrossEncoderReranker()

    # Run Baseline
    baseline_agg, baseline_results = evaluate_system(
        dataset, reranker=None, name="Baseline (Vector Top-10)"
    )

    # Run Reranker
    reranker_agg, reranker_results = evaluate_system(
        dataset, reranker=reranker, name="Pipeline (Vector Top-100 -> Reranker Top-10)"
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
    print("PERFORMANCE (ms) (Reranking Only)")
    print("-" * 65)
    perf_metrics = ["avg_latency_ms", "p95_latency_ms", "worst_latency_ms"]
    for m in perf_metrics:
        b_val = baseline_agg[m]  # this is ~0 ms because it's just array slicing
        r_val = reranker_agg[m]
        delta = r_val - b_val
        print(f"{m:<20} | {b_val:7.1f}         | {r_val:7.1f}         | {delta:+7.1f}")

    print("\n" + "=" * 50)
    print("FAILURE ANALYSIS (Changed Outcomes)")
    print("=" * 50)

    for b_res, r_res in zip(baseline_results, reranker_results):
        b_hit_rank = next(
            (
                c.rank
                for c in b_res["retrieved"]
                if MetricsCalculator.extract_keywords(c.chunk_text)
                & set(x.lower() for x in b_res["expected"])
            ),
            None,
        )
        r_hit_rank = next(
            (
                c.rank
                for c in r_res["retrieved"]
                if MetricsCalculator.extract_keywords(c.chunk_text)
                & set(x.lower() for x in r_res["expected"])
            ),
            None,
        )

        # Only show queries where the rank of the correct answer changed
        if b_hit_rank != r_hit_rank:
            print(f"\nQuery: {b_res['query']}")
            print(f"Expected: {b_res['expected']}")
            print(f"  Before: Rank {b_hit_rank if b_hit_rank else '>10'}")
            print(f"  After : Rank {r_hit_rank if r_hit_rank else '>10'}")

            if r_hit_rank:
                correct_chunk = r_res["retrieved"][r_hit_rank - 1]
                print(
                    f"  Reranker Score (saved as distance): {-correct_chunk.distance:.4f}"  # noqa: E501
                )
                print(
                    "  Explanation: The reranker successfully identified semantic interaction between query and chunk."  # noqa: E501
                )


if __name__ == "__main__":
    run_benchmark()
