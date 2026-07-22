"""Reranker Audit Script."""

import json
import time
from typing import Any, Dict, List, Tuple

from Weft.core.retrieval import CrossEncoderReranker
from Weft.evaluation.core.metrics import MetricsCalculator, RetrievalResult
from Weft.utils.exceptions import WeftException


def evaluate_with_reranker(
    dataset: List[Dict[str, Any]], reranker: CrossEncoderReranker, top_n: int
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    """Evaluate reranker on exactly top_n candidates."""
    results: List[Dict[str, Any]] = []
    latencies = []

    for item in dataset:
        query_text = item["query"]
        expected = [item["expected_phrase"]]

        candidates = []
        for c in item["top_100_candidates"][:top_n]:  # Take exactly top_n
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
        retrieved = reranker.rerank(query_text, candidates, top_k=10)
        end = time.perf_counter()

        latencies.append(end - start)
        metrics = MetricsCalculator.evaluate_query(query_text, expected, retrieved)
        results.append({"metrics": metrics, "retrieved": retrieved})

    agg = {
        "hit_at_1": (
            sum(1 for r in results if r["metrics"].hit_at_1) / len(results)
            if results
            else 0
        ),
        "mrr": sum(r["metrics"].mrr for r in results) / len(results) if results else 0,
        "avg_latency_ms": (sum(latencies) / len(latencies)) * 1000 if latencies else 0,
    }
    return agg, results


def run_ablation_tests(dataset: List[Dict[str, Any]]):
    print("\n--- PHASE 7: ABLATION TESTS ---")
    reranker = CrossEncoderReranker()

    for top_n in [10, 20, 50, 100]:
        agg, _ = evaluate_with_reranker(dataset, reranker, top_n)
        print(
            f"Top {top_n:<3} Reranked -> Hit@1: {agg['hit_at_1']:.3f} | MRR: {agg['mrr']:.3f} | Latency: {agg['avg_latency_ms']:.1f}ms"  # noqa: E501
        )


def run_chunking_vs_coherent_test(dataset: List[Dict[str, Any]]):
    print("\n--- PHASE 7: CHUNKING VS COHERENT TEST ---")
    reranker = CrossEncoderReranker()

    # We will pick a specific query that dropped significantly
    # e.g., "tell me about my accounting knowledge"
    target_query = "tell me about my accounting knowledge"
    item = next((i for i in dataset if i["query"] == target_query), None)

    if not item:
        print("Could not find target query in dataset.")
        return

    print(f"Query: {target_query}")

    # 1. Evaluate with current chunks (top 100)
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

    retrieved_fractured = reranker.rerank(target_query, candidates, top_k=10)

    print("\nFractured Chunks (Baseline):")
    for r in retrieved_fractured[:3]:
        print(
            f"Rank {r.rank} | Score {-r.distance:.4f} | {r.chunk_text[:60].replace(chr(10), ' ')}"  # noqa: E501
        )

    # 2. Evaluate with manually coherent text
    # We will create a fake coherent message by merging top 5 chunks into one block,
    # to simulate message-level chunking
    merged_text = " ".join([c.chunk_text for c in candidates[:5]])
    coherent_candidate = RetrievalResult(
        chunk_text=merged_text,
        distance=0.0,
        conversation_id="fake",
        message_id="fake",
        chunk_order=0,
        rank=1,
    )

    # We will score this single coherent chunk vs the same query
    retrieved_coherent = reranker.rerank(target_query, [coherent_candidate], top_k=1)

    print("\nCoherent Reconstructed Chunk:")
    for r in retrieved_coherent:
        print(
            f"Rank {r.rank} | Score {-r.distance:.4f} | {r.chunk_text[:60].replace(chr(10), ' ')}"  # noqa: E501
        )

    print("\nComparison:")
    score_fractured = -retrieved_fractured[0].distance if retrieved_fractured else -999
    score_coherent = -retrieved_coherent[0].distance if retrieved_coherent else -999
    print(f"Top Fractured Score: {score_fractured:.4f}")
    print(f"Coherent Score     : {score_coherent:.4f}")


def analyze_scores(dataset: List[Dict[str, Any]]):
    print("\n--- PHASE 3 & 5: SCORE AND INPUT VALIDATION ---")
    reranker = CrossEncoderReranker()

    item = dataset[0]  # Just use the first one
    query_text = item["query"]
    print(f"Query: {query_text}")

    # Take first 5 candidates
    candidates = []
    for c in item["top_100_candidates"][:5]:
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

    for c in candidates:
        print(f"\nOriginal Rank: {c.rank}")
        print(f"Embedding Dist: {c.distance:.4f}")
        print(f"Char count: {len(c.chunk_text)}")
        print(f"Chunk Preview: {c.chunk_text[:100].replace(chr(10), ' ')}")

    retrieved = reranker.rerank(query_text, candidates, top_k=5)
    print("\nAfter Reranking:")
    for r in retrieved:
        print(
            f"New Rank: {r.rank} | CE Score: {-r.distance:.4f} | Char count: {len(r.chunk_text)}"  # noqa: E501
        )


def main():
    try:
        with open(
            "Weft/evaluation/data/reranking_dataset.json", "r", encoding="utf-8"
        ) as f:
            data = json.load(f)
            dataset = data["candidates"]
    except Exception as e:
        raise WeftException(str(e), e) from e
        return

    analyze_scores(dataset)
    run_ablation_tests(dataset)
    run_chunking_vs_coherent_test(dataset)


if __name__ == "__main__":
    main()
