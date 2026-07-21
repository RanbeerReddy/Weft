import json
import time
from pathlib import Path
from typing import Any, Dict, List

from Weft.core.retrieval import HybridRetriever, LexicalRetriever, VectorRetriever
from Weft.evaluation.core.memory_metrics import (
    MemoryEvaluationSummary,
    MemoryMetricsCalculator,
    MemoryRetrievalResult,
)
from Weft.evaluation.core.memory_retrieval_eval import load_memory_queries
from Weft.evaluation.core.metrics import RetrievalResult


def convert_to_memory_result(
    results: List[RetrievalResult],
) -> List[MemoryRetrievalResult]:
    memory_results = []
    for r in results:
        memory_results.append(
            MemoryRetrievalResult(
                rank=r.rank,
                distance=r.distance,
                chunk_text=r.chunk_text,
                conversation_id=r.conversation_id,
                conversation_title=r.conversation_title,
                message_id=r.message_id,
                message_role=r.message_role,
                message_timestamp=r.message_timestamp,
            )
        )
    return memory_results


def evaluate_strategy(
    queries: List[Dict[str, Any]], strategy: str, alpha: float = 0.5
) -> MemoryEvaluationSummary:
    query_metrics = []

    if strategy == "semantic":
        retriever = VectorRetriever()
    elif strategy == "lexical":
        retriever = LexicalRetriever()
    else:
        retriever = HybridRetriever()

    print(f"\n[*] Evaluating strategy: {strategy} (alpha={alpha})")

    start_time = time.time()

    for i, query_dict in enumerate(queries, start=1):
        query_text = query_dict.get("query", "")
        expected_phrase = query_dict.get("expected_phrase", "")
        query_type = query_dict.get("query_type")

        if not query_text or not expected_phrase:
            continue

        if strategy == "semantic":
            raw_results = retriever.retrieve(query_text, k=50)
        elif strategy == "lexical":
            raw_results = retriever.retrieve(query_text, k=50)
        elif strategy == "hybrid_rrf":
            raw_results = retriever.retrieve(query_text, k=50, fusion_strategy="rrf")
        elif strategy.startswith("hybrid_linear"):
            raw_results = retriever.retrieve(
                query_text, k=50, fusion_strategy="linear", alpha=alpha
            )

        retrieved = convert_to_memory_result(raw_results)

        metrics = MemoryMetricsCalculator.evaluate_memory_query(
            query_text, expected_phrase, retrieved, query_type
        )
        query_metrics.append(metrics)

    elapsed = time.time() - start_time

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

    print(
        f"    -> Done in {elapsed:.2f}s | Hit@1: {summary.memory_hit_at_1_rate:.2%} | MRR: {summary.avg_phrase_mrr:.4f} | Latency/q: {(elapsed/len(query_metrics))*1000:.1f}ms"
    )

    return summary, elapsed


def main():
    queries_path = Path(__file__).parent.parent / "data" / "memory_queries.json"
    queries = load_memory_queries(str(queries_path))

    strategies = [
        ("semantic", 0.0),
        ("lexical", 0.0),
        ("hybrid_rrf", 0.0),
        ("hybrid_linear_50_50", 0.5),
        ("hybrid_linear_75_25", 0.75),
        ("hybrid_linear_25_75", 0.25),
    ]

    results = {}

    for strategy_name, alpha in strategies:
        summary, elapsed = evaluate_strategy(queries, strategy_name, alpha)

        results[strategy_name] = {
            "hit_at_1": summary.memory_hit_at_1_rate,
            "hit_at_3": summary.memory_hit_at_3_rate,
            "hit_at_5": summary.memory_hit_at_5_rate,
            "hit_at_10": summary.memory_hit_at_10_rate,
            "mrr": summary.avg_phrase_mrr,
            "latency_ms_per_query": (elapsed / summary.total_queries) * 1000,
            "per_query": [
                {
                    "query": m.query,
                    "type": m.query_type,
                    "rank": m.phrase_rank,
                    "hit_at_10": m.memory_hit_at_10,
                }
                for m in summary.per_query_metrics
            ],
        }

    # Failure Analysis: Compare queries that changed rank
    print("\n" + "=" * 50)
    print("FAILURE & RANK SHIFT ANALYSIS (Semantic vs Hybrid RRF)")
    print("=" * 50)

    semantic_queries = {m["query"]: m for m in results["semantic"]["per_query"]}
    hybrid_queries = {m["query"]: m for m in results["hybrid_rrf"]["per_query"]}
    lexical_queries = {m["query"]: m for m in results["lexical"]["per_query"]}

    for q_text, sem_m in semantic_queries.items():
        hyb_m = hybrid_queries[q_text]
        lex_m = lexical_queries[q_text]

        sem_rank = sem_m["rank"] or 999
        hyb_rank = hyb_m["rank"] or 999
        lex_rank = lex_m["rank"] or 999

        if sem_rank != hyb_rank:
            if hyb_rank < sem_rank:
                print(f"[IMPROVED] Type: {sem_m['type']}")
                print(f"  Q: {q_text}")
                print(
                    f"  Rank: Semantic={sem_rank} -> Hybrid={hyb_rank} (Lexical={lex_rank})"
                )
            elif hyb_rank > sem_rank:
                print(f"[WORSENED] Type: {sem_m['type']}")
                print(f"  Q: {q_text}")
                print(
                    f"  Rank: Semantic={sem_rank} -> Hybrid={hyb_rank} (Lexical={lex_rank})"
                )

    with open("hybrid_benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nSaved full results to hybrid_benchmark_results.json")


if __name__ == "__main__":
    main()
