import json
import time
from pathlib import Path

from Weft.core.retrieval import HybridRetriever, LexicalRetriever, VectorRetriever
from Weft.evaluation.core.memory_metrics import (
    MemoryEvaluationSummary,
    MemoryMetricsCalculator,
    MemoryRetrievalResult,
)


def main():  # noqa: C901
    data_path = Path("Weft/evaluation/data/memory_queries.json")
    with open(data_path, "r") as f:
        queries = json.load(f)

    strategies = {
        "pure_semantic": lambda: VectorRetriever(),
        "pure_lexical": lambda: LexicalRetriever(),
        "hybrid_rrf": lambda: HybridRetriever(),
        "hybrid_linear_0.5": lambda: HybridRetriever(),
    }

    results = {}

    for name, init_retriever in strategies.items():
        print(f"\n--- Benchmarking Strategy: {name} ---")
        retriever = init_retriever()

        eval_summary = MemoryEvaluationSummary(total_queries=len(queries))
        eval_summary.per_query_metrics = []

        total_time = 0
        for q_idx, q in enumerate(queries):
            query_text = q["query"]
            expected = q["expected_phrase"]

            start_time = time.time()
            if name == "pure_semantic":
                raw_results = retriever.retrieve(query_text, k=20)
            elif name == "pure_lexical":
                raw_results = retriever.retrieve(query_text, k=20)
            elif name == "hybrid_rrf":
                raw_results = retriever.retrieve(
                    query_text, k=20, fusion_strategy="rrf"
                )
            elif name == "hybrid_linear_0.5":
                raw_results = retriever.retrieve(
                    query_text, k=20, fusion_strategy="linear", alpha=0.5
                )

            total_time += time.time() - start_time

            # Convert to MemoryRetrievalResult
            mem_results = []
            for r in raw_results:
                mem_results.append(
                    MemoryRetrievalResult(
                        rank=r.rank,
                        distance=r.distance,
                        chunk_text=r.chunk_text,
                        conversation_id=r.conversation_id,
                        message_id=r.message_id,
                    )
                )

            metrics = MemoryMetricsCalculator.evaluate_memory_query(
                query=query_text,
                expected_phrase=expected,
                retrieved_chunks=mem_results,
                query_type=q.get("query_type", "general"),
            )

            eval_summary.per_query_metrics.append(metrics)

            if metrics.memory_hit_at_1:
                eval_summary.queries_with_hits_at_1 += 1
            if metrics.memory_hit_at_3:
                eval_summary.queries_with_hits_at_3 += 1
            if metrics.memory_hit_at_5:
                eval_summary.queries_with_hits_at_5 += 1
            if metrics.memory_hit_at_10:
                eval_summary.queries_with_hits_at_10 += 1

            if metrics.candidate_recall_at_10:
                eval_summary.queries_with_candidate_recall_at_10 += 1
            if metrics.candidate_recall_at_20:
                eval_summary.queries_with_candidate_recall_at_20 += 1
            if metrics.candidate_recall_at_50:
                eval_summary.queries_with_candidate_recall_at_50 += 1

            eval_summary.avg_phrase_mrr += metrics.phrase_mrr

        eval_summary.avg_phrase_mrr /= len(queries)
        eval_summary.compute_rates()

        avg_latency = (total_time / len(queries)) * 1000

        results[name] = {
            "hit_at_1": eval_summary.memory_hit_at_1_rate,
            "hit_at_5": eval_summary.memory_hit_at_5_rate,
            "hit_at_10": eval_summary.memory_hit_at_10_rate,
            "mrr": eval_summary.avg_phrase_mrr,
            "latency_ms": avg_latency,
            "queries": [
                {
                    "query": m.query,
                    "expected": m.expected_phrase,
                    "type": m.query_type,
                    "hit_at_1": m.memory_hit_at_1,
                    "hit_at_10": m.memory_hit_at_10,
                    "rank": m.phrase_rank,
                }
                for m in eval_summary.per_query_metrics
            ],
        }

        print(f"Hit@1:  {eval_summary.memory_hit_at_1_rate:.2%}")
        print(f"Hit@5:  {eval_summary.memory_hit_at_5_rate:.2%}")
        print(f"Hit@10: {eval_summary.memory_hit_at_10_rate:.2%}")
        print(f"MRR:    {eval_summary.avg_phrase_mrr:.4f}")
        print(f"Avg Latency: {avg_latency:.1f}ms")

    out_path = Path("Weft/evaluation/reports/hybrid_benchmark_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
