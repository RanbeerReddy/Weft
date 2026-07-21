"""Phase 4 — Embedding Analysis.

For failed retrievals where data EXISTS (Category B and C):

    1. Find the correct chunk in the DB
    2. Encode the query using the same model
    3. Fetch the correct chunk's stored embedding
    4. Compute cosine_similarity(query_embedding, correct_chunk_embedding)
    5. Compare against the top-1 retrieved chunk's similarity
    6. Report distance gap to determine whether the issue is:
        - Embedding quality (large gap → model doesn't associate concepts)
        - Retrieval/ranking logic (small gap → correct chunk is close but loses)
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from sqlalchemy import func, select

from Weft.storage.database import SessionLocal
from Weft.storage.models import Chunk, Conversation, Embedding

MODEL = None


def get_model() -> SentenceTransformer:
    global MODEL
    if MODEL is None:
        print("[*] Loading embedding model: BAAI/bge-small-en-v1.5")
        MODEL = SentenceTransformer("BAAI/bge-small-en-v1.5")
    return MODEL


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def find_correct_chunks(db, phrase: str) -> List[Dict[str, Any]]:
    """Find all chunks containing the expected phrase."""
    stmt = select(
        Chunk.id,
        Chunk.conversation_id,
        Chunk.message_id,
        Chunk.chunk_text,
    ).where(func.lower(Chunk.chunk_text).contains(phrase.lower()))
    results = db.execute(stmt).fetchall()
    return [
        {
            "chunk_id": r.id,
            "conversation_id": r.conversation_id,
            "message_id": r.message_id,
            "chunk_text": r.chunk_text,
        }
        for r in results
    ]


def get_chunk_embedding(db, chunk_id: int) -> Optional[List[float]]:
    """Fetch the stored embedding vector for a chunk."""
    stmt = select(Embedding.embedding_vector).where(Embedding.chunk_order == chunk_id)
    result = db.execute(stmt).fetchone()
    if result and result[0] is not None:
        return list(result[0])
    return None


def get_top_k_with_embeddings(db, query_vector: List[float], k: int = 10):
    """Retrieve top-k results with their embedding vectors."""
    distance_attr = Embedding.embedding_vector.cosine_distance(query_vector)

    stmt = (
        select(
            Embedding.conversation_id,
            Embedding.message_id,
            Embedding.chunk_order,
            Embedding.embedding_vector,
            Chunk.chunk_text,
            distance_attr.label("distance"),
            Conversation.title,
        )
        .join(Chunk, Chunk.id == Embedding.chunk_order)
        .join(Conversation, Conversation.id == Embedding.conversation_id)
        .order_by("distance")
        .limit(k)
    )
    return db.execute(stmt).fetchall()


def analyze_embedding_for_query(
    db, query: str, expected_phrase: str, query_type: str = None
) -> Dict[str, Any]:
    """Analyze embedding quality for a specific query.

    Returns:
        Dict with distance comparisons and diagnosis
    """
    model = get_model()
    result = {
        "query": query,
        "expected_phrase": expected_phrase,
        "query_type": query_type,
    }

    # 1. Encode query
    query_vector = model.encode(query, normalize_embeddings=True)
    result["query_vector_norm"] = float(np.linalg.norm(query_vector))

    # 2. Find correct chunks
    correct_chunks = find_correct_chunks(db, expected_phrase)
    if not correct_chunks:
        result["status"] = "NO_CORRECT_CHUNK"
        result["explanation"] = "Phrase not found in any chunk (Category A)"
        return result

    result["correct_chunk_count"] = len(correct_chunks)

    # 3. Get embeddings for correct chunks and compute similarities
    correct_similarities = []
    for cc in correct_chunks:
        emb = get_chunk_embedding(db, cc["chunk_id"])
        if emb is not None:
            emb_array = np.array(emb)
            sim = cosine_similarity(query_vector, emb_array)
            dist = 1.0 - sim  # cosine distance
            correct_similarities.append(
                {
                    "chunk_id": cc["chunk_id"],
                    "conversation_id": cc["conversation_id"],
                    "chunk_preview": cc["chunk_text"][:120],
                    "cosine_similarity": round(sim, 6),
                    "cosine_distance": round(dist, 6),
                    "embedding_norm": round(float(np.linalg.norm(emb_array)), 6),
                }
            )

    if not correct_similarities:
        result["status"] = "NO_EMBEDDING_FOR_CORRECT_CHUNK"
        result["explanation"] = "Correct chunk exists but has no embedding"
        return result

    # Sort by similarity (best match first)
    correct_similarities.sort(key=lambda x: x["cosine_similarity"], reverse=True)
    result["correct_chunks"] = correct_similarities

    best_correct = correct_similarities[0]
    result["best_correct_similarity"] = best_correct["cosine_similarity"]
    result["best_correct_distance"] = best_correct["cosine_distance"]

    # 4. Get top-1 retrieved result
    top_results = get_top_k_with_embeddings(db, query_vector.tolist(), k=10)
    if top_results:
        top1 = top_results[0]
        top1_emb = np.array(list(top1.embedding_vector))
        top1_sim = cosine_similarity(query_vector, top1_emb)

        result["top_1"] = {
            "distance": round(float(top1.distance), 6),
            "cosine_similarity": round(top1_sim, 6),
            "chunk_preview": top1.chunk_text[:120] if top1.chunk_text else "",
            "conversation_title": top1.title,
        }

        # 5. Compute gap
        gap = best_correct["cosine_distance"] - float(top1.distance)
        result["distance_gap"] = round(gap, 6)
        result["similarity_gap"] = round(
            top1_sim - best_correct["cosine_similarity"], 6
        )

        # 6. Diagnosis
        if gap < 0.02:
            result["diagnosis"] = "CLOSE_BUT_OUTRANKED"
            result["diagnosis_detail"] = (
                f"Correct chunk distance ({best_correct['cosine_distance']:.4f}) "
                f"is very close to top-1 ({float(top1.distance):.4f}). "
                f"Gap={gap:.4f}. A reranker could fix this."
            )
        elif gap < 0.10:
            result["diagnosis"] = "MODERATE_EMBEDDING_GAP"
            result["diagnosis_detail"] = (
                f"Moderate gap ({gap:.4f}) between correct chunk and top-1. "
                f"Hybrid search or query expansion might help."
            )
        else:
            result["diagnosis"] = "LARGE_EMBEDDING_GAP"
            result["diagnosis_detail"] = (
                f"Large gap ({gap:.4f}) between correct chunk and top-1. "
                f"The embedding model fundamentally fails to associate "
                f"this query with the correct chunk."
            )
    else:
        result["top_1"] = None
        result["diagnosis"] = "NO_RESULTS"

    result["status"] = "ANALYZED"
    return result


def run_embedding_analysis(
    failure_report_path: str = None,
    memory_queries_path: str = None,
) -> Dict[str, Any]:
    """Run embedding analysis for all failed queries where data exists."""
    print("=" * 70)
    print("PHASE 4 — EMBEDDING ANALYSIS")
    print("=" * 70)

    report = {"timestamp": datetime.now().isoformat()}

    # Load failure analysis to identify B and C failures
    if failure_report_path is None:
        failure_report_path = "failure_analysis_report.json"

    queries_to_analyze = []

    fa_path = Path(failure_report_path)
    if fa_path.exists():
        with open(fa_path, "r", encoding="utf-8") as f:
            fa_data = json.load(f)
        for c in fa_data.get("classifications", []):
            if c.get("category") in ("B", "C"):
                queries_to_analyze.append(c)
        print(f"[+] Found {len(queries_to_analyze)} Category B/C failures to analyze")
    else:
        # Fall back to analyzing all queries from memory_queries.json
        print("[!] No failure report found — analyzing ALL memory queries")
        if memory_queries_path is None:
            memory_queries_path = str(Path(__file__).parent / "memory_queries.json")
        with open(memory_queries_path, "r", encoding="utf-8") as f:
            queries = json.load(f)
        for q in queries:
            queries_to_analyze.append(
                {
                    "query": q["query"],
                    "expected_phrase": q["expected_phrase"],
                    "query_type": q.get("query_type"),
                }
            )
        print(f"[+] Analyzing {len(queries_to_analyze)} queries")

    db = SessionLocal()
    try:
        analyses = []
        diagnosis_counts = {}

        for i, q in enumerate(queries_to_analyze, start=1):
            query_text = q.get("query", "")
            expected = q.get("expected_phrase", "")
            qtype = q.get("query_type", "")

            print(f"\n[{i}/{len(queries_to_analyze)}] {query_text}")
            print(f"  Expected: {expected}")

            analysis = analyze_embedding_for_query(db, query_text, expected, qtype)
            analyses.append(analysis)

            diag = analysis.get("diagnosis", "UNKNOWN")
            diagnosis_counts[diag] = diagnosis_counts.get(diag, 0) + 1

            if "diagnosis_detail" in analysis:
                print(f"  {analysis['diagnosis']}: {analysis['diagnosis_detail']}")
            elif "explanation" in analysis:
                print(f"  {analysis['explanation']}")

        report["analyses"] = analyses
        report["diagnosis_counts"] = diagnosis_counts

        # Summary
        print("\n" + "=" * 70)
        print("EMBEDDING ANALYSIS SUMMARY")
        print("=" * 70)
        print(f"\n  Total analyzed: {len(analyses)}")
        for diag, count in sorted(diagnosis_counts.items()):
            print(f"    {diag}: {count}")

        # Compute average gaps
        gaps = [a["distance_gap"] for a in analyses if "distance_gap" in a]
        if gaps:
            print("\n  Distance gap statistics:")
            print(f"    Mean gap:   {sum(gaps)/len(gaps):.4f}")
            print(f"    Min gap:    {min(gaps):.4f}")
            print(f"    Max gap:    {max(gaps):.4f}")

    finally:
        db.close()

    return report


def main():
    parser = argparse.ArgumentParser(description="Weft Embedding Analysis")
    parser.add_argument(
        "--output",
        type=str,
        default="embedding_analysis_report.json",
        help="Output report file",
    )
    args = parser.parse_args()

    report = run_embedding_analysis()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n[+] Report saved to {output_path}")


if __name__ == "__main__":
    main()
