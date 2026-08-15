"""Memory-based retrieval metrics.

Measures whether the system retrieves the CORRECT MEMORY (ID-based)
rather than just topically related content (keyword/phrase-based).
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class MemoryRetrievalResult:
    """Single retrieval result with metadata for diagnosis."""

    rank: int
    distance: float
    chunk_text: str
    memory_id: str
    conversation_id: str
    conversation_title: Optional[str] = None
    message_id: Optional[str] = None
    message_role: Optional[str] = None
    message_timestamp: Optional[str] = None
    is_hit: bool = False


@dataclass
class MemoryQueryMetrics:
    """Metrics for memory retrieval evaluation."""

    query: str
    expected_id: str
    query_type: Optional[str] = None

    # Memory hit metrics (ID-based)
    memory_hit_at_1: bool = False
    memory_hit_at_3: bool = False
    memory_hit_at_5: bool = False
    memory_hit_at_10: bool = False

    # Candidate recall (how far down is the correct memory?)
    candidate_recall_at_10: bool = False
    candidate_recall_at_20: bool = False
    candidate_recall_at_50: bool = False

    # Ranking quality
    memory_mrr: float = 0.0  # 1 / rank where memory found, 0 if not found
    memory_rank: Optional[int] = None  # First rank where memory found

    # Diagnostics
    retrieved_chunks: Optional[List[MemoryRetrievalResult]] = None
    has_hit: bool = False


@dataclass
class MemoryEvaluationSummary:
    """Aggregated memory evaluation metrics."""

    total_queries: int = 0
    queries_with_hits_at_1: int = 0
    queries_with_hits_at_3: int = 0
    queries_with_hits_at_5: int = 0
    queries_with_hits_at_10: int = 0
    queries_with_candidate_recall_at_10: int = 0
    queries_with_candidate_recall_at_20: int = 0
    queries_with_candidate_recall_at_50: int = 0

    avg_memory_mrr: float = 0.0
    per_query_metrics: Optional[List[MemoryQueryMetrics]] = None

    def compute_rates(self):
        """Calculate success rates."""
        if self.total_queries == 0:
            return

        self.memory_hit_at_1_rate = self.queries_with_hits_at_1 / self.total_queries
        self.memory_hit_at_3_rate = self.queries_with_hits_at_3 / self.total_queries
        self.memory_hit_at_5_rate = self.queries_with_hits_at_5 / self.total_queries
        self.memory_hit_at_10_rate = self.queries_with_hits_at_10 / self.total_queries

        self.candidate_recall_at_10_rate = (
            self.queries_with_candidate_recall_at_10 / self.total_queries
        )
        self.candidate_recall_at_20_rate = (
            self.queries_with_candidate_recall_at_20 / self.total_queries
        )
        self.candidate_recall_at_50_rate = (
            self.queries_with_candidate_recall_at_50 / self.total_queries
        )


class MemoryMetricsCalculator:
    """Calculate memory retrieval metrics using stable ground-truth IDs."""

    @staticmethod
    def find_memory_rank(
        expected_id: str, retrieved_chunks: List[MemoryRetrievalResult], max_rank: int = 50
    ) -> Optional[int]:
        """Find the rank of first occurrence of the expected ID in retrieved chunks.

        Args:
            expected_id: Expected memory ID
            retrieved_chunks: List of retrieved chunks ordered by rank
            max_rank: Maximum rank to search (return None if not found within this)

        Returns:
            Rank (1-indexed) where memory found, or None if not found
        """
        for chunk in retrieved_chunks[:max_rank]:
            if chunk.memory_id == expected_id:
                return chunk.rank
        return None

    @staticmethod
    def calculate_memory_hit_at_k(
        retrieved_chunks: List[MemoryRetrievalResult], expected_id: str, k: int
    ) -> bool:
        """Calculate MemoryHit@k: is the expected memory in top-k results?

        Args:
            retrieved_chunks: List of retrieved chunks (must be ordered by rank)
            expected_id: ID to search for
            k: Cutoff rank

        Returns:
            True if memory found in top-k
        """
        rank = MemoryMetricsCalculator.find_memory_rank(
            expected_id, retrieved_chunks, max_rank=k
        )
        return rank is not None

    @staticmethod
    def calculate_memory_mrr(
        retrieved_chunks: List[MemoryRetrievalResult], expected_id: str
    ) -> float:
        """Calculate MRR for memory position.

        MRR = 1 / rank of first relevant result
        MRR = 0 if memory not found

        Args:
            retrieved_chunks: List of retrieved chunks
            expected_id: Expected memory ID to search for

        Returns:
            MRR value (0.0 to 1.0)
        """
        rank = MemoryMetricsCalculator.find_memory_rank(
            expected_id, retrieved_chunks, max_rank=50
        )

        if rank is None:
            return 0.0
        else:
            return 1.0 / rank

    @staticmethod
    def evaluate_memory_query(
        query: str,
        expected_id: str,
        retrieved_chunks: List[MemoryRetrievalResult],
        query_type: Optional[str] = None,
    ) -> MemoryQueryMetrics:
        """Evaluate a single query using memory-based metrics.

        Args:
            query: Query string
            expected_id: Expected memory ID to find
            retrieved_chunks: Retrieved results (must include top-50)
            query_type: Optional query type for categorization

        Returns:
            MemoryQueryMetrics with all metrics calculated
        """
        # Find rank where memory first appears (within top 50)
        memory_rank = MemoryMetricsCalculator.find_memory_rank(
            expected_id, retrieved_chunks, max_rank=50
        )

        # Calculate memory hits
        memory_hit_at_1 = MemoryMetricsCalculator.calculate_memory_hit_at_k(
            retrieved_chunks, expected_id, 1
        )
        memory_hit_at_3 = MemoryMetricsCalculator.calculate_memory_hit_at_k(
            retrieved_chunks, expected_id, 3
        )
        memory_hit_at_5 = MemoryMetricsCalculator.calculate_memory_hit_at_k(
            retrieved_chunks, expected_id, 5
        )
        memory_hit_at_10 = MemoryMetricsCalculator.calculate_memory_hit_at_k(
            retrieved_chunks, expected_id, 10
        )

        # Calculate candidate recall (how far down does it appear?)
        candidate_recall_at_10 = memory_rank is not None and memory_rank <= 10
        candidate_recall_at_20 = memory_rank is not None and memory_rank <= 20
        candidate_recall_at_50 = memory_rank is not None and memory_rank <= 50

        # Calculate MRR
        memory_mrr = MemoryMetricsCalculator.calculate_memory_mrr(
            retrieved_chunks, expected_id
        )

        # Determine if query passed
        has_hit = memory_hit_at_10  # Success if found in top-10

        return MemoryQueryMetrics(
            query=query,
            expected_id=expected_id,
            query_type=query_type,
            memory_hit_at_1=memory_hit_at_1,
            memory_hit_at_3=memory_hit_at_3,
            memory_hit_at_5=memory_hit_at_5,
            memory_hit_at_10=memory_hit_at_10,
            candidate_recall_at_10=candidate_recall_at_10,
            candidate_recall_at_20=candidate_recall_at_20,
            candidate_recall_at_50=candidate_recall_at_50,
            memory_mrr=memory_mrr,
            memory_rank=memory_rank,
            retrieved_chunks=retrieved_chunks,
            has_hit=has_hit,
        )
