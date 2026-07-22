"""Memory-based retrieval metrics.

Measures whether the system retrieves the CORRECT MEMORY (phrase-based)
rather than just topically related content (keyword-based).
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class MemoryRetrievalResult:
    """Single retrieval result with metadata for diagnosis."""

    rank: int
    distance: float
    chunk_text: str
    conversation_id: str
    conversation_title: Optional[str] = None
    message_id: Optional[str] = None
    message_role: Optional[str] = None
    message_timestamp: Optional[str] = None
    phrase_found: bool = False
    phrase_rank: Optional[int] = None  # Rank where phrase was found


@dataclass
class MemoryQueryMetrics:
    """Metrics for memory retrieval evaluation."""

    query: str
    expected_phrase: str
    query_type: Optional[str] = None

    # Memory hit metrics (phrase-based, not keyword-based)
    memory_hit_at_1: bool = False
    memory_hit_at_3: bool = False
    memory_hit_at_5: bool = False
    memory_hit_at_10: bool = False

    # Candidate recall (how far down is the correct phrase?)
    candidate_recall_at_10: bool = False
    candidate_recall_at_20: bool = False
    candidate_recall_at_50: bool = False

    # Ranking quality
    phrase_mrr: float = 0.0  # 1 / rank where phrase found, 0 if not found
    phrase_rank: Optional[int] = None  # First rank where phrase found

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

    avg_phrase_mrr: float = 0.0
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
    """Calculate memory retrieval metrics."""

    @staticmethod
    def normalize_phrase(phrase: str) -> str:
        """Normalize phrase for matching (lowercase, strip whitespace)."""
        return phrase.lower().strip()

    @staticmethod
    def phrase_in_text(phrase: str, text: str, case_sensitive: bool = False) -> bool:
        """Check if phrase appears in text.

        Args:
            phrase: Phrase to search for
            text: Text to search in
            case_sensitive: If True, require exact case match

        Returns:
            True if phrase found
        """
        if case_sensitive:
            return phrase in text
        else:
            return phrase.lower() in text.lower()

    @staticmethod
    def find_phrase_rank(
        phrase: str, retrieved_chunks: List[MemoryRetrievalResult], max_rank: int = 50
    ) -> Optional[int]:
        """Find the rank of first occurrence of phrase in retrieved chunks.

        Args:
            phrase: Phrase to find
            retrieved_chunks: List of retrieved chunks ordered by rank
            max_rank: Maximum rank to search (return None if not found within this)

        Returns:
            Rank (1-indexed) where phrase found, or None if not found
        """
        normalized_phrase = MemoryMetricsCalculator.normalize_phrase(phrase)

        for chunk in retrieved_chunks[:max_rank]:
            if MemoryMetricsCalculator.phrase_in_text(
                normalized_phrase, chunk.chunk_text
            ):
                return chunk.rank

        return None

    @staticmethod
    def calculate_memory_hit_at_k(
        retrieved_chunks: List[MemoryRetrievalResult], expected_phrase: str, k: int
    ) -> bool:
        """Calculate MemoryHit@k: is phrase in top-k results?

        Args:
            retrieved_chunks: List of retrieved chunks (must be ordered by rank)
            expected_phrase: Phrase to search for
            k: Cutoff rank

        Returns:
            True if phrase found in top-k
        """
        rank = MemoryMetricsCalculator.find_phrase_rank(
            expected_phrase, retrieved_chunks, max_rank=k
        )
        return rank is not None

    @staticmethod
    def calculate_phrase_mrr(
        retrieved_chunks: List[MemoryRetrievalResult], expected_phrase: str
    ) -> float:
        """Calculate MRR for phrase position.

        MRR = 1 / rank of first relevant result
        MRR = 0 if phrase not found

        Args:
            retrieved_chunks: List of retrieved chunks
            expected_phrase: Phrase to search for

        Returns:
            MRR value (0.0 to 1.0)
        """
        rank = MemoryMetricsCalculator.find_phrase_rank(
            expected_phrase, retrieved_chunks, max_rank=50
        )

        if rank is None:
            return 0.0
        else:
            return 1.0 / rank

    @staticmethod
    def evaluate_memory_query(
        query: str,
        expected_phrase: str,
        retrieved_chunks: List[MemoryRetrievalResult],
        query_type: Optional[str] = None,
    ) -> MemoryQueryMetrics:
        """Evaluate a single query using memory-based metrics.

        Args:
            query: Query string
            expected_phrase: Expected phrase to find
            retrieved_chunks: Retrieved results (must include top-50)
            query_type: Optional query type for categorization

        Returns:
            MemoryQueryMetrics with all metrics calculated
        """
        # Find rank where phrase first appears (within top 50)
        phrase_rank = MemoryMetricsCalculator.find_phrase_rank(
            expected_phrase, retrieved_chunks, max_rank=50
        )

        # Calculate memory hits
        memory_hit_at_1 = MemoryMetricsCalculator.calculate_memory_hit_at_k(
            retrieved_chunks, expected_phrase, 1
        )
        memory_hit_at_3 = MemoryMetricsCalculator.calculate_memory_hit_at_k(
            retrieved_chunks, expected_phrase, 3
        )
        memory_hit_at_5 = MemoryMetricsCalculator.calculate_memory_hit_at_k(
            retrieved_chunks, expected_phrase, 5
        )
        memory_hit_at_10 = MemoryMetricsCalculator.calculate_memory_hit_at_k(
            retrieved_chunks, expected_phrase, 10
        )

        # Calculate candidate recall (how far down does it appear?)
        candidate_recall_at_10 = phrase_rank is not None and phrase_rank <= 10
        candidate_recall_at_20 = phrase_rank is not None and phrase_rank <= 20
        candidate_recall_at_50 = phrase_rank is not None and phrase_rank <= 50

        # Calculate MRR
        phrase_mrr = MemoryMetricsCalculator.calculate_phrase_mrr(
            retrieved_chunks, expected_phrase
        )

        # Determine if query passed
        has_hit = memory_hit_at_10  # Success if found in top-10

        return MemoryQueryMetrics(
            query=query,
            expected_phrase=expected_phrase,
            query_type=query_type,
            memory_hit_at_1=memory_hit_at_1,
            memory_hit_at_3=memory_hit_at_3,
            memory_hit_at_5=memory_hit_at_5,
            memory_hit_at_10=memory_hit_at_10,
            candidate_recall_at_10=candidate_recall_at_10,
            candidate_recall_at_20=candidate_recall_at_20,
            candidate_recall_at_50=candidate_recall_at_50,
            phrase_mrr=phrase_mrr,
            phrase_rank=phrase_rank,
            retrieved_chunks=retrieved_chunks,
            has_hit=has_hit,
        )
