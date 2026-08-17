import unittest

from Weft.evaluation.core.memory_metrics import (
    MemoryMetricsCalculator,
    MemoryRetrievalResult,
)


class TestMemoryEvaluationMetrics(unittest.TestCase):
    def setUp(self):
        # Create some dummy retrieval results
        self.retrieved_chunks = [
            MemoryRetrievalResult(
                rank=1,
                distance=0.1,
                chunk_text="Selected for Sensovibe internship",
                memory_id="mem-101",
                conversation_id="conv-1",
            ),
            MemoryRetrievalResult(
                rank=2,
                distance=0.2,
                chunk_text="Discussed several internship opportunities",
                memory_id="mem-102",
                conversation_id="conv-1",
            ),
            MemoryRetrievalResult(
                rank=3,
                distance=0.3,
                chunk_text="Learning strategies focus on spaced repetition",
                memory_id="mem-103",
                conversation_id="conv-2",
            ),
            MemoryRetrievalResult(
                rank=4,
                distance=0.4,
                chunk_text="Weft uses PostgreSQL",
                memory_id="mem-106",
                conversation_id="conv-5",
            ),
            MemoryRetrievalResult(
                rank=5,
                distance=0.5,
                chunk_text="Using BAAI/bge-small-en-v1.5",
                memory_id="mem-107",
                conversation_id="conv-6",
            ),
        ]

    def test_exact_ground_truth_memory_hit(self):
        """Test that exact ground-truth memory ID results in a hit."""
        metrics = MemoryMetricsCalculator.evaluate_memory_query(
            query="what internship did I get selected for?",
            expected_id="mem-101",
            retrieved_chunks=self.retrieved_chunks,
        )

        self.assertTrue(metrics.memory_hit_at_1)
        self.assertTrue(metrics.memory_hit_at_3)
        self.assertTrue(metrics.memory_hit_at_5)
        self.assertTrue(metrics.memory_hit_at_10)
        self.assertEqual(metrics.memory_mrr, 1.0)
        self.assertEqual(metrics.memory_rank, 1)

    def test_same_topic_incorrect_memory_miss(self):
        """Test that same-topic but incorrect memory ID results in a miss for top rank."""
        # mem-102 is rank 2, so hit@1 should be False, but hit@3 should be True
        metrics = MemoryMetricsCalculator.evaluate_memory_query(
            query="what internship opportunities did I discuss?",
            expected_id="mem-102",
            retrieved_chunks=self.retrieved_chunks,
        )

        self.assertFalse(metrics.memory_hit_at_1)
        self.assertTrue(metrics.memory_hit_at_3)
        self.assertTrue(metrics.memory_hit_at_5)
        self.assertEqual(metrics.memory_mrr, 0.5)  # Rank 2 = 1/2
        self.assertEqual(metrics.memory_rank, 2)

    def test_rank_5_hit(self):
        """Test ground truth at rank 5 produces correct Hit@5 and 0 for Hit@1."""
        metrics = MemoryMetricsCalculator.evaluate_memory_query(
            query="what embedding model?",
            expected_id="mem-107",
            retrieved_chunks=self.retrieved_chunks,
        )

        self.assertFalse(metrics.memory_hit_at_1)
        self.assertFalse(metrics.memory_hit_at_3)
        self.assertTrue(metrics.memory_hit_at_5)
        self.assertTrue(metrics.memory_hit_at_10)
        self.assertEqual(metrics.memory_mrr, 0.2)  # Rank 5 = 1/5
        self.assertEqual(metrics.memory_rank, 5)

    def test_complete_miss(self):
        """Test memory not in retrieved chunks."""
        metrics = MemoryMetricsCalculator.evaluate_memory_query(
            query="what is my favorite color?",
            expected_id="mem-999",
            retrieved_chunks=self.retrieved_chunks,
        )

        self.assertFalse(metrics.memory_hit_at_1)
        self.assertFalse(metrics.memory_hit_at_5)
        self.assertFalse(metrics.memory_hit_at_10)
        self.assertFalse(metrics.candidate_recall_at_10)
        self.assertFalse(metrics.candidate_recall_at_50)
        self.assertEqual(metrics.memory_mrr, 0.0)
        self.assertIsNone(metrics.memory_rank)

    def test_duplicate_results(self):
        """Duplicate results should not inflate metrics, rank is first occurrence."""
        # Insert a duplicate at rank 3 for mem-101 (which is already rank 1)
        duplicate_chunks = list(self.retrieved_chunks)
        duplicate_chunks.insert(
            2,
            MemoryRetrievalResult(
                rank=3,
                distance=0.15,
                chunk_text="Duplicate chunk text",
                memory_id="mem-101",
                conversation_id="conv-1",
            ),
        )

        metrics = MemoryMetricsCalculator.evaluate_memory_query(
            query="internship?",
            expected_id="mem-101",
            retrieved_chunks=duplicate_chunks,
        )

        # Still rank 1, MRR 1.0
        self.assertEqual(metrics.memory_rank, 1)
        self.assertEqual(metrics.memory_mrr, 1.0)


if __name__ == "__main__":
    unittest.main()
