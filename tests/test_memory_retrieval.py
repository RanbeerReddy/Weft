# Ensure paths are correct
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Weft.core.context_assembler import assemble_context
from Weft.storage.models import Memory, MemoryType


class TestMemoryRetrieval(unittest.TestCase):
    @patch("Weft.core.context_assembler.SessionLocal")
    @patch("Weft.core.context_assembler.SentenceTransformer")
    def test_context_assembler_logic(self, mock_transformer_class, mock_session_local):
        # 1. Setup mock embeddings
        # A simple fake encode that maps queries to vectors
        import numpy as np

        def fake_encode(text, **kwargs):
            if "internship did I get selected for" in text:
                return np.array([1.0, 0.0, 0.0] * 128)  # Target query
            elif "What database does Weft use" in text:
                return np.array([0.0, 1.0, 0.0] * 128)  # Unrelated query
            elif "What do I need to work on next" in text:
                return np.array([0.0, 0.0, 1.0] * 128)  # Stop word query
            return np.array([0.0, 0.0, 0.0] * 128)

        mock_model = MagicMock()
        mock_model.encode.side_effect = fake_encode
        mock_transformer_class.return_value = mock_model

        # 2. Setup mock memories
        mem_sensovibe = Memory(
            id="m1",
            type_id="t1",
            value={"event": "Selected for Sensovibe internship"},
            embedding_vector=[1.0, 0.0, 0.0] * 128,  # Exact match for query 1
        )
        mem_generic = Memory(
            id="m2",
            type_id="t1",
            value={"event": "Discussed several internship opportunities"},
            embedding_vector=[0.5, 0.5, 0.0]
            * 128,  # Distance = sum((0.5)^2 + (-0.5)^2)*128 = 0.5*128 = 64 > 1.0
        )
        mem_duplicate = Memory(
            id="m1",  # Same ID, simulates deduplication test
            type_id="t1",
            value={"event": "Selected for Sensovibe internship"},
            embedding_vector=[1.0, 0.0, 0.0] * 128,
        )

        # We will mock the database session methods
        mock_db = MagicMock()
        mock_session_local.return_value = mock_db

        # Handle the chaining: db.scalars().all()
        # The first call in assemble_context is for Preferences, second for Query-Specific Memories

        # First call is for Preferences (return empty list)
        # Second call is for Query-Specific Memories
        mock_prefs = MagicMock()
        mock_prefs.all.return_value = []

        mock_queries = MagicMock()
        mock_queries.all.return_value = [mem_sensovibe, mem_generic, mem_duplicate]

        # We need it to return mock_prefs then mock_queries for each assemble_context call
        # Since we call assemble_context 3 times, we need 6 returns total
        mock_db.scalars.side_effect = [
            mock_prefs,
            mock_queries,
            mock_prefs,
            mock_queries,
            mock_prefs,
            mock_queries,
        ]

        mock_db.execute.return_value.all.return_value = (
            []
        )  # Mock conversation history to empty

        # Mock db.get for MemoryType
        def fake_get(model, id):
            t = MemoryType(id="t1", name="Experience")
            return t

        mock_db.get.side_effect = fake_get

        # Test 1: Relevant memory
        context1 = assemble_context("What internship did I get selected for?")
        self.assertIn("Sensovibe internship", context1)
        self.assertNotIn(
            "Discussed several internship", context1
        )  # Threshold should filter this
        self.assertEqual(
            context1.count("Sensovibe internship"), 1
        )  # Deduplication check

        # Test 2: Unrelated query
        context2 = assemble_context("What database does Weft use?")
        self.assertNotIn("Sensovibe internship", context2)

        # Test 3: Stop-word heavy
        context3 = assemble_context("What do I need to work on next?")
        self.assertNotIn("Sensovibe internship", context3)
        self.assertNotIn("Discussed several internship", context3)


if __name__ == "__main__":
    unittest.main()
