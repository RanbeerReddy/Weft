"""Core retrieval pipeline for Weft.

This module provides reusable components for vector search and cross-encoder reranking.
"""

from typing import List, Optional
import time

from sentence_transformers import SentenceTransformer, CrossEncoder
from sqlalchemy import select
from sqlalchemy.orm import Session

from Weft.storage.database import SessionLocal
from Weft.storage.models import Embedding, Chunk
from Weft.evaluation.metrics import RetrievalResult


class VectorRetriever:
    """Handles top-k vector retrieval using pgvector."""
    
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        print(f"[*] Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        
    def retrieve(self, query: str, k: int = 10, session: Optional[Session] = None) -> List[RetrievalResult]:
        """Retrieve top-k chunks for a query from the database.
        
        Args:
            query: The search string.
            k: Number of results to return.
            session: Optional SQLAlchemy session. If None, creates a temporary one.
            
        Returns:
            List of RetrievalResult objects ranked by cosine distance.
        """
        db = session or SessionLocal()
        
        try:
            # 1. Encode query
            vector_embedding = self.model.encode(query, normalize_embeddings=True).tolist()
            
            # 2. Compute distance attribute for pgvector
            distance_attr = Embedding.embedding_vector.cosine_distance(vector_embedding)
            
            # 3. Build query
            stmt = (
                select(
                    Embedding.conversation_id,
                    Embedding.message_id,
                    Embedding.chunk_order,
                    Chunk.chunk_text,
                    distance_attr.label("distance")
                )
                .join(Chunk, Chunk.id == Embedding.chunk_order)
                .order_by("distance")
                .limit(k)
            )
            
            # 4. Execute
            results = db.execute(stmt).fetchall()
            
            # 5. Convert to RetrievalResult
            retrieved = []
            for rank, row in enumerate(results, start=1):
                retrieved.append(
                    RetrievalResult(
                        chunk_text=row.chunk_text,
                        distance=float(row.distance),
                        conversation_id=row.conversation_id,
                        message_id=row.message_id,
                        chunk_order=row.chunk_order,
                        rank=rank
                    )
                )
            
            return retrieved
            
        finally:
            if session is None:
                db.close()


class CrossEncoderReranker:
    """Handles cross-encoder reranking of candidate chunks."""
    
    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        print(f"[*] Loading cross-encoder model: {model_name}")
        self.model = CrossEncoder(model_name, max_length=512)
        
    def rerank(self, query: str, candidates: List[RetrievalResult], top_k: int = 10) -> List[RetrievalResult]:
        """Rerank candidates using a cross-encoder.
        
        Args:
            query: The search string.
            candidates: List of RetrievalResult objects from the first stage.
            top_k: Number of top reranked results to return.
            
        Returns:
            List of RetrievalResult objects re-ranked by the cross-encoder. 
            The 'distance' attribute is updated to hold the reranker score (higher is better, 
            so we negate it or document it. For compatibility with MetricsCalculator which expects distance
            where lower is better, we will store -score in distance).
        """
        if not candidates:
            return []
            
        # Format input for cross-encoder as list of (query, document) pairs
        pairs = [[query, candidate.chunk_text] for candidate in candidates]
        
        # Predict scores (returns logits)
        scores = self.model.predict(pairs, batch_size=32)
        
        # Zip candidates with their new scores
        scored_candidates = list(zip(candidates, scores))
        
        # Sort by score descending (higher is better for cross-encoder)
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Take top_k
        top_candidates = scored_candidates[:top_k]
        
        # Rebuild RetrievalResult objects with updated rank and "distance" (using -score for sorting compatibility)
        reranked_results = []
        for new_rank, (candidate, score) in enumerate(top_candidates, start=1):
            reranked_results.append(
                RetrievalResult(
                    chunk_text=candidate.chunk_text,
                    distance=-float(score),  # Negate so lower distance = better rank
                    conversation_id=candidate.conversation_id,
                    message_id=candidate.message_id,
                    chunk_order=candidate.chunk_order,
                    rank=new_rank
                )
            )
            
        return reranked_results


class RetrievalPipeline:
    """Composes vector retrieval and cross-encoder reranking."""
    
    def __init__(self, 
                 retriever: Optional[VectorRetriever] = None, 
                 reranker: Optional[CrossEncoderReranker] = None):
        self.retriever = retriever or VectorRetriever()
        self.reranker = reranker or CrossEncoderReranker()
        
    def search(self, query: str, top_n: int = 100, final_k: int = 10) -> List[RetrievalResult]:
        """Execute the two-stage retrieval pipeline.
        
        Args:
            query: The search string.
            top_n: Number of candidates to fetch using vector search.
            final_k: Number of results to return after reranking.
            
        Returns:
            Final list of top-k RetrievalResult objects.
        """
        # Stage 1: Fast vector retrieval
        candidates = self.retriever.retrieve(query, k=top_n)
        
        # Stage 2: Cross-encoder reranking
        final_results = self.reranker.rerank(query, candidates, top_k=final_k)
        
        return final_results
