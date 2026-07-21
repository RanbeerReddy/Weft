"""Core retrieval pipeline for Weft.

This module provides reusable components for vector search and cross-encoder reranking.
"""

from typing import List, Optional
import time

from sentence_transformers import SentenceTransformer, CrossEncoder
from sqlalchemy import select, text, func
from sqlalchemy.orm import Session

from Weft.storage.database import SessionLocal
from Weft.storage.models import Embedding, Chunk, Conversation, Message
from Weft.evaluation.core.metrics import RetrievalResult
from datetime import datetime


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
            
            stmt = (
                select(
                    Embedding.conversation_id,
                    Embedding.message_id,
                    Embedding.chunk_order,
                    Chunk.chunk_text,
                    distance_attr.label("distance"),
                    Conversation.title,
                    Message.role,
                    Message.create_time
                )
                .join(Chunk, Chunk.id == Embedding.chunk_order)
                .join(Message, Message.id == Embedding.message_id)
                .join(Conversation, Conversation.id == Embedding.conversation_id)
                .order_by("distance")
                .limit(k)
            )
            
            # 4. Execute
            results = db.execute(stmt).fetchall()
            
            # 5. Convert to RetrievalResult
            retrieved = []
            for rank, row in enumerate(results, start=1):
                timestamp = None
                if row.create_time:
                    try:
                        ts = row.create_time
                        if isinstance(ts, (int, float)):
                            timestamp = datetime.fromtimestamp(ts).isoformat()
                        else:
                            timestamp = str(ts)
                    except Exception:
                        timestamp = str(row.create_time)
                        
                retrieved.append(
                    RetrievalResult(
                        chunk_text=row.chunk_text,
                        distance=float(row.distance),
                        conversation_id=row.conversation_id,
                        message_id=row.message_id,
                        chunk_order=row.chunk_order,
                        rank=rank,
                        conversation_title=row.title,
                        message_role=row.role,
                        message_timestamp=timestamp
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
                    rank=new_rank,
                    conversation_title=candidate.conversation_title,
                    message_role=candidate.message_role,
                    message_timestamp=candidate.message_timestamp
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


class LexicalRetriever:
    """Handles PostgreSQL FTS (BM25) using tsvector."""
    
    def retrieve(self, query: str, k: int = 10, session: Optional[Session] = None) -> List[RetrievalResult]:
        db = session or SessionLocal()
        
        if not query.strip():
            return []
            
        try:
            ts_query = func.websearch_to_tsquery('english', query)
            stmt = (
                select(
                    Chunk.conversation_id,
                    Chunk.message_id,
                    Chunk.chunk_order,
                    Chunk.chunk_text,
                    func.ts_rank_cd(Chunk.chunk_tsvector, ts_query).label('rank_score'),
                    Conversation.title,
                    Message.role,
                    Message.create_time
                )
                .join(Message, Message.id == Chunk.message_id)
                .join(Conversation, Conversation.id == Chunk.conversation_id)
                .where(Chunk.chunk_tsvector.op('@@')(ts_query))
                .order_by(text('rank_score DESC'))
                .limit(k)
            )
            
            results = db.execute(stmt).fetchall()
            
            retrieved = []
            for rank, row in enumerate(results, start=1):
                timestamp = None
                if row.create_time:
                    try:
                        ts = row.create_time
                        if isinstance(ts, (int, float)):
                            timestamp = datetime.fromtimestamp(ts).isoformat()
                        else:
                            timestamp = str(ts)
                    except Exception:
                        timestamp = str(row.create_time)
                        
                retrieved.append(
                    RetrievalResult(
                        chunk_text=row.chunk_text,
                        distance=-float(row.rank_score),  # Negative score to maintain "lower is better" sorting
                        conversation_id=row.conversation_id,
                        message_id=row.message_id,
                        chunk_order=row.chunk_order,
                        rank=rank,
                        conversation_title=row.title,
                        message_role=row.role,
                        message_timestamp=timestamp
                    )
                )
            
            return retrieved
            
        finally:
            if session is None:
                db.close()


class HybridRetriever:
    """Fuses results from VectorRetriever and LexicalRetriever."""
    
    def __init__(self, vector_retriever: Optional[VectorRetriever] = None, lexical_retriever: Optional[LexicalRetriever] = None):
        self.vector_retriever = vector_retriever or VectorRetriever()
        self.lexical_retriever = lexical_retriever or LexicalRetriever()
        
    def retrieve(self, query: str, k: int = 10, fusion_strategy: str = "rrf", alpha: float = 0.5, session: Optional[Session] = None) -> List[RetrievalResult]:
        fetch_k = max(k * 2, 60)
        
        vec_results = self.vector_retriever.retrieve(query, k=fetch_k, session=session)
        lex_results = self.lexical_retriever.retrieve(query, k=fetch_k, session=session)
        
        if fusion_strategy == "rrf":
            return self._rrf_fusion(vec_results, lex_results, k=k)
        elif fusion_strategy == "linear":
            return self._linear_fusion(vec_results, lex_results, alpha=alpha, k=k)
        else:
            raise ValueError(f"Unknown fusion strategy: {fusion_strategy}")
            
    def _rrf_fusion(self, vec_results, lex_results, k=10, rrf_k=60):
        scores = {}
        chunks = {}
        
        def get_key(r): return f"{r.conversation_id}_{r.message_id}_{r.chunk_order}"
        
        for rank, r in enumerate(vec_results, start=1):
            key = get_key(r)
            if key not in scores:
                scores[key] = 0.0
                chunks[key] = r
            scores[key] += 1.0 / (rrf_k + rank)
            
        for rank, r in enumerate(lex_results, start=1):
            key = get_key(r)
            if key not in scores:
                scores[key] = 0.0
                chunks[key] = r
            scores[key] += 1.0 / (rrf_k + rank)
            
        sorted_keys = sorted(scores.keys(), key=lambda k_id: scores[k_id], reverse=True)
        
        final_results = []
        for rank, key in enumerate(sorted_keys[:k], start=1):
            r = chunks[key]
            final_results.append(
                RetrievalResult(
                    chunk_text=r.chunk_text,
                    distance=-scores[key],
                    conversation_id=r.conversation_id,
                    message_id=r.message_id,
                    chunk_order=r.chunk_order,
                    rank=rank,
                    conversation_title=r.conversation_title,
                    message_role=r.message_role,
                    message_timestamp=r.message_timestamp
                )
            )
            
        return final_results
        
    def _linear_fusion(self, vec_results, lex_results, alpha=0.5, k=10):
        scores = {}
        chunks = {}
        
        def get_key(r): return f"{r.conversation_id}_{r.message_id}_{r.chunk_order}"
        
        vec_min = min((r.distance for r in vec_results), default=0)
        vec_max = max((r.distance for r in vec_results), default=1)
        if vec_max == vec_min: vec_max = vec_min + 1e-5
        
        for r in vec_results:
            key = get_key(r)
            norm_score = 1.0 - ((r.distance - vec_min) / (vec_max - vec_min))
            scores[key] = alpha * norm_score
            chunks[key] = r
            
        lex_min = min((r.distance for r in lex_results), default=0)
        lex_max = max((r.distance for r in lex_results), default=1)
        if lex_max == lex_min: lex_max = lex_min + 1e-5
        
        for r in lex_results:
            key = get_key(r)
            norm_score = 1.0 - ((r.distance - lex_min) / (lex_max - lex_min))
            if key not in scores:
                scores[key] = 0.0
                chunks[key] = r
            scores[key] += (1.0 - alpha) * norm_score
            
        sorted_keys = sorted(scores.keys(), key=lambda k_id: scores[k_id], reverse=True)
        
        final_results = []
        for rank, key in enumerate(sorted_keys[:k], start=1):
            r = chunks[key]
            final_results.append(
                RetrievalResult(
                    chunk_text=r.chunk_text,
                    distance=-scores[key],
                    conversation_id=r.conversation_id,
                    message_id=r.message_id,
                    chunk_order=r.chunk_order,
                    rank=rank,
                    conversation_title=r.conversation_title,
                    message_role=r.message_role,
                    message_timestamp=r.message_timestamp
                )
            )
            
        return final_results
