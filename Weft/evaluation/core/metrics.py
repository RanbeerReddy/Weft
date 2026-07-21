"""Retrieval evaluation metrics."""

import re
from typing import List, Dict, Any, Set
from dataclasses import dataclass


@dataclass
class RetrievalResult:
    """Single retrieval result."""
    chunk_text: str
    distance: float
    conversation_id: str
    message_id: str
    chunk_order: int
    rank: int  # position in result set (1-indexed)
    conversation_title: str = None
    message_role: str = None
    message_timestamp: str = None


@dataclass
class QueryMetrics:
    """Metrics for a single query evaluation."""
    query: str
    expected_keywords: List[str]
    retrieved_chunks: List[RetrievalResult]
    
    hit_at_1: bool
    hit_at_3: bool
    hit_at_5: bool
    hit_at_10: bool
    
    keyword_recall: float  # what % of expected keywords found
    keyword_precision: float  # what % of retrieved keywords are expected
    mrr: float  # mean reciprocal rank
    
    has_hit: bool  # did any chunk match expected keywords


class MetricsCalculator:
    """Calculate retrieval metrics."""
    
    @staticmethod
    def extract_keywords(text: str) -> Set[str]:
        """Extract normalized keywords from text.
        
        Converts to lowercase, splits on whitespace/punctuation.
        Filters out very short tokens and common words.
        """
        # Lowercase and split on non-alphanumeric
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Filter: min length 3, exclude common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'be', 'been',
            'have', 'has', 'had', 'do', 'does', 'did', 'what', 'how', 'when',
            'where', 'why', 'which', 'who', 'i', 'you', 'he', 'she', 'it',
            'about', 'into', 'through', 'during', 'before', 'after', 'above',
            'below', 'up', 'down', 'out', 'off', 'over', 'under', 'again',
            'further', 'then', 'once', 'my', 'me', 'we', 'us', 'them', 'their'
        }
        
        return {w for w in words if len(w) >= 3 and w not in stop_words}
    
    @staticmethod
    def normalize_keywords(keywords: List[str]) -> Set[str]:
        """Normalize expected keywords for matching."""
        return {kw.lower().strip() for kw in keywords}
    
    @staticmethod
    def calculate_hit_at_k(retrieved_chunks: List[RetrievalResult], 
                          expected_keywords: Set[str], 
                          k: int) -> bool:
        """Calculate Hit@k: did any of top-k results contain an expected keyword?"""
        for chunk in retrieved_chunks[:k]:
            chunk_keywords = MetricsCalculator.extract_keywords(chunk.chunk_text)
            if chunk_keywords & expected_keywords:  # intersection
                return True
        return False
    
    @staticmethod
    def calculate_keyword_recall(retrieved_chunks: List[RetrievalResult],
                                 expected_keywords: Set[str]) -> float:
        """Keyword Recall: % of expected keywords found in top-k results."""
        if not expected_keywords:
            return 0.0
        
        found_keywords = set()
        for chunk in retrieved_chunks:
            chunk_keywords = MetricsCalculator.extract_keywords(chunk.chunk_text)
            found_keywords.update(chunk_keywords & expected_keywords)
        
        return len(found_keywords) / len(expected_keywords)
    
    @staticmethod
    def calculate_keyword_precision(retrieved_chunks: List[RetrievalResult],
                                    expected_keywords: Set[str]) -> float:
        """Keyword Precision: % of retrieved keywords that are expected."""
        if not retrieved_chunks:
            return 0.0
        
        all_retrieved_keywords = set()
        for chunk in retrieved_chunks:
            chunk_keywords = MetricsCalculator.extract_keywords(chunk.chunk_text)
            all_retrieved_keywords.update(chunk_keywords)
        
        if not all_retrieved_keywords:
            return 0.0
        
        matching_keywords = all_retrieved_keywords & expected_keywords
        return len(matching_keywords) / len(all_retrieved_keywords)
    
    @staticmethod
    def calculate_mrr(retrieved_chunks: List[RetrievalResult],
                     expected_keywords: Set[str]) -> float:
        """Mean Reciprocal Rank: 1 / (rank of first relevant result).
        
        A relevant result is one that contains any expected keyword.
        """
        for chunk in retrieved_chunks:
            chunk_keywords = MetricsCalculator.extract_keywords(chunk.chunk_text)
            if chunk_keywords & expected_keywords:
                return 1.0 / chunk.rank
        
        return 0.0
    
    @staticmethod
    def evaluate_query(query: str,
                      expected_keywords: List[str],
                      retrieved_chunks: List[RetrievalResult]) -> QueryMetrics:
        """Evaluate a single query against retrieved results."""
        
        normalized_keywords = MetricsCalculator.normalize_keywords(expected_keywords)
        
        # Compute metrics
        hit_at_1 = MetricsCalculator.calculate_hit_at_k(retrieved_chunks, normalized_keywords, 1)
        hit_at_3 = MetricsCalculator.calculate_hit_at_k(retrieved_chunks, normalized_keywords, 3)
        hit_at_5 = MetricsCalculator.calculate_hit_at_k(retrieved_chunks, normalized_keywords, 5)
        hit_at_10 = MetricsCalculator.calculate_hit_at_k(retrieved_chunks, normalized_keywords, 10)
        
        keyword_recall = MetricsCalculator.calculate_keyword_recall(retrieved_chunks, normalized_keywords)
        keyword_precision = MetricsCalculator.calculate_keyword_precision(retrieved_chunks, normalized_keywords)
        mrr = MetricsCalculator.calculate_mrr(retrieved_chunks, normalized_keywords)
        
        has_hit = hit_at_1 or hit_at_3 or hit_at_5 or hit_at_10
        
        return QueryMetrics(
            query=query,
            expected_keywords=expected_keywords,
            retrieved_chunks=retrieved_chunks,
            hit_at_1=hit_at_1,
            hit_at_3=hit_at_3,
            hit_at_5=hit_at_5,
            hit_at_10=hit_at_10,
            keyword_recall=keyword_recall,
            keyword_precision=keyword_precision,
            mrr=mrr,
            has_hit=has_hit
        )


class EvaluationSummary:
    """Summary statistics across all queries."""
    
    def __init__(self, query_metrics: List[QueryMetrics]):
        self.query_metrics = query_metrics
        self.total_queries = len(query_metrics)
        
        # Compute aggregates
        self.hit_at_1_rate = sum(1 for m in query_metrics if m.hit_at_1) / len(query_metrics) if query_metrics else 0.0
        self.hit_at_3_rate = sum(1 for m in query_metrics if m.hit_at_3) / len(query_metrics) if query_metrics else 0.0
        self.hit_at_5_rate = sum(1 for m in query_metrics if m.hit_at_5) / len(query_metrics) if query_metrics else 0.0
        self.hit_at_10_rate = sum(1 for m in query_metrics if m.hit_at_10) / len(query_metrics) if query_metrics else 0.0
        
        self.avg_keyword_recall = sum(m.keyword_recall for m in query_metrics) / len(query_metrics) if query_metrics else 0.0
        self.avg_keyword_precision = sum(m.keyword_precision for m in query_metrics) / len(query_metrics) if query_metrics else 0.0
        self.avg_mrr = sum(m.mrr for m in query_metrics) / len(query_metrics) if query_metrics else 0.0
        
        # Failed queries
        self.failed_queries = [m for m in query_metrics if not m.has_hit]
        self.passed_queries = [m for m in query_metrics if m.has_hit]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'total_queries': self.total_queries,
            'passed_queries': len(self.passed_queries),
            'failed_queries': len(self.failed_queries),
            'hit_at_1_rate': round(self.hit_at_1_rate, 3),
            'hit_at_3_rate': round(self.hit_at_3_rate, 3),
            'hit_at_5_rate': round(self.hit_at_5_rate, 3),
            'hit_at_10_rate': round(self.hit_at_10_rate, 3),
            'avg_keyword_recall': round(self.avg_keyword_recall, 3),
            'avg_keyword_precision': round(self.avg_keyword_precision, 3),
            'avg_mrr': round(self.avg_mrr, 3),
        }
