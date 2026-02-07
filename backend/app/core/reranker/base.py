"""Base classes for rerankers."""

from abc import ABC, abstractmethod
from typing import List

from app.core.retrievers.base import NodeWithScore


class BaseReranker(ABC):
    """Abstract base class for all rerankers."""

    @abstractmethod
    def rerank(
        self, query: str, results: List[NodeWithScore], top_k: int = None
    ) -> List[NodeWithScore]:
        """
        Rerank retrieval results.

        Args:
            query: Original query string
            results: List of retrieval results to rerank
            top_k: Number of top results to return after reranking

        Returns:
            Reranked list of nodes with new scores
        """
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Get the name of the reranking model."""
        pass
