"""Hybrid retriever combining dense and sparse retrieval with RRF fusion."""

from typing import List, Optional
import asyncio

from llama_index.core.schema import TextNode
from llama_index.core.embeddings import BaseEmbedding

from app.core.retrievers.base import BaseRetriever, NodeWithScore
from app.core.retrievers.dense_retriever import DenseRetriever
from app.core.retrievers.sparse_retriever import BM25Retriever
from app.core.fusion.rrf_fusion import (
    reciprocal_rank_fusion,
    weighted_reciprocal_rank_fusion,
)


class HybridRetriever(BaseRetriever):
    """
    Hybrid retriever that combines dense (vector) and sparse (BM25) retrieval
    using Reciprocal Rank Fusion (RRF).

    This approach leverages the strengths of both retrieval methods:
    - Dense retrieval: Good for semantic similarity and understanding context
    - Sparse retrieval: Good for exact keyword matching and term frequency

    The RRF algorithm fuses the ranked results without requiring training.

    Example:
        ```python
        retriever = HybridRetriever(
            embed_model=embed_model,
            dense_top_k=10,
            sparse_top_k=10,
            final_top_k=5,
            rrf_k=60.0
        )
        retriever.add_documents(nodes)
        results = retriever.retrieve("your query", top_k=5)
        ```
    """

    def __init__(
        self,
        embed_model: BaseEmbedding,
        dense_top_k: int = 10,
        sparse_top_k: int = 10,
        final_top_k: int = 5,
        rrf_k: float = 60.0,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
        cache_enabled: bool = True,
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize hybrid retriever.

        Args:
            embed_model: Embedding model for dense retrieval
            dense_top_k: Number of results from dense retrieval
            sparse_top_k: Number of results from sparse retrieval
            final_top_k: Number of final results after fusion
            rrf_k: RRF constant (default: 60.0)
            bm25_k1: BM25 parameter k1 (default: 1.5)
            bm25_b: BM25 parameter b (default: 0.75)
            cache_enabled: Whether to enable BM25 index caching (default: True)
            cache_dir: Directory for BM25 cache files (default: None, uses .bm25_cache)
        """
        self.dense_retriever = DenseRetriever(embed_model=embed_model)
        self.sparse_retriever = BM25Retriever(
            k1=bm25_k1, b=bm25_b, cache_enabled=cache_enabled, cache_dir=cache_dir
        )

        self.dense_top_k = dense_top_k
        self.sparse_top_k = sparse_top_k
        self.final_top_k = final_top_k
        self.rrf_k = rrf_k
        self.dense_weight = 0.5
        self.sparse_weight = 0.5
        self.fusion_mode = "rrf"  # Options: "rrf", "weighted"
        self._reranker = None

    def set_reranker(self, reranker) -> None:
        """
        Set an optional reranker for post-fusion reranking.

        Args:
            reranker: A reranker instance (e.g., CrossEncoderReranker)
        """
        self._reranker = reranker

    def set_fusion_weights(
        self, dense_weight: float, sparse_weight: float, mode: str = "weighted"
    ) -> None:
        """
        Set fusion weights for dense and sparse retrieval.

        Args:
            dense_weight: Weight for dense retrieval results (0-1)
            sparse_weight: Weight for sparse retrieval results (0-1)
            mode: Fusion mode - "rrf" or "weighted" (default: "weighted")
        """
        if mode not in ["rrf", "weighted"]:
            raise ValueError(f"Invalid fusion mode: {mode}. Must be 'rrf' or 'weighted'")
        if not (0 <= dense_weight <= 1 and 0 <= sparse_weight <= 1):
            raise ValueError("Weights must be between 0 and 1")
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.fusion_mode = mode

    def _fuse_results(self, dense_results: list, sparse_results: list) -> list:
        """
        Fuse results using configured fusion method.

        Args:
            dense_results: Results from dense retrieval
            sparse_results: Results from sparse retrieval

        Returns:
            Fused results
        """
        if self.fusion_mode == "weighted":
            return weighted_reciprocal_rank_fusion(
                [dense_results, sparse_results],
                [self.dense_weight, self.sparse_weight],
                k=self.rrf_k,
            )
        else:
            return reciprocal_rank_fusion([dense_results, sparse_results], k=self.rrf_k)

    def add_documents(self, nodes: List[TextNode]) -> None:
        """
        Add documents to both retrievers.

        Args:
            nodes: List of text nodes to add
        """
        if not nodes:
            return

        # Add to both retrievers
        self.dense_retriever.add_documents(nodes)
        self.sparse_retriever.add_documents(nodes)

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[NodeWithScore]:
        """
        Retrieve documents using hybrid approach.

        This method performs retrieval from both dense and sparse channels,
        then fuses the results using RRF.

        Args:
            query: Query string
            top_k: Number of final results (defaults to final_top_k)

        Returns:
            List of fused nodes with RRF scores
        """
        if self.is_empty:
            return []

        final_k = top_k or self.final_top_k

        # Retrieve from both channels
        dense_results = self.dense_retriever.retrieve(query, top_k=self.dense_top_k)
        sparse_results = self.sparse_retriever.retrieve(query, top_k=self.sparse_top_k)

        # Fuse results using configured fusion method
        fused_results = self._fuse_results(dense_results, sparse_results)

        # Apply reranking if configured
        if self._reranker:
            fused_results = self._reranker.rerank(query, fused_results, top_k=final_k)

        # Return top-k results
        return fused_results[:final_k]

    async def aretrieve(self, query: str, top_k: Optional[int] = None) -> List[NodeWithScore]:
        """
        Async retrieve documents using hybrid approach.

        This method performs retrieval from both channels in parallel.

        Args:
            query: Query string
            top_k: Number of final results (defaults to final_top_k)

        Returns:
            List of fused nodes with RRF scores
        """
        if self.is_empty:
            return []

        final_k = top_k or self.final_top_k

        # Retrieve from both channels in parallel
        dense_task = asyncio.create_task(
            asyncio.to_thread(self.dense_retriever.retrieve, query, self.dense_top_k)
        )
        sparse_task = asyncio.create_task(
            asyncio.to_thread(self.sparse_retriever.retrieve, query, self.sparse_top_k)
        )

        # Wait for both to complete
        dense_results, sparse_results = await asyncio.gather(dense_task, sparse_task)

        # Fuse results using configured fusion method
        fused_results = self._fuse_results(dense_results, sparse_results)

        # Apply reranking if configured
        if self._reranker:
            fused_results = self._reranker.rerank(query, fused_results, top_k=final_k)

        return fused_results[:final_k]

    def clear(self) -> None:
        """Clear all documents from both retrievers."""
        self.dense_retriever.clear()
        self.sparse_retriever.clear()

    @property
    def is_empty(self) -> bool:
        """Check if retriever has no documents."""
        # Both should be in sync, but check both for safety
        return self.dense_retriever.is_empty or self.sparse_retriever.is_empty

    @property
    def document_count(self) -> int:
        """Get the number of documents."""
        return self.dense_retriever.document_count
