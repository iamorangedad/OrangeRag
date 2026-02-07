"""Hybrid retriever combining dense and sparse retrieval with RRF fusion."""

from typing import List, Optional
import asyncio

from llama_index.core.schema import TextNode
from llama_index.core.embeddings import BaseEmbedding

from app.core.retrievers.base import BaseRetriever, NodeWithScore
from app.core.retrievers.dense_retriever import DenseRetriever
from app.core.retrievers.sparse_retriever import BM25Retriever
from app.core.fusion.rrf_fusion import reciprocal_rank_fusion


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
        """
        self.dense_retriever = DenseRetriever(embed_model=embed_model)
        self.sparse_retriever = BM25Retriever(k1=bm25_k1, b=bm25_b)

        self.dense_top_k = dense_top_k
        self.sparse_top_k = sparse_top_k
        self.final_top_k = final_top_k
        self.rrf_k = rrf_k

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

        # Fuse results using RRF
        fused_results = reciprocal_rank_fusion([dense_results, sparse_results], k=self.rrf_k)

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

        # Fuse results using RRF
        fused_results = reciprocal_rank_fusion([dense_results, sparse_results], k=self.rrf_k)

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
