"""Cross-encoder reranker for improving retrieval results."""

import logging
from typing import List, Optional

from app.core.reranker.base import BaseReranker
from app.core.retrievers.base import NodeWithScore

logger = logging.getLogger(__name__)


class CrossEncoderReranker(BaseReranker):
    """
    Cross-encoder reranker using sentence-transformers.

    Cross-encoders process query-document pairs together, allowing them to
    capture fine-grained interactions between queries and documents. This
    typically results in better ranking quality than dual-encoders or
    retrieval-only approaches.

    Note: This requires the sentence-transformers library to be installed.
    Install with: pip install sentence-transformers

    Example:
        ```python
        reranker = CrossEncoderReranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")

        # Get initial retrieval results
        results = retriever.retrieve("your query")

        # Rerank for better quality
        reranked = reranker.rerank("your query", results, top_k=5)
        ```
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: Optional[str] = None,
        max_length: int = 512,
        batch_size: int = 32,
    ):
        """
        Initialize cross-encoder reranker.

        Args:
            model_name: Name of the cross-encoder model
            device: Device to run on ('cpu', 'cuda', 'mps', or None for auto)
            max_length: Maximum sequence length for model
            batch_size: Batch size for inference

        Raises:
            ImportError: If sentence-transformers is not installed
        """
        self._model_name = model_name
        self._max_length = max_length
        self._batch_size = batch_size
        self._model = None
        self._device = device

        self._check_dependencies()

    def _check_dependencies(self) -> None:
        """Check if required dependencies are installed."""
        try:
            from sentence_transformers import CrossEncoder

            self._CrossEncoder = CrossEncoder
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for CrossEncoderReranker. "
                "Install with: pip install sentence-transformers"
            )

    def _load_model(self):
        """Lazy load the cross-encoder model."""
        if self._model is None:
            logger.info(f"[Reranker] Loading cross-encoder model: {self._model_name}")
            self._model = self._CrossEncoder(
                self._model_name,
                device=self._device,
                max_length=self._max_length,
            )
            logger.info("[Reranker] Cross-encoder model loaded successfully")

    @property
    def model_name(self) -> str:
        """Get the name of the reranking model."""
        return self._model_name

    def rerank(
        self, query: str, results: List[NodeWithScore], top_k: int = None
    ) -> List[NodeWithScore]:
        """
        Rerank retrieval results using cross-encoder.

        Args:
            query: Original query string
            results: List of retrieval results to rerank
            top_k: Number of top results to return after reranking
                   (default: None, returns all reranked results)

        Returns:
            Reranked list of nodes with cross-encoder scores
        """
        if not results:
            return []

        # Load model if not already loaded
        self._load_model()

        # Prepare query-document pairs
        pairs = [(query, result.text) for result in results]

        # Get cross-encoder scores
        logger.debug(f"[Reranker] Reranking {len(pairs)} documents")
        scores = self._model.predict(pairs, batch_size=self._batch_size, show_progress_bar=False)

        # Create new NodeWithScore objects with reranked scores
        reranked = []
        for result, score in zip(results, scores):
            new_node = NodeWithScore(node=result.node, score=float(score))
            reranked.append(new_node)

        # Sort by new scores descending
        reranked.sort(key=lambda x: x.score, reverse=True)

        # Return top-k if specified
        if top_k:
            return reranked[:top_k]

        return reranked

    def rerank_batch(
        self, queries: List[str], results_list: List[List[NodeWithScore]], top_k: int = None
    ) -> List[List[NodeWithScore]]:
        """
        Rerank multiple queries' results in batch.

        Args:
            queries: List of query strings
            results_list: List of result lists (one per query)
            top_k: Number of top results to return after reranking

        Returns:
            List of reranked result lists
        """
        if len(queries) != len(results_list):
            raise ValueError(
                f"Number of queries ({len(queries)}) must match "
                f"number of result lists ({len(results_list)})"
            )

        # Load model if not already loaded
        self._load_model()

        reranked_list = []
        for query, results in zip(queries, results_list):
            reranked = self.rerank(query, results, top_k)
            reranked_list.append(reranked)

        return reranked_list
