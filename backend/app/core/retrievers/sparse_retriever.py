"""Sparse retriever using BM25 algorithm."""

import re
from typing import List

from rank_bm25 import BM25Okapi
from llama_index.core.schema import TextNode

from app.core.retrievers.base import BaseRetriever, NodeWithScore


class BM25Retriever(BaseRetriever):
    """
    Sparse retriever using BM25 algorithm for keyword-based retrieval.

    BM25 is a probabilistic ranking function that scores documents based on
    query term frequency and document length. It's effective for exact keyword
    matching and term frequency-based retrieval.
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        """
        Initialize BM25 retriever.

        Args:
            k1: BM25 parameter controlling term frequency saturation
                (higher = more saturation, default: 1.5)
            b: BM25 parameter controlling document length normalization
                (0-1, default: 0.75)
        """
        self.k1 = k1
        self.b = b
        self._nodes: List[TextNode] = []
        self._tokenized_corpus: List[List[str]] = []
        self._bm25: BM25Okapi = None

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text for BM25 indexing.

        Args:
            text: Input text

        Returns:
            List of lowercase tokens
        """
        # Simple tokenization: lowercase and split on non-alphanumeric
        text = text.lower()
        tokens = re.findall(r"\b\w+\b", text)
        return tokens

    def add_documents(self, nodes: List[TextNode]) -> None:
        """
        Add documents to the retriever.

        Args:
            nodes: List of text nodes to add
        """
        if not nodes:
            return

        # Add to node list
        self._nodes.extend(nodes)

        # Tokenize and add to corpus
        for node in nodes:
            tokens = self._tokenize(node.text)
            self._tokenized_corpus.append(tokens)

        # Rebuild BM25 index
        if self._tokenized_corpus:
            self._bm25 = BM25Okapi(self._tokenized_corpus, k1=self.k1, b=self.b)

    def retrieve(self, query: str, top_k: int = 10) -> List[NodeWithScore]:
        """
        Retrieve documents using BM25 scoring.

        Args:
            query: Query string
            top_k: Number of top results to return

        Returns:
            List of nodes with BM25 scores
        """
        if self.is_empty or self._bm25 is None:
            return []

        # Tokenize query
        tokenized_query = self._tokenize(query)

        # Get BM25 scores for all documents
        scores = self._bm25.get_scores(tokenized_query)

        # Create (index, score) pairs and sort by score descending
        scored_indices = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

        # Build results
        results = []
        for idx, score in scored_indices[:top_k]:
            if score > 0:  # Only return documents with positive scores
                node = self._nodes[idx]
                results.append(NodeWithScore(node=node, score=float(score)))

        return results

    def clear(self) -> None:
        """Clear all documents from the retriever."""
        self._nodes.clear()
        self._tokenized_corpus.clear()
        self._bm25 = None

    @property
    def is_empty(self) -> bool:
        """Check if retriever has no documents."""
        return len(self._nodes) == 0

    @property
    def document_count(self) -> int:
        """Get the number of documents."""
        return len(self._nodes)
