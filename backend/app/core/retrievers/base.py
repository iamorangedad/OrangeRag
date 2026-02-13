"""Base classes for retrievers."""

from abc import ABC, abstractmethod
from typing import List, Any
from dataclasses import dataclass

from llama_index.core.schema import TextNode


@dataclass
class NodeWithScore:
    """Node with relevance score."""

    node: TextNode
    score: float

    @property
    def id_(self) -> str:
        """Get node ID."""
        return self.node.id_

    @property
    def text(self) -> str:
        """Get node text."""
        return self.node.text

    @property
    def metadata(self) -> dict:
        """Get node metadata."""
        return self.node.metadata


class BaseRetriever(ABC):
    """Abstract base class for all retrievers."""

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 10) -> List[NodeWithScore]:
        """
        Retrieve documents based on query.

        Args:
            query: Query string
            top_k: Number of top results to return

        Returns:
            List of nodes with scores
        """
        pass

    @abstractmethod
    def add_documents(self, nodes: List[TextNode]) -> None:
        """
        Add documents to the retriever.

        Args:
            nodes: List of text nodes to add
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all documents from the retriever."""
        pass

    @property
    @abstractmethod
    def is_empty(self) -> bool:
        """Check if retriever has no documents."""
        pass

    @property
    @abstractmethod
    def document_count(self) -> int:
        """Get the number of documents."""
        pass
