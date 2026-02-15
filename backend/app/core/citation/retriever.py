"""Citation retriever for precise text snippet retrieval using BM25.

This module provides a dedicated BM25-based retriever optimized for finding
exact text snippets to be used as citations. It operates independently from
the Hybrid RAG retrieval pipeline.

Example:
    retriever = CitationRetriever()
    retriever.build_index(nodes)

    # Global citation retrieval
    citations = retriever.retrieve("installation steps")

    # Restricted citation retrieval (within specific document/pages)
    citations = retriever.retrieve(
        "installation steps",
        metadata_filter={"file_name": "manual.pdf", "pages": [5, 6]}
    )
"""

import re
import logging
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
from pathlib import Path

from llama_index.core.schema import TextNode

from app.core.retrievers.sparse_retriever import BM25Retriever
from app.core.cache.bm25_multi_cache import BM25MultiIndexCache, CacheType

logger = logging.getLogger(__name__)


@dataclass
class CitationCandidate:
    """
    A candidate citation with source information.

    Attributes:
        index: Citation number [1], [2], etc.
        text: The exact text snippet
        source: Source document file name
        page: Page number (if available)
        page_label: Human-readable page label
        section: Section title (if available)
        score: BM25 relevance score
        metadata: Full metadata dict
        char_start: Character start position in original document
        char_end: Character end position in original document
    """

    index: int
    text: str
    source: str
    page: Optional[int] = None
    page_label: Optional[str] = None
    section: Optional[str] = None
    score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    char_start: Optional[int] = None
    char_end: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "index": self.index,
            "text": self.text[:300] + "..." if len(self.text) > 300 else self.text,
            "source": self.source,
            "page": self.page,
            "page_label": self.page_label,
            "section": self.section,
            "score": round(self.score, 4),
            "char_start": self.char_start,
            "char_end": self.char_end,
        }

    def format_citation(self, format_type: str = "short") -> str:
        """
        Format citation for display.

        Args:
            format_type: "short" (e.g., "[1]") or "full" (e.g., "[1] doc.pdf, Page 5")

        Returns:
            Formatted citation string
        """
        if format_type == "short":
            return f"[{self.index}]"

        # Full format
        parts = [f"[{self.index}]"]
        parts.append(self.source)

        if self.page:
            parts.append(self.page_label or f"Page {self.page}")

        if self.section:
            parts.append(self.section)

        return " ".join(parts)


@dataclass
class MetadataFilter:
    """
    Filter for restricting citation retrieval to specific documents/pages.

    Attributes:
        file_names: List of file names to restrict to
        pages: Dict mapping file_name -> list of page numbers
        exclude_file_names: Files to exclude
    """

    file_names: Optional[List[str]] = None
    pages: Optional[Dict[str, List[int]]] = None
    exclude_file_names: Optional[List[str]] = None

    def should_include(self, metadata: Dict[str, Any]) -> bool:
        """
        Check if a node with given metadata should be included.

        Args:
            metadata: Node metadata dict

        Returns:
            True if should be included in retrieval
        """
        file_name = metadata.get("file_name", "")

        # Check exclusion first
        if self.exclude_file_names and file_name in self.exclude_file_names:
            return False

        # Check file name filter
        if self.file_names is not None:
            if file_name not in self.file_names:
                return False

        # Check page filter
        if self.pages and file_name in self.pages:
            page = metadata.get("page_number")
            if page is not None and page not in self.pages[file_name]:
                return False

        return True


class CitationRetriever:
    """
    BM25-based retriever optimized for citation retrieval.

    This retriever uses a separate BM25 index with optimized parameters
    for finding exact text snippets. It supports both global retrieval
    and metadata-filtered retrieval.

    Key differences from standard BM25Retriever:
    - Separate index (doesn't interfere with Hybrid RAG)
    - Optimized BM25 parameters for citation (lower k1 for precision)
    - Supports metadata filtering
    - Returns CitationCandidate objects with full source info
    """

    # Optimized BM25 parameters for citation retrieval
    # Lower k1 = less term frequency saturation = more emphasis on exact matches
    DEFAULT_K1 = 1.2
    DEFAULT_B = 0.75

    def __init__(
        self,
        k1: float = DEFAULT_K1,
        b: float = DEFAULT_B,
        cache_enabled: bool = True,
        cache_dir: str = ".bm25_cache",
        min_score_threshold: float = 0.0,
    ):
        """
        Initialize citation retriever.

        Args:
            k1: BM25 k1 parameter (default 1.2, lower than standard 1.5)
            b: BM25 b parameter (default 0.75)
            cache_enabled: Whether to enable caching
            cache_dir: Cache directory
            min_score_threshold: Minimum BM25 score to include result
        """
        self.k1 = k1
        self.b = b
        self.min_score_threshold = min_score_threshold

        # Internal BM25 retriever
        self._bm25_retriever = BM25Retriever(
            k1=k1,
            b=b,
            cache_enabled=False,  # We handle caching separately
        )

        # Cache manager
        self._cache = BM25MultiIndexCache(cache_dir=cache_dir)
        self._cache_enabled = cache_enabled

        # Track all nodes for filtering
        self._all_nodes: List[TextNode] = []

        logger.info(
            f"[CitationRetriever] Initialized with k1={k1}, b={b}, threshold={min_score_threshold}"
        )

    def build_index(self, nodes: List[TextNode]) -> None:
        """
        Build citation index from document nodes.

        Args:
            nodes: List of TextNode chunks
        """
        logger.info(f"[CitationRetriever] Building index from {len(nodes)} nodes")

        if not nodes:
            logger.warning("[CitationRetriever] No nodes provided for indexing")
            return

        # Store all nodes for filtering
        self._all_nodes = nodes.copy()

        # Check cache first
        if self._cache_enabled:
            cached = self._cache.load_citation_index(nodes, k1=self.k1, b=self.b)
            if cached:
                retriever, loaded_nodes = cached
                self._bm25_retriever = retriever
                self._all_nodes = loaded_nodes
                logger.info(f"[CitationRetriever] Loaded {len(loaded_nodes)} nodes from cache")
                return

        # Build new index
        self._bm25_retriever = BM25Retriever(k1=self.k1, b=self.b, cache_enabled=False)
        self._bm25_retriever.add_documents(nodes)

        logger.info(
            f"[CitationRetriever] Indexed {len(nodes)} nodes, "
            f"{self._bm25_retriever.document_count} documents"
        )

        # Save to cache
        if self._cache_enabled:
            self._cache.save_citation_index(self._bm25_retriever, nodes, k1=self.k1, b=self.b)
            logger.info("[CitationRetriever] Saved index to cache")

    def retrieve(
        self,
        query: str,
        metadata_filter: Optional[MetadataFilter] = None,
        top_k: int = 5,
    ) -> List[CitationCandidate]:
        """
        Retrieve citation candidates for a query.

        Args:
            query: User query string
            metadata_filter: Optional filter to restrict results
            top_k: Number of top results to return

        Returns:
            List of CitationCandidate objects
        """
        if self._bm25_retriever.is_empty:
            logger.warning("[CitationRetriever] Cannot retrieve from empty index")
            return []

        # Get BM25 results
        results = self._bm25_retriever.retrieve(query, top_k=top_k * 2)  # Get extra for filtering

        if not results:
            return []

        # Filter and convert to candidates
        candidates = []
        seen_texts: Set[str] = set()  # Deduplication

        for idx, node_with_score in enumerate(results, start=1):
            # Check score threshold
            if node_with_score.score < self.min_score_threshold:
                continue

            # Get node metadata
            metadata = node_with_score.metadata

            # Apply metadata filter if provided
            if metadata_filter and not metadata_filter.should_include(metadata):
                continue

            # Deduplicate by text content
            text = node_with_score.text.strip()
            text_hash = hash(text[:100])  # Use first 100 chars for dedup
            if text_hash in seen_texts:
                continue
            seen_texts.add(text_hash)

            # Create citation candidate
            candidate = CitationCandidate(
                index=len(candidates) + 1,
                text=text,
                source=metadata.get("file_name", "Unknown"),
                page=metadata.get("page_number"),
                page_label=metadata.get("page_label"),
                section=metadata.get("section_title"),
                score=node_with_score.score,
                metadata=metadata,
                char_start=metadata.get("char_start"),
                char_end=metadata.get("char_end"),
            )

            candidates.append(candidate)

            # Stop when we have enough
            if len(candidates) >= top_k:
                break

        # Renumber indices
        for i, candidate in enumerate(candidates, start=1):
            candidate.index = i

        logger.debug(
            f"[CitationRetriever] Retrieved {len(candidates)} citations for query: {query[:50]}..."
        )

        return candidates

    def retrieve_for_document(
        self,
        query: str,
        file_name: str,
        pages: Optional[List[int]] = None,
        top_k: int = 5,
    ) -> List[CitationCandidate]:
        """
        Convenience method to retrieve citations from a specific document.

        Args:
            query: User query
            file_name: Target document file name
            pages: Optional specific pages to search
            top_k: Number of results

        Returns:
            List of CitationCandidate objects
        """
        filter_dict = {"file_names": [file_name]}
        if pages:
            filter_dict["pages"] = {file_name: pages}

        metadata_filter = MetadataFilter(**filter_dict)
        return self.retrieve(query, metadata_filter=metadata_filter, top_k=top_k)

    def get_document_citations(
        self,
        file_name: str,
        top_k: int = 10,
    ) -> List[CitationCandidate]:
        """
        Get all citation candidates from a specific document.

        Args:
            file_name: Document file name
            top_k: Maximum number to return

        Returns:
            List of citation candidates from the document
        """
        # Get all nodes for this document
        doc_nodes = [
            node for node in self._all_nodes if node.metadata.get("file_name") == file_name
        ]

        if not doc_nodes:
            return []

        # Create candidates
        candidates = []
        for i, node in enumerate(doc_nodes[:top_k], start=1):
            metadata = node.metadata
            candidate = CitationCandidate(
                index=i,
                text=node.text,
                source=file_name,
                page=metadata.get("page_number"),
                page_label=metadata.get("page_label"),
                section=metadata.get("section_title"),
                score=0.0,  # No score for direct listing
                metadata=metadata,
            )
            candidates.append(candidate)

        return candidates

    def clear(self) -> None:
        """Clear the citation index."""
        self._bm25_retriever.clear()
        self._all_nodes = []
        logger.info("[CitationRetriever] Index cleared")

    @property
    def is_empty(self) -> bool:
        """Check if citation index is empty."""
        return self._bm25_retriever.is_empty

    @property
    def document_count(self) -> int:
        """Get number of documents in index."""
        return self._bm25_retriever.document_count

    @property
    def node_count(self) -> int:
        """Get number of indexed nodes."""
        return len(self._all_nodes)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get retriever statistics.

        Returns:
            Dict with statistics
        """
        # Count unique documents and pages
        doc_pages: Dict[str, Set[int]] = {}

        for node in self._all_nodes:
            file_name = node.metadata.get("file_name", "")
            page = node.metadata.get("page_number")

            if file_name:
                if file_name not in doc_pages:
                    doc_pages[file_name] = set()
                if page:
                    doc_pages[file_name].add(page)

        return {
            "is_empty": self.is_empty,
            "document_count": len(doc_pages),
            "node_count": self.node_count,
            "documents": {
                name: {"pages": sorted(list(pages))} for name, pages in doc_pages.items()
            },
            "bm25_params": {
                "k1": self.k1,
                "b": self.b,
            },
        }


def create_citation_filter_from_match(
    matched_docs: List[str],
    matched_pages: Optional[Dict[str, List[int]]] = None,
) -> MetadataFilter:
    """
    Create a metadata filter from metadata match results.

    Args:
        matched_docs: List of matched document names
        matched_pages: Dict of file_name -> list of page numbers

    Returns:
        MetadataFilter configured for the match results
    """
    return MetadataFilter(
        file_names=matched_docs,
        pages=matched_pages,
    )


class CitationRetrieverManager:
    """
    Manager for multiple citation retrievers (per conversation).

    This is useful when different conversations need separate
    citation indices.
    """

    def __init__(self, cache_dir: str = ".bm25_cache"):
        """
        Initialize manager.

        Args:
            cache_dir: Cache directory
        """
        self._retrievers: Dict[str, CitationRetriever] = {}
        self._cache_dir = cache_dir

    def get_or_create_retriever(
        self,
        conversation_id: str,
        nodes: Optional[List[TextNode]] = None,
    ) -> CitationRetriever:
        """
        Get or create a citation retriever for a conversation.

        Args:
            conversation_id: Conversation ID
            nodes: Nodes to index (if creating new)

        Returns:
            CitationRetriever instance
        """
        if conversation_id in self._retrievers:
            return self._retrievers[conversation_id]

        retriever = CitationRetriever(cache_dir=self._cache_dir)

        if nodes:
            retriever.build_index(nodes)

        self._retrievers[conversation_id] = retriever
        return retriever

    def get_retriever(self, conversation_id: str) -> Optional[CitationRetriever]:
        """
        Get existing retriever for a conversation.

        Args:
            conversation_id: Conversation ID

        Returns:
            CitationRetriever or None if not found
        """
        return self._retrievers.get(conversation_id)

    def clear_conversation(self, conversation_id: str) -> None:
        """
        Clear retriever for a conversation.

        Args:
            conversation_id: Conversation ID
        """
        if conversation_id in self._retrievers:
            self._retrievers[conversation_id].clear()
            del self._retrievers[conversation_id]

    def clear_all(self) -> None:
        """Clear all retrievers."""
        for retriever in self._retrievers.values():
            retriever.clear()
        self._retrievers.clear()
