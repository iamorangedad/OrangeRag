"""Metadata matcher for exact document/page matching using BM25.

This module provides BM25-based metadata matching to validate if a query
references specific documents, pages, or sections.

Example:
    matcher = MetadataMatcher()
    matcher.build_index(nodes)

    # Query with document reference
    result = matcher.match("OM0R028U.pdf 第5页的内容")
    # result.status == "exact"
    # result.matched_docs == ["OM0R028U.pdf"]
    # result.matched_pages == {"OM0R028U.pdf": [5]}
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from pathlib import Path

from llama_index.core.schema import TextNode

from app.core.cache.bm25_multi_cache import BM25MultiIndexCache, CacheType

logger = logging.getLogger(__name__)


@dataclass
class ParsedQueryMetadata:
    """Parsed metadata from user query."""

    file_names: List[str] = field(default_factory=list)
    page_numbers: List[int] = field(default_factory=list)
    section_hints: List[str] = field(default_factory=list)
    raw_query: str = ""


@dataclass
class MetadataMatchResult:
    """Result of metadata matching."""

    status: str  # "exact" | "partial" | "none"
    matched_docs: List[str]
    matched_pages: Dict[str, List[int]]
    confidence: float  # 0.0 - 1.0
    query_metadata: ParsedQueryMetadata
    message: str = ""  # Human readable message

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "status": self.status,
            "matched_docs": self.matched_docs,
            "matched_pages": self.matched_pages,
            "confidence": round(self.confidence, 4),
            "query_metadata": {
                "file_names": self.query_metadata.file_names,
                "page_numbers": self.query_metadata.page_numbers,
                "section_hints": self.query_metadata.section_hints,
            },
            "message": self.message,
        }


class QueryMetadataParser:
    """
    Parse user queries to extract metadata references.

    Recognizes patterns like:
    - File names: "document.pdf", "doc 123"
    - Page numbers: "第5页", "page 5", "第 5 页"
    - Section hints: "第一章", "section 1"
    """

    # Pattern for Chinese page numbers
    CHINESE_PAGE_PATTERN = re.compile(r"第\s*(\d+)\s*[页頁]", re.IGNORECASE)

    # Pattern for English page numbers
    ENGLISH_PAGE_PATTERN = re.compile(r"\bpage\s*(\d+)\b", re.IGNORECASE)

    # Pattern for file name hints (filename with extension)
    FILE_NAME_PATTERN = re.compile(r"\b([\w\-]+\.(pdf|doc|docx|txt))\b", re.IGNORECASE)

    # Pattern for file name without extension (word + number format)
    FILE_HINT_PATTERN = re.compile(r"\b([a-zA-Z]+\d+[a-zA-Z\d]*)\b", re.IGNORECASE)

    # Pattern for Chinese sections
    CHINESE_SECTION_PATTERN = re.compile(r"第[一二三四五六七八九十\d]+[章节]", re.IGNORECASE)

    def parse(self, query: str) -> ParsedQueryMetadata:
        """
        Parse query to extract metadata references.

        Args:
            query: User query string

        Returns:
            ParsedQueryMetadata with extracted references
        """
        result = ParsedQueryMetadata(raw_query=query)

        # Extract file names
        result.file_names = self._extract_file_names(query)

        # Extract page numbers
        result.page_numbers = self._extract_page_numbers(query)

        # Extract section hints
        result.section_hints = self._extract_sections(query)

        logger.debug(f"[QueryParser] Parsed query '{query}': {result}")
        return result

    def _extract_file_names(self, query: str) -> List[str]:
        """Extract file name references from query."""
        # Find explicit file names with extensions
        matches = self.FILE_NAME_PATTERN.findall(query)
        file_names = [match[0] for match in matches]

        # Also look for patterns like "OM0R028U" (alphanumeric)
        if not file_names:
            hint_matches = self.FILE_HINT_PATTERN.findall(query)
            file_names = [match for match in hint_matches]

        return file_names

    def _extract_page_numbers(self, query: str) -> List[int]:
        """Extract page number references from query."""
        pages = []

        # Chinese format: 第5页, 第 5 页
        for match in self.CHINESE_PAGE_PATTERN.finditer(query):
            pages.append(int(match.group(1)))

        # English format: page 5
        for match in self.ENGLISH_PAGE_PATTERN.finditer(query):
            pages.append(int(match.group(1)))

        return pages

    def _extract_sections(self, query: str) -> List[str]:
        """Extract section references from query."""
        return self.CHINESE_SECTION_PATTERN.findall(query)


class MetadataMatcher:
    """
    BM25-based metadata matcher for exact document/page matching.

    This class uses a separate BM25 index on metadata fields to validate
    if user queries reference specific documents or pages.

    Attributes:
        _metadata_index: List of metadata dictionaries
        _doc_metadata_map: Map of file_name -> list of metadata entries
        _cache: BM25MultiIndexCache for persistence
        _parser: QueryMetadataParser for extracting query metadata
    """

    def __init__(
        self,
        cache_dir: str = ".bm25_cache",
        match_threshold: float = 0.8,
        enable_fuzzy_match: bool = True,
    ):
        """
        Initialize metadata matcher.

        Args:
            cache_dir: Directory for metadata cache
            match_threshold: Minimum BM25 score to consider a match (0-1)
            enable_fuzzy_match: Whether to enable fuzzy file name matching
        """
        self._metadata_index: List[Dict[str, Any]] = []
        self._doc_metadata_map: Dict[str, List[Dict[str, Any]]] = {}
        self._cache = BM25MultiIndexCache(cache_dir=cache_dir)
        self._parser = QueryMetadataParser()
        self._match_threshold = match_threshold
        self._enable_fuzzy_match = enable_fuzzy_match

        logger.info(f"[MetadataMatcher] Initialized with threshold={match_threshold}")

    def build_index(self, nodes: List[TextNode]) -> None:
        """
        Build metadata index from document nodes.

        Extracts metadata from each node and builds searchable index.

        Args:
            nodes: List of TextNode with metadata
        """
        logger.info(f"[MetadataMatcher] Building index from {len(nodes)} nodes")

        self._metadata_index = []
        self._doc_metadata_map = {}

        for node in nodes:
            metadata = self._extract_metadata_from_node(node)
            if metadata:
                self._metadata_index.append(metadata)

                # Group by file name
                file_name = metadata.get("file_name", "")
                if file_name:
                    if file_name not in self._doc_metadata_map:
                        self._doc_metadata_map[file_name] = []
                    self._doc_metadata_map[file_name].append(metadata)

        logger.info(f"[MetadataMatcher] Indexed {len(self._metadata_index)} metadata entries")
        logger.info(f"[MetadataMatcher] Documents: {list(self._doc_metadata_map.keys())}")

        # Save to cache
        self._cache.save_metadata_index(self._metadata_index)

    def _extract_metadata_from_node(self, node: TextNode) -> Optional[Dict[str, Any]]:
        """
        Extract searchable metadata from a TextNode.

        Args:
            node: TextNode with metadata

        Returns:
            Metadata dict or None if insufficient metadata
        """
        meta = node.metadata

        # Required: file_name
        file_name = meta.get("file_name", "")
        if not file_name:
            return None

        # Build searchable metadata entry
        entry = {
            "file_name": file_name,
            "file_path": meta.get("file_path", ""),
            "page_number": meta.get("page_number"),
            "total_pages": meta.get("total_pages"),
            "chunk_index": meta.get("chunk_index"),
            "total_chunks": meta.get("total_chunks"),
            # Searchable text combines multiple fields
            "searchable_text": self._build_searchable_text(file_name, meta),
        }

        return entry

    def _build_searchable_text(self, file_name: str, metadata: Dict[str, Any]) -> str:
        """
        Build searchable text from metadata fields.

        Args:
            file_name: Document file name
            metadata: Node metadata dict

        Returns:
            Searchable text string
        """
        parts = [file_name]

        # Add page number if available
        page = metadata.get("page_number")
        if page:
            parts.append(f"page {page}")
            parts.append(f"第{page}页")

        # Add file name without extension
        name_without_ext = Path(file_name).stem
        if name_without_ext != file_name:
            parts.append(name_without_ext)

        return " ".join(parts)

    def match(self, query: str) -> MetadataMatchResult:
        """
        Match query against metadata index.

        Args:
            query: User query string

        Returns:
            MetadataMatchResult with match status and details
        """
        # Parse query for metadata references
        query_meta = self._parser.parse(query)

        # If no metadata references found in query, return none status
        if not query_meta.file_names and not query_meta.page_numbers:
            logger.debug(f"[MetadataMatcher] No metadata references in query: {query}")
            return MetadataMatchResult(
                status="none",
                matched_docs=[],
                matched_pages={},
                confidence=0.0,
                query_metadata=query_meta,
                message="Query does not reference specific documents or pages",
            )

        # Try to match file names
        matched_docs = []
        matched_pages: Dict[str, List[int]] = {}
        confidence_scores = []

        for file_hint in query_meta.file_names:
            matched_file, score = self._match_file_name(file_hint)
            if matched_file:
                matched_docs.append(matched_file)
                confidence_scores.append(score)

                # Check if specific pages are requested
                if query_meta.page_numbers:
                    valid_pages = self._validate_pages(matched_file, query_meta.page_numbers)
                    if valid_pages:
                        matched_pages[matched_file] = valid_pages

        # Determine match status
        if matched_docs:
            avg_confidence = sum(confidence_scores) / len(confidence_scores)

            if len(matched_docs) == len(query_meta.file_names):
                status = "exact"
                message = f"Matched documents: {matched_docs}"
            else:
                status = "partial"
                message = f"Partially matched. Found: {matched_docs}, Missing: {set(query_meta.file_names) - set(matched_docs)}"

            return MetadataMatchResult(
                status=status,
                matched_docs=matched_docs,
                matched_pages=matched_pages,
                confidence=avg_confidence,
                query_metadata=query_meta,
                message=message,
            )
        else:
            return MetadataMatchResult(
                status="none",
                matched_docs=[],
                matched_pages={},
                confidence=0.0,
                query_metadata=query_meta,
                message=f"Could not match file names: {query_meta.file_names}",
            )

    def _match_file_name(self, file_hint: str) -> Tuple[Optional[str], float]:
        """
        Match a file name hint against indexed documents.

        Args:
            file_hint: File name or hint from query

        Returns:
            Tuple of (matched_file_name, confidence_score)
        """
        available_files = list(self._doc_metadata_map.keys())

        if not available_files:
            return None, 0.0

        # Exact match
        if file_hint in available_files:
            return file_hint, 1.0

        # Case-insensitive match
        file_hint_lower = file_hint.lower()
        for file in available_files:
            if file.lower() == file_hint_lower:
                return file, 1.0

        # Fuzzy match (if enabled)
        if self._enable_fuzzy_match:
            # Name without extension match
            hint_stem = Path(file_hint).stem.lower()
            for file in available_files:
                file_stem = Path(file).stem.lower()
                if file_stem == hint_stem:
                    return file, 0.95
                if hint_stem in file_stem or file_stem in hint_stem:
                    return file, 0.85

            best_match = None
            best_score = 0.0

            for file in available_files:
                score = self._compute_similarity(file_hint_lower, file.lower())
                if score > best_score and score > self._match_threshold:
                    best_score = score
                    best_match = file

            if best_match:
                return best_match, best_score

        return None, 0.0

    def _compute_similarity(self, s1: str, s2: str) -> float:
        """
        Compute string similarity (simple implementation).

        Args:
            s1: First string
            s2: Second string

        Returns:
            Similarity score (0.0 - 1.0)
        """
        # Simple character-based similarity
        if not s1 or not s2:
            return 0.0

        # Check if one contains the other
        if s1 in s2 or s2 in s1:
            return max(len(s1), len(s2)) / min(len(s1), len(s2))

        # Common substring ratio
        common = set(s1.lower()) & set(s2.lower())
        total = set(s1.lower()) | set(s2.lower())

        if not total:
            return 0.0

        return len(common) / len(total)

    def _validate_pages(self, file_name: str, page_numbers: List[int]) -> List[int]:
        """
        Validate that page numbers exist in document.

        Args:
            file_name: Document file name
            page_numbers: Requested page numbers

        Returns:
            List of valid page numbers
        """
        if file_name not in self._doc_metadata_map:
            return []

        # Get available pages for this document
        available_pages = set()
        for meta in self._doc_metadata_map[file_name]:
            page = meta.get("page_number")
            if page:
                available_pages.add(page)

        if not available_pages:
            # If no page metadata, assume all pages valid
            return page_numbers

        # Filter valid pages
        valid_pages = [p for p in page_numbers if p in available_pages]

        return valid_pages

    def get_document_pages(self, file_name: str) -> List[int]:
        """
        Get list of available pages for a document.

        Args:
            file_name: Document file name

        Returns:
            List of page numbers
        """
        if file_name not in self._doc_metadata_map:
            return []

        pages = set()
        for meta in self._doc_metadata_map[file_name]:
            page = meta.get("page_number")
            if page:
                pages.add(page)

        return sorted(list(pages))

    def get_available_documents(self) -> List[str]:
        """
        Get list of all indexed documents.

        Returns:
            List of file names
        """
        return list(self._doc_metadata_map.keys())

    def load_from_cache(self) -> bool:
        """
        Load metadata index from cache.

        Returns:
            True if loaded successfully
        """
        cached = self._cache.load_metadata_index([])
        if cached:
            self._metadata_index = cached

            # Rebuild doc map
            self._doc_metadata_map = {}
            for meta in self._metadata_index:
                file_name = meta.get("file_name", "")
                if file_name:
                    if file_name not in self._doc_metadata_map:
                        self._doc_metadata_map[file_name] = []
                    self._doc_metadata_map[file_name].append(meta)

            logger.info(f"[MetadataMatcher] Loaded {len(self._metadata_index)} entries from cache")
            return True

        return False

    def clear(self) -> None:
        """Clear the metadata index."""
        self._metadata_index = []
        self._doc_metadata_map = {}
        logger.info("[MetadataMatcher] Index cleared")

    @property
    def is_empty(self) -> bool:
        """Check if metadata index is empty."""
        return len(self._metadata_index) == 0

    @property
    def document_count(self) -> int:
        """Get number of indexed documents."""
        return len(self._doc_metadata_map)

    @property
    def metadata_count(self) -> int:
        """Get total number of metadata entries."""
        return len(self._metadata_index)


def create_metadata_matcher_from_nodes(nodes: List[TextNode]) -> MetadataMatcher:
    """
    Convenience function to create and build matcher from nodes.

    Args:
        nodes: List of TextNode documents

    Returns:
        Built MetadataMatcher
    """
    matcher = MetadataMatcher()
    matcher.build_index(nodes)
    return matcher
