"""Tests for metadata matcher module."""

import pytest
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path


class TestParsedQueryMetadata:
    """Test ParsedQueryMetadata dataclass."""

    def test_default_initialization(self):
        """Test default values."""
        from app.core.metadata.matcher import ParsedQueryMetadata

        meta = ParsedQueryMetadata()
        assert meta.file_names == []
        assert meta.page_numbers == []
        assert meta.section_hints == []
        assert meta.raw_query == ""


class TestMetadataMatchResult:
    """Test MetadataMatchResult dataclass."""

    def test_basic_creation(self):
        """Test creating match result."""
        from app.core.metadata.matcher import MetadataMatchResult, ParsedQueryMetadata

        query_meta = ParsedQueryMetadata(file_names=["test.pdf"])
        result = MetadataMatchResult(
            status="exact",
            matched_docs=["test.pdf"],
            matched_pages={"test.pdf": [1, 2]},
            confidence=0.95,
            query_metadata=query_meta,
        )

        assert result.status == "exact"
        assert result.confidence == 0.95

    def test_to_dict(self):
        """Test conversion to dict."""
        from app.core.metadata.matcher import MetadataMatchResult, ParsedQueryMetadata

        query_meta = ParsedQueryMetadata(file_names=["doc.pdf"], page_numbers=[5], raw_query="test")
        result = MetadataMatchResult(
            status="exact",
            matched_docs=["doc.pdf"],
            matched_pages={"doc.pdf": [5]},
            confidence=0.9,
            query_metadata=query_meta,
            message="Matched",
        )

        d = result.to_dict()
        assert d["status"] == "exact"
        assert d["confidence"] == 0.9
        assert d["matched_docs"] == ["doc.pdf"]
        assert d["query_metadata"]["file_names"] == ["doc.pdf"]


class TestQueryMetadataParser:
    """Test QueryMetadataParser."""

    def test_parse_file_name(self):
        """Test parsing file names."""
        from app.core.metadata.matcher import QueryMetadataParser

        parser = QueryMetadataParser()
        result = parser.parse("Find info in document.pdf")

        assert "document.pdf" in result.file_names

    def test_parse_chinese_page(self):
        """Test parsing Chinese page numbers."""
        from app.core.metadata.matcher import QueryMetadataParser

        parser = QueryMetadataParser()
        result = parser.parse("查看第5页的内容")

        assert 5 in result.page_numbers

    def test_parse_english_page(self):
        """Test parsing English page numbers."""
        from app.core.metadata.matcher import QueryMetadataParser

        parser = QueryMetadataParser()
        result = parser.parse("Check page 10 for details")

        assert 10 in result.page_numbers

    def test_parse_multiple_pages(self):
        """Test parsing multiple page references."""
        from app.core.metadata.matcher import QueryMetadataParser

        parser = QueryMetadataParser()
        result = parser.parse("Compare page 5 and page 10")

        assert 5 in result.page_numbers
        assert 10 in result.page_numbers

    def test_parse_file_hint_pattern(self):
        """Test parsing file hints without extension."""
        from app.core.metadata.matcher import QueryMetadataParser

        parser = QueryMetadataParser()
        result = parser.parse("OM0R028U says what?")

        assert "OM0R028U" in result.file_names

    def test_parse_combined(self):
        """Test parsing combined references."""
        from app.core.metadata.matcher import QueryMetadataParser

        parser = QueryMetadataParser()
        result = parser.parse("doc.pdf 第3页说了什么")

        assert "doc.pdf" in result.file_names
        assert 3 in result.page_numbers

    def test_parse_no_metadata(self):
        """Test parsing query without metadata."""
        from app.core.metadata.matcher import QueryMetadataParser

        parser = QueryMetadataParser()
        result = parser.parse("What is AI?")

        assert result.file_names == []
        assert result.page_numbers == []


class TestMetadataMatcher:
    """Test MetadataMatcher."""

    def test_init_default(self):
        """Test default initialization."""
        from app.core.metadata.matcher import MetadataMatcher

        matcher = MetadataMatcher()
        assert matcher._match_threshold == 0.8
        assert matcher._enable_fuzzy_match is True

    def test_init_custom(self):
        """Test custom initialization."""
        from app.core.metadata.matcher import MetadataMatcher

        matcher = MetadataMatcher(match_threshold=0.9, enable_fuzzy_match=False)
        assert matcher._match_threshold == 0.9
        assert matcher._enable_fuzzy_match is False

    def test_build_index_empty(self):
        """Test building index with empty nodes."""
        from app.core.metadata.matcher import MetadataMatcher

        matcher = MetadataMatcher()
        matcher.build_index([])

        assert matcher.is_empty

    def test_build_index_with_nodes(self):
        """Test building index with nodes."""
        from app.core.metadata.matcher import MetadataMatcher
        from llama_index.core.schema import TextNode

        matcher = MetadataMatcher()
        nodes = [
            TextNode(id_="1", text="content", metadata={"file_name": "doc.pdf", "page_number": 1}),
            TextNode(id_="2", text="content2", metadata={"file_name": "doc.pdf", "page_number": 2}),
        ]

        matcher.build_index(nodes)

        assert not matcher.is_empty
        assert matcher.document_count == 1
        assert matcher.metadata_count == 2
        assert "doc.pdf" in matcher.get_available_documents()

    def test_build_index_no_file_name(self):
        """Test building index with nodes missing file_name."""
        from app.core.metadata.matcher import MetadataMatcher
        from llama_index.core.schema import TextNode

        matcher = MetadataMatcher()
        nodes = [
            TextNode(id_="1", text="content", metadata={}),  # Missing file_name
        ]

        matcher.build_index(nodes)

        assert matcher.is_empty  # Should be empty since no valid metadata

    def test_match_exact_file(self):
        """Test exact file name matching."""
        from app.core.metadata.matcher import MetadataMatcher
        from llama_index.core.schema import TextNode

        matcher = MetadataMatcher()
        nodes = [
            TextNode(
                id_="1", text="content", metadata={"file_name": "document.pdf", "page_number": 5}
            )
        ]
        matcher.build_index(nodes)

        result = matcher.match("document.pdf content")

        assert result.status == "exact"
        assert "document.pdf" in result.matched_docs
        assert result.confidence == 1.0

    def test_match_case_insensitive(self):
        """Test case-insensitive file matching."""
        from app.core.metadata.matcher import MetadataMatcher
        from llama_index.core.schema import TextNode

        matcher = MetadataMatcher()
        nodes = [TextNode(id_="1", text="content", metadata={"file_name": "Document.PDF"})]
        matcher.build_index(nodes)

        result = matcher.match("document.pdf")

        assert result.status == "exact"
        assert "Document.PDF" in result.matched_docs

    def test_match_with_pages(self):
        """Test matching with page numbers."""
        from app.core.metadata.matcher import MetadataMatcher
        from llama_index.core.schema import TextNode

        matcher = MetadataMatcher()
        nodes = [
            TextNode(id_="1", text="content", metadata={"file_name": "doc.pdf", "page_number": 3}),
            TextNode(id_="2", text="content2", metadata={"file_name": "doc.pdf", "page_number": 5}),
        ]
        matcher.build_index(nodes)

        result = matcher.match("doc.pdf 第5页")

        assert result.status == "exact"
        assert result.matched_pages.get("doc.pdf") == [5]

    def test_match_no_metadata_in_query(self):
        """Test matching when query has no metadata references."""
        from app.core.metadata.matcher import MetadataMatcher
        from llama_index.core.schema import TextNode

        matcher = MetadataMatcher()
        nodes = [TextNode(id_="1", text="content", metadata={"file_name": "doc.pdf"})]
        matcher.build_index(nodes)

        result = matcher.match("What is AI?")

        assert result.status == "none"
        assert result.confidence == 0.0

    def test_match_no_match(self):
        """Test when file is not found."""
        from app.core.metadata.matcher import MetadataMatcher
        from llama_index.core.schema import TextNode

        matcher = MetadataMatcher()
        nodes = [TextNode(id_="1", text="content", metadata={"file_name": "doc.pdf"})]
        matcher.build_index(nodes)

        result = matcher.match("other.pdf content")

        assert result.status == "none"
        assert result.confidence == 0.0

    def test_get_available_documents(self):
        """Test getting list of documents."""
        from app.core.metadata.matcher import MetadataMatcher
        from llama_index.core.schema import TextNode

        matcher = MetadataMatcher()
        nodes = [
            TextNode(id_="1", text="c1", metadata={"file_name": "a.pdf"}),
            TextNode(id_="2", text="c2", metadata={"file_name": "b.pdf"}),
        ]
        matcher.build_index(nodes)

        docs = matcher.get_available_documents()

        assert len(docs) == 2
        assert "a.pdf" in docs
        assert "b.pdf" in docs

    def test_get_document_pages(self):
        """Test getting pages for a document."""
        from app.core.metadata.matcher import MetadataMatcher
        from llama_index.core.schema import TextNode

        matcher = MetadataMatcher()
        nodes = [
            TextNode(id_="1", text="c", metadata={"file_name": "doc.pdf", "page_number": 1}),
            TextNode(id_="2", text="c", metadata={"file_name": "doc.pdf", "page_number": 3}),
            TextNode(id_="3", text="c", metadata={"file_name": "doc.pdf", "page_number": 2}),
        ]
        matcher.build_index(nodes)

        pages = matcher.get_document_pages("doc.pdf")

        assert pages == [1, 2, 3]  # Should be sorted

    def test_clear(self):
        """Test clearing index."""
        from app.core.metadata.matcher import MetadataMatcher
        from llama_index.core.schema import TextNode

        matcher = MetadataMatcher()
        nodes = [TextNode(id_="1", text="c", metadata={"file_name": "doc.pdf"})]
        matcher.build_index(nodes)

        assert not matcher.is_empty

        matcher.clear()

        assert matcher.is_empty
        assert matcher.document_count == 0


class TestMetadataMatcherCache:
    """Test MetadataMatcher cache functionality."""

    def test_load_from_cache_empty(self):
        """Test loading when cache is empty."""
        from app.core.metadata.matcher import MetadataMatcher

        matcher = MetadataMatcher()
        result = matcher.load_from_cache()

        assert result is False

    @patch("app.core.metadata.matcher.BM25MultiIndexCache")
    def test_load_from_cache_success(self, mock_cache_class):
        """Test successful cache loading."""
        from app.core.metadata.matcher import MetadataMatcher

        # Setup mock cache
        mock_cache = MagicMock()
        mock_cache.load_metadata_index.return_value = [
            {"file_name": "cached.pdf", "page_number": 1}
        ]
        mock_cache_class.return_value = mock_cache

        matcher = MetadataMatcher()
        result = matcher.load_from_cache()

        assert result is True
        assert matcher.document_count == 1


class TestCreateMetadataMatcher:
    """Test convenience function."""

    def test_create_from_nodes(self):
        """Test creating matcher from nodes."""
        from app.core.metadata.matcher import create_metadata_matcher_from_nodes
        from llama_index.core.schema import TextNode

        nodes = [TextNode(id_="1", text="c", metadata={"file_name": "doc.pdf"})]

        matcher = create_metadata_matcher_from_nodes(nodes)

        assert not matcher.is_empty
        assert matcher.document_count == 1


class TestFuzzyMatching:
    """Test fuzzy matching capabilities."""

    def test_fuzzy_disabled_no_match(self):
        """Test that fuzzy match is not used when disabled."""
        from app.core.metadata.matcher import MetadataMatcher
        from llama_index.core.schema import TextNode

        matcher = MetadataMatcher(enable_fuzzy_match=False)
        nodes = [TextNode(id_="1", text="c", metadata={"file_name": "document.pdf"})]
        matcher.build_index(nodes)

        # "doc" is not an exact match for "document.pdf"
        result = matcher.match("doc.pdf")

        # With fuzzy disabled, this should not match
        # Note: "doc.pdf" will be parsed as file hint, but won't match exactly
        assert result.status in ["none", "partial"]

    def test_similarity_computation(self):
        """Test similarity computation."""
        from app.core.metadata.matcher import MetadataMatcher

        matcher = MetadataMatcher()

        # Same strings
        assert matcher._compute_similarity("test", "test") > 0.9

        # One contains other
        assert matcher._compute_similarity("doc", "document") > 0.5

        # Empty strings
        assert matcher._compute_similarity("", "test") == 0.0


class TestIntegration:
    """Integration tests."""

    def test_full_workflow(self):
        """Test complete workflow from build to match."""
        from app.core.metadata.matcher import MetadataMatcher, create_metadata_matcher_from_nodes
        from llama_index.core.schema import TextNode

        # Create nodes
        nodes = [
            TextNode(
                id_="1",
                text="Page 1 content",
                metadata={"file_name": "report.pdf", "page_number": 1},
            ),
            TextNode(
                id_="2",
                text="Page 2 content",
                metadata={"file_name": "report.pdf", "page_number": 2},
            ),
            TextNode(
                id_="3", text="Other doc", metadata={"file_name": "other.pdf", "page_number": 1}
            ),
        ]

        # Build matcher
        matcher = create_metadata_matcher_from_nodes(nodes)

        # Test various queries
        result1 = matcher.match("report.pdf")
        assert result1.status == "exact"

        result2 = matcher.match("report.pdf 第2页")
        assert result2.status == "exact"
        assert 2 in result2.matched_pages.get("report.pdf", [])

        result3 = matcher.match("nonexistent.pdf")
        assert result3.status == "none"

        result4 = matcher.match("What is this?")  # No metadata
        assert result4.status == "none"
