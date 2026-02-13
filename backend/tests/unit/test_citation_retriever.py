"""Tests for citation retriever module."""

import pytest
from unittest.mock import Mock, MagicMock, patch


class TestCitationCandidate:
    """Test CitationCandidate dataclass."""

    def test_basic_creation(self):
        """Test creating a citation candidate."""
        from app.core.citation.retriever import CitationCandidate

        candidate = CitationCandidate(
            index=1, text="Test citation text", source="doc.pdf", page=5, score=0.95
        )

        assert candidate.index == 1
        assert candidate.text == "Test citation text"
        assert candidate.source == "doc.pdf"
        assert candidate.page == 5
        assert candidate.score == 0.95

    def test_to_dict(self):
        """Test conversion to dict."""
        from app.core.citation.retriever import CitationCandidate

        candidate = CitationCandidate(
            index=1, text="Short text", source="doc.pdf", page=5, score=0.9
        )

        d = candidate.to_dict()
        assert d["index"] == 1
        assert d["source"] == "doc.pdf"
        assert d["page"] == 5
        assert d["score"] == 0.9

    def test_to_dict_long_text_truncation(self):
        """Test that long text is truncated in to_dict."""
        from app.core.citation.retriever import CitationCandidate

        long_text = "A" * 400
        candidate = CitationCandidate(index=1, text=long_text, source="doc.pdf")

        d = candidate.to_dict()
        assert len(d["text"]) < 400
        assert "..." in d["text"]

    def test_format_citation_short(self):
        """Test short citation format."""
        from app.core.citation.retriever import CitationCandidate

        candidate = CitationCandidate(index=1, text="t", source="doc.pdf")

        assert candidate.format_citation("short") == "[1]"

    def test_format_citation_full(self):
        """Test full citation format."""
        from app.core.citation.retriever import CitationCandidate

        candidate = CitationCandidate(
            index=1, text="t", source="doc.pdf", page=5, page_label="Page 5", section="Intro"
        )

        formatted = candidate.format_citation("full")
        assert "[1]" in formatted
        assert "doc.pdf" in formatted
        assert "Page 5" in formatted
        assert "Intro" in formatted


class TestMetadataFilter:
    """Test MetadataFilter class."""

    def test_no_filters_allow_all(self):
        """Test that empty filter allows everything."""
        from app.core.citation.retriever import MetadataFilter

        filter_obj = MetadataFilter()

        assert filter_obj.should_include({"file_name": "any.pdf"}) is True

    def test_file_name_filter(self):
        """Test file name filtering."""
        from app.core.citation.retriever import MetadataFilter

        filter_obj = MetadataFilter(file_names=["doc1.pdf", "doc2.pdf"])

        assert filter_obj.should_include({"file_name": "doc1.pdf"}) is True
        assert filter_obj.should_include({"file_name": "doc3.pdf"}) is False

    def test_page_filter(self):
        """Test page number filtering."""
        from app.core.citation.retriever import MetadataFilter

        filter_obj = MetadataFilter(file_names=["doc.pdf"], pages={"doc.pdf": [1, 2, 3]})

        assert filter_obj.should_include({"file_name": "doc.pdf", "page_number": 2}) is True

        assert filter_obj.should_include({"file_name": "doc.pdf", "page_number": 5}) is False

    def test_exclude_filter(self):
        """Test exclusion filter."""
        from app.core.citation.retriever import MetadataFilter

        filter_obj = MetadataFilter(exclude_file_names=["bad.pdf"])

        assert filter_obj.should_include({"file_name": "good.pdf"}) is True
        assert filter_obj.should_include({"file_name": "bad.pdf"}) is False

    def test_combined_filters(self):
        """Test combined filters."""
        from app.core.citation.retriever import MetadataFilter

        filter_obj = MetadataFilter(
            file_names=["doc.pdf"],
            pages={"doc.pdf": [1, 2]},
            exclude_file_names=["doc.pdf"],  # Excluded takes precedence
        )

        assert (
            filter_obj.should_include({"file_name": "doc.pdf", "page_number": 1}) is False
        )  # Excluded


class TestCitationRetriever:
    """Test CitationRetriever class."""

    def test_init_default(self):
        """Test default initialization."""
        from app.core.citation.retriever import CitationRetriever

        retriever = CitationRetriever()

        assert retriever.k1 == 1.2
        assert retriever.b == 0.75
        assert retriever.min_score_threshold == 0.0
        assert retriever.is_empty is True

    def test_init_custom(self):
        """Test custom initialization."""
        from app.core.citation.retriever import CitationRetriever

        retriever = CitationRetriever(k1=1.5, b=0.5, min_score_threshold=0.2)

        assert retriever.k1 == 1.5
        assert retriever.b == 0.5
        assert retriever.min_score_threshold == 0.2

    @patch("app.core.citation.retriever.BM25MultiIndexCache")
    def test_build_index(self, mock_cache_class):
        """Test building index."""
        from app.core.citation.retriever import CitationRetriever
        from llama_index.core.schema import TextNode

        # Setup mock cache
        mock_cache = MagicMock()
        mock_cache.load_citation_index.return_value = None
        mock_cache_class.return_value = mock_cache

        retriever = CitationRetriever()
        nodes = [
            TextNode(id_="1", text="content1", metadata={"file_name": "doc.pdf"}),
            TextNode(id_="2", text="content2", metadata={"file_name": "doc.pdf"}),
        ]

        retriever.build_index(nodes)

        assert not retriever.is_empty
        assert retriever.node_count == 2
        mock_cache.save_citation_index.assert_called_once()

    @patch("app.core.citation.retriever.BM25Retriever")
    @patch("app.core.citation.retriever.BM25MultiIndexCache")
    def test_retrieve_global(self, mock_cache_class, mock_bm25_class):
        """Test global citation retrieval."""
        from app.core.citation.retriever import CitationRetriever
        from app.core.retrievers.base import NodeWithScore
        from llama_index.core.schema import TextNode

        # Setup mocks
        mock_cache = MagicMock()
        mock_cache.load_citation_index.return_value = None
        mock_cache_class.return_value = mock_cache

        # Setup BM25 retriever mock
        mock_bm25 = MagicMock()
        mock_bm25.is_empty = False
        mock_node = TextNode(
            id_="1", text="relevant text", metadata={"file_name": "doc.pdf", "page_number": 5}
        )
        mock_bm25.retrieve.return_value = [NodeWithScore(node=mock_node, score=0.9)]
        mock_bm25_class.return_value = mock_bm25

        retriever = CitationRetriever()
        retriever._bm25_retriever = mock_bm25
        retriever._all_nodes = [mock_node]

        citations = retriever.retrieve("query", top_k=1)

        assert len(citations) == 1
        assert citations[0].source == "doc.pdf"
        assert citations[0].page == 5
        assert citations[0].text == "relevant text"

    def test_retrieve_empty_index(self):
        """Test retrieval from empty index."""
        from app.core.citation.retriever import CitationRetriever

        retriever = CitationRetriever()

        citations = retriever.retrieve("query")

        assert citations == []

    @patch("app.core.citation.retriever.BM25Retriever")
    @patch("app.core.citation.retriever.BM25MultiIndexCache")
    def test_retrieve_with_filter(self, mock_cache_class, mock_bm25_class):
        """Test retrieval with metadata filter."""
        from app.core.citation.retriever import CitationRetriever, MetadataFilter
        from app.core.retrievers.base import NodeWithScore
        from llama_index.core.schema import TextNode

        # Setup mocks
        mock_cache = MagicMock()
        mock_cache.load_citation_index.return_value = None
        mock_cache_class.return_value = mock_cache

        mock_bm25 = MagicMock()
        mock_bm25.is_empty = False

        # Create nodes with different file names
        node1 = TextNode(
            id_="1", text="text1", metadata={"file_name": "doc1.pdf", "page_number": 1}
        )
        node2 = TextNode(
            id_="2", text="text2", metadata={"file_name": "doc2.pdf", "page_number": 1}
        )

        mock_bm25.retrieve.return_value = [
            NodeWithScore(node=node1, score=0.9),
            NodeWithScore(node=node2, score=0.8),
        ]
        mock_bm25_class.return_value = mock_bm25

        retriever = CitationRetriever()
        retriever._bm25_retriever = mock_bm25
        retriever._all_nodes = [node1, node2]

        # Filter to only doc1.pdf
        filter_obj = MetadataFilter(file_names=["doc1.pdf"])
        citations = retriever.retrieve("query", metadata_filter=filter_obj)

        assert len(citations) == 1
        assert citations[0].source == "doc1.pdf"

    @patch("app.core.citation.retriever.BM25Retriever")
    @patch("app.core.citation.retriever.BM25MultiIndexCache")
    def test_retrieve_with_score_threshold(self, mock_cache_class, mock_bm25_class):
        """Test that low scores are filtered out."""
        from app.core.citation.retriever import CitationRetriever
        from app.core.retrievers.base import NodeWithScore
        from llama_index.core.schema import TextNode

        mock_cache = MagicMock()
        mock_cache.load_citation_index.return_value = None
        mock_cache_class.return_value = mock_cache

        mock_bm25 = MagicMock()
        mock_bm25.is_empty = False

        node = TextNode(id_="1", text="text", metadata={"file_name": "doc.pdf"})
        mock_bm25.retrieve.return_value = [
            NodeWithScore(node=node, score=0.05),  # Below threshold
        ]
        mock_bm25_class.return_value = mock_bm25

        retriever = CitationRetriever(min_score_threshold=0.1)
        retriever._bm25_retriever = mock_bm25
        retriever._all_nodes = [node]

        citations = retriever.retrieve("query")

        assert len(citations) == 0  # Filtered by threshold

    def test_retrieve_for_document(self):
        """Test retrieve_for_document convenience method."""
        from app.core.citation.retriever import CitationRetriever

        retriever = CitationRetriever()

        # Mock the retrieve method
        retriever.retrieve = MagicMock(return_value=[])

        retriever.retrieve_for_document("query", file_name="doc.pdf", pages=[1, 2], top_k=3)

        # Verify retrieve was called with correct filter
        call_args = retriever.retrieve.call_args
        assert call_args[1]["top_k"] == 3
        assert call_args[1]["metadata_filter"].file_names == ["doc.pdf"]
        assert call_args[1]["metadata_filter"].pages == {"doc.pdf": [1, 2]}

    def test_get_document_citations(self):
        """Test getting all citations for a document."""
        from app.core.citation.retriever import CitationRetriever
        from llama_index.core.schema import TextNode

        retriever = CitationRetriever()
        retriever._all_nodes = [
            TextNode(id_="1", text="text1", metadata={"file_name": "doc.pdf", "page_number": 1}),
            TextNode(id_="2", text="text2", metadata={"file_name": "doc.pdf", "page_number": 2}),
            TextNode(id_="3", text="text3", metadata={"file_name": "other.pdf"}),
        ]

        citations = retriever.get_document_citations("doc.pdf")

        assert len(citations) == 2
        assert all(c.source == "doc.pdf" for c in citations)

    def test_clear(self):
        """Test clearing the index."""
        from app.core.citation.retriever import CitationRetriever
        from llama_index.core.schema import TextNode

        retriever = CitationRetriever()
        # Use build_index to properly set up the retriever
        retriever.build_index([TextNode(id_="1", text="t", metadata={})])

        assert not retriever.is_empty

        retriever.clear()

        assert retriever.is_empty
        assert retriever.node_count == 0

    def test_get_stats(self):
        """Test getting statistics."""
        from app.core.citation.retriever import CitationRetriever
        from llama_index.core.schema import TextNode

        retriever = CitationRetriever()
        # Use build_index to properly set up the retriever
        retriever.build_index(
            [
                TextNode(id_="1", text="t", metadata={"file_name": "doc.pdf", "page_number": 1}),
                TextNode(id_="2", text="t", metadata={"file_name": "doc.pdf", "page_number": 2}),
            ]
        )

        stats = retriever.get_stats()

        assert stats["document_count"] == 1
        assert stats["node_count"] == 2
        assert stats["is_empty"] is False
        assert "bm25_params" in stats


class TestCreateCitationFilterFromMatch:
    """Test helper function."""

    def test_create_filter(self):
        """Test creating filter from match results."""
        from app.core.citation.retriever import create_citation_filter_from_match

        filter_obj = create_citation_filter_from_match(
            matched_docs=["doc1.pdf", "doc2.pdf"],
            matched_pages={"doc1.pdf": [1, 2], "doc2.pdf": [3]},
        )

        assert filter_obj.file_names == ["doc1.pdf", "doc2.pdf"]
        assert filter_obj.pages == {"doc1.pdf": [1, 2], "doc2.pdf": [3]}

    def test_create_filter_no_pages(self):
        """Test creating filter without pages."""
        from app.core.citation.retriever import create_citation_filter_from_match

        filter_obj = create_citation_filter_from_match(matched_docs=["doc.pdf"])

        assert filter_obj.file_names == ["doc.pdf"]
        assert filter_obj.pages is None


class TestCitationRetrieverManager:
    """Test CitationRetrieverManager class."""

    def test_init(self):
        """Test initialization."""
        from app.core.citation.retriever import CitationRetrieverManager

        manager = CitationRetrieverManager()

        assert manager._cache_dir == ".bm25_cache"

    def test_get_or_create_retriever_new(self):
        """Test creating new retriever."""
        from app.core.citation.retriever import CitationRetrieverManager

        manager = CitationRetrieverManager()

        retriever = manager.get_or_create_retriever("conv-123")

        assert retriever is not None
        assert "conv-123" in manager._retrievers

    def test_get_or_create_retriever_existing(self):
        """Test getting existing retriever."""
        from app.core.citation.retriever import CitationRetrieverManager

        manager = CitationRetrieverManager()

        # Create first
        retriever1 = manager.get_or_create_retriever("conv-123")

        # Get again
        retriever2 = manager.get_or_create_retriever("conv-123")

        assert retriever1 is retriever2

    def test_get_retriever_not_found(self):
        """Test getting non-existent retriever."""
        from app.core.citation.retriever import CitationRetrieverManager

        manager = CitationRetrieverManager()

        retriever = manager.get_retriever("non-existent")

        assert retriever is None

    def test_clear_conversation(self):
        """Test clearing conversation retriever."""
        from app.core.citation.retriever import CitationRetrieverManager

        manager = CitationRetrieverManager()
        retriever = manager.get_or_create_retriever("conv-123")

        assert "conv-123" in manager._retrievers

        manager.clear_conversation("conv-123")

        assert "conv-123" not in manager._retrievers

    def test_clear_all(self):
        """Test clearing all retrievers."""
        from app.core.citation.retriever import CitationRetrieverManager

        manager = CitationRetrieverManager()
        manager.get_or_create_retriever("conv-1")
        manager.get_or_create_retriever("conv-2")

        assert len(manager._retrievers) == 2

        manager.clear_all()

        assert len(manager._retrievers) == 0


class TestIntegration:
    """Integration tests."""

    def test_full_retrieval_workflow(self):
        """Test complete retrieval workflow."""
        from app.core.citation.retriever import (
            CitationRetriever,
            MetadataFilter,
            create_citation_filter_from_match,
        )
        from llama_index.core.schema import TextNode

        # Setup
        retriever = CitationRetriever(cache_enabled=False)
        nodes = [
            TextNode(
                id_="1",
                text="Installation requires Python 3.8 or higher.",
                metadata={"file_name": "manual.pdf", "page_number": 5, "page_label": "Page 5"},
            ),
            TextNode(
                id_="2",
                text="To install, run pip install package.",
                metadata={"file_name": "manual.pdf", "page_number": 6, "page_label": "Page 6"},
            ),
            TextNode(
                id_="3",
                text="Unrelated content from other document.",
                metadata={"file_name": "other.pdf", "page_number": 1},
            ),
        ]

        retriever.build_index(nodes)

        # Test 1: Global retrieval
        citations = retriever.retrieve("install Python", top_k=2)
        assert len(citations) > 0

        # Test 2: Filtered retrieval
        filter_obj = MetadataFilter(file_names=["manual.pdf"])
        citations = retriever.retrieve("install", metadata_filter=filter_obj)
        assert all(c.source == "manual.pdf" for c in citations)

        # Test 3: Create filter from match results
        # Use query that matches content on page 5 (which contains "Python")
        filter_obj = create_citation_filter_from_match(
            matched_docs=["manual.pdf"], matched_pages={"manual.pdf": [5]}
        )
        citations = retriever.retrieve("Python", metadata_filter=filter_obj)
        assert len(citations) > 0
