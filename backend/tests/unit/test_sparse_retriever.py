"""Unit tests for sparse BM25 retriever."""

import pytest
from llama_index.core.schema import TextNode

from app.core.retrievers.sparse_retriever import BM25Retriever


class TestBM25Retriever:
    """Test BM25 sparse retriever."""

    @pytest.fixture
    def sample_nodes(self):
        """Create sample document nodes."""
        return [
            TextNode(
                id_="doc1",
                text="Machine learning is a subset of artificial intelligence",
                metadata={"source": "test1"},
            ),
            TextNode(
                id_="doc2",
                text="Deep learning uses neural networks for AI applications",
                metadata={"source": "test2"},
            ),
            TextNode(
                id_="doc3",
                text="Python is a popular programming language",
                metadata={"source": "test3"},
            ),
            TextNode(
                id_="doc4",
                text="Artificial intelligence and machine learning are related fields",
                metadata={"source": "test4"},
            ),
        ]

    def test_initialization(self):
        """Test BM25 retriever initialization."""
        retriever = BM25Retriever(k1=1.5, b=0.75)
        assert retriever.k1 == 1.5
        assert retriever.b == 0.75
        assert retriever.is_empty
        assert retriever.document_count == 0

    def test_add_documents(self, sample_nodes):
        """Test adding documents."""
        retriever = BM25Retriever()
        retriever.add_documents(sample_nodes)

        assert not retriever.is_empty
        assert retriever.document_count == 4

    def test_retrieve_basic(self, sample_nodes):
        """Test basic retrieval."""
        retriever = BM25Retriever()
        retriever.add_documents(sample_nodes)

        results = retriever.retrieve("machine learning", top_k=2)

        assert len(results) <= 2
        assert all(hasattr(r, "score") for r in results)
        assert all(hasattr(r, "node") for r in results)

    def test_retrieve_exact_keyword_match(self, sample_nodes):
        """Test that exact keywords get higher scores."""
        retriever = BM25Retriever()
        retriever.add_documents(sample_nodes)

        results = retriever.retrieve("Python", top_k=1)

        assert len(results) == 1
        assert "Python" in results[0].text

    def test_retrieve_empty_query(self, sample_nodes):
        """Test retrieval with empty query."""
        retriever = BM25Retriever()
        retriever.add_documents(sample_nodes)

        results = retriever.retrieve("", top_k=5)
        assert len(results) == 0

    def test_retrieve_no_match(self, sample_nodes):
        """Test retrieval with non-matching query."""
        retriever = BM25Retriever()
        retriever.add_documents(sample_nodes)

        results = retriever.retrieve("quantum physics", top_k=5)
        assert len(results) == 0

    def test_retrieve_from_empty_retriever(self):
        """Test retrieval from empty retriever."""
        retriever = BM25Retriever()
        results = retriever.retrieve("test", top_k=5)
        assert len(results) == 0

    def test_clear(self, sample_nodes):
        """Test clearing retriever."""
        retriever = BM25Retriever()
        retriever.add_documents(sample_nodes)

        assert not retriever.is_empty

        retriever.clear()

        assert retriever.is_empty
        assert retriever.document_count == 0

    def test_tokenization(self, sample_nodes):
        """Test text tokenization."""
        retriever = BM25Retriever()
        tokens = retriever._tokenize("Hello, World! This is a test.")

        assert "hello" in tokens
        assert "world" in tokens
        assert "this" in tokens
        assert "," not in tokens
        assert "!" not in tokens

    def test_bm25_parameters(self, sample_nodes):
        """Test different BM25 parameters."""
        # Test with different k1 and b values
        retriever1 = BM25Retriever(k1=1.2, b=0.5)
        retriever2 = BM25Retriever(k1=2.0, b=0.8)

        retriever1.add_documents(sample_nodes)
        retriever2.add_documents(sample_nodes)

        results1 = retriever1.retrieve("machine learning", top_k=3)
        results2 = retriever2.retrieve("machine learning", top_k=3)

        # Both should return results
        assert len(results1) > 0
        assert len(results2) > 0

    def test_multiple_add_calls(self, sample_nodes):
        """Test adding documents in multiple calls."""
        retriever = BM25Retriever()

        retriever.add_documents(sample_nodes[:2])
        assert retriever.document_count == 2

        retriever.add_documents(sample_nodes[2:])
        assert retriever.document_count == 4

    def test_retrieve_top_k_limit(self, sample_nodes):
        """Test top_k limit is respected."""
        retriever = BM25Retriever()
        retriever.add_documents(sample_nodes)

        results = retriever.retrieve("learning", top_k=2)
        assert len(results) <= 2

    def test_node_metadata_preserved(self, sample_nodes):
        """Test that node metadata is preserved."""
        retriever = BM25Retriever()
        retriever.add_documents(sample_nodes)

        results = retriever.retrieve("Python", top_k=1)

        assert len(results) == 1
        assert results[0].metadata["source"] == "test3"
