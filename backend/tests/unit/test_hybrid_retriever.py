"""Unit tests for hybrid retriever."""

import pytest
import asyncio
from unittest.mock import Mock, MagicMock
from llama_index.core.schema import TextNode

from app.core.retrievers.hybrid_retriever import HybridRetriever
from app.core.retrievers.base import NodeWithScore


class TestHybridRetriever:
    """Test hybrid retriever combining dense and sparse retrieval."""

    @pytest.fixture
    def mock_embed_model(self):
        """Create mock embedding model."""
        mock = Mock()
        mock.get_text_embedding.return_value = [0.1] * 768
        mock.get_text_embedding_batch.return_value = [[0.1] * 768 for _ in range(4)]
        return mock

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

    def test_initialization(self, mock_embed_model):
        """Test hybrid retriever initialization."""
        retriever = HybridRetriever(
            embed_model=mock_embed_model,
            dense_top_k=10,
            sparse_top_k=10,
            final_top_k=5,
            rrf_k=60.0,
        )

        assert retriever.dense_top_k == 10
        assert retriever.sparse_top_k == 10
        assert retriever.final_top_k == 5
        assert retriever.rrf_k == 60.0
        assert retriever.fusion_mode == "rrf"
        assert retriever.is_empty

    def test_initialization_with_cache_disabled(self, mock_embed_model):
        """Test initialization with cache disabled."""
        retriever = HybridRetriever(
            embed_model=mock_embed_model,
            cache_enabled=False,
        )

        assert not retriever.sparse_retriever._cache_enabled

    def test_add_documents(self, mock_embed_model, sample_nodes):
        """Test adding documents."""
        retriever = HybridRetriever(embed_model=mock_embed_model)
        retriever.add_documents(sample_nodes)

        assert not retriever.is_empty
        assert retriever.document_count == 4

    def test_add_empty_list(self, mock_embed_model):
        """Test adding empty document list."""
        retriever = HybridRetriever(embed_model=mock_embed_model)
        retriever.add_documents([])

        assert retriever.is_empty

    def test_retrieve_from_empty_retriever(self, mock_embed_model):
        """Test retrieval from empty retriever."""
        retriever = HybridRetriever(embed_model=mock_embed_model)
        results = retriever.retrieve("test query")

        assert results == []

    def test_retrieve_basic(self, mock_embed_model, sample_nodes):
        """Test basic hybrid retrieval."""
        retriever = HybridRetriever(embed_model=mock_embed_model)
        retriever.add_documents(sample_nodes)

        results = retriever.retrieve("machine learning", top_k=3)

        assert isinstance(results, list)
        assert len(results) <= 3

    def test_set_fusion_weights(self, mock_embed_model):
        """Test setting fusion weights."""
        retriever = HybridRetriever(embed_model=mock_embed_model)

        retriever.set_fusion_weights(0.7, 0.3, mode="weighted")

        assert retriever.dense_weight == 0.7
        assert retriever.sparse_weight == 0.3
        assert retriever.fusion_mode == "weighted"

    def test_set_fusion_weights_invalid_mode(self, mock_embed_model):
        """Test setting invalid fusion mode."""
        retriever = HybridRetriever(embed_model=mock_embed_model)

        with pytest.raises(ValueError, match="Invalid fusion mode"):
            retriever.set_fusion_weights(0.5, 0.5, mode="invalid")

    def test_set_fusion_weights_out_of_range(self, mock_embed_model):
        """Test setting out of range weights."""
        retriever = HybridRetriever(embed_model=mock_embed_model)

        with pytest.raises(ValueError, match="Weights must be between 0 and 1"):
            retriever.set_fusion_weights(1.5, -0.5)

    def test_set_reranker(self, mock_embed_model):
        """Test setting reranker."""
        retriever = HybridRetriever(embed_model=mock_embed_model)
        mock_reranker = Mock()

        retriever.set_reranker(mock_reranker)

        assert retriever._reranker == mock_reranker

    def test_fuse_results_rrf(self, mock_embed_model, sample_nodes):
        """Test RRF fusion."""
        retriever = HybridRetriever(embed_model=mock_embed_model)
        retriever.fusion_mode = "rrf"

        # Create mock results
        dense_results = [
            NodeWithScore(node=sample_nodes[0], score=0.9),
            NodeWithScore(node=sample_nodes[1], score=0.8),
        ]
        sparse_results = [
            NodeWithScore(node=sample_nodes[1], score=0.85),
            NodeWithScore(node=sample_nodes[2], score=0.75),
        ]

        fused = retriever._fuse_results(dense_results, sparse_results)

        assert len(fused) > 0
        assert all(hasattr(r, "score") for r in fused)

    def test_fuse_results_weighted(self, mock_embed_model, sample_nodes):
        """Test weighted RRF fusion."""
        retriever = HybridRetriever(embed_model=mock_embed_model)
        retriever.set_fusion_weights(0.6, 0.4, mode="weighted")

        dense_results = [
            NodeWithScore(node=sample_nodes[0], score=0.9),
        ]
        sparse_results = [
            NodeWithScore(node=sample_nodes[0], score=0.8),
        ]

        fused = retriever._fuse_results(dense_results, sparse_results)

        assert len(fused) == 1

    def test_clear(self, mock_embed_model, sample_nodes):
        """Test clearing retriever."""
        retriever = HybridRetriever(embed_model=mock_embed_model)
        retriever.add_documents(sample_nodes)

        assert not retriever.is_empty

        retriever.clear()

        assert retriever.is_empty
        assert retriever.document_count == 0

    @pytest.mark.asyncio
    async def test_aretrieve(self, mock_embed_model, sample_nodes):
        """Test async retrieval."""
        retriever = HybridRetriever(embed_model=mock_embed_model)
        retriever.add_documents(sample_nodes)

        results = await retriever.aretrieve("machine learning", top_k=3)

        assert isinstance(results, list)
        assert len(results) <= 3

    @pytest.mark.asyncio
    async def test_aretrieve_empty(self, mock_embed_model):
        """Test async retrieval from empty retriever."""
        retriever = HybridRetriever(embed_model=mock_embed_model)
        results = await retriever.aretrieve("test query")

        assert results == []

    def test_match_metadata(self, mock_embed_model, sample_nodes):
        """Test metadata matching."""
        retriever = HybridRetriever(embed_model=mock_embed_model)
        retriever.add_documents(sample_nodes)

        result = retriever.match_metadata("document 1")

        assert hasattr(result, "status")
        assert hasattr(result, "matched_docs")

    def test_metadata_matcher_property(self, mock_embed_model):
        """Test metadata matcher property."""
        retriever = HybridRetriever(embed_model=mock_embed_model)

        matcher = retriever.metadata_matcher

        assert matcher is not None

    def test_retrieve_with_reranker(self, mock_embed_model, sample_nodes):
        """Test retrieval with reranker."""
        retriever = HybridRetriever(embed_model=mock_embed_model)
        retriever.add_documents(sample_nodes)

        # Mock reranker
        mock_reranker = Mock()
        mock_reranker.rerank = Mock(return_value=[NodeWithScore(node=sample_nodes[0], score=0.95)])
        retriever.set_reranker(mock_reranker)

        results = retriever.retrieve("test query", top_k=1)

        mock_reranker.rerank.assert_called_once()
        assert len(results) <= 1

    def test_bm25_parameters_passed(self, mock_embed_model):
        """Test BM25 parameters are passed correctly."""
        retriever = HybridRetriever(
            embed_model=mock_embed_model,
            bm25_k1=1.2,
            bm25_b=0.5,
        )

        assert retriever.sparse_retriever.k1 == 1.2
        assert retriever.sparse_retriever.b == 0.5
