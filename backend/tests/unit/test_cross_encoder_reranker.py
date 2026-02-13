"""Unit tests for cross-encoder reranker."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from llama_index.core.schema import TextNode

from app.core.reranker.cross_encoder import CrossEncoderReranker
from app.core.retrievers.base import NodeWithScore


class TestCrossEncoderReranker:
    """Test cross-encoder reranker."""

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
                text="Deep learning uses neural networks",
                metadata={"source": "test2"},
            ),
            TextNode(
                id_="doc3",
                text="Python is a programming language",
                metadata={"source": "test3"},
            ),
        ]

    @pytest.fixture
    def sample_results(self, sample_nodes):
        """Create sample results with scores."""
        return [
            NodeWithScore(node=sample_nodes[0], score=0.8),
            NodeWithScore(node=sample_nodes[1], score=0.7),
            NodeWithScore(node=sample_nodes[2], score=0.6),
        ]

    def test_initialization(self):
        """Test reranker initialization."""
        with patch.dict("sys.modules", {"sentence_transformers": MagicMock()}):
            reranker = CrossEncoderReranker(
                model_name="cross-encoder/test-model",
                device="cpu",
                max_length=256,
                batch_size=16,
            )

            assert reranker.model_name == "cross-encoder/test-model"
            assert reranker._device == "cpu"
            assert reranker._max_length == 256
            assert reranker._batch_size == 16
            assert reranker._model is None  # Lazy loading

    def test_initialization_without_sentence_transformers(self):
        """Test initialization fails without sentence-transformers."""
        with patch.dict("sys.modules", {"sentence_transformers": None}):
            with pytest.raises(ImportError, match="sentence-transformers"):
                CrossEncoderReranker()

    def test_model_name_property(self):
        """Test model_name property."""
        with patch.dict("sys.modules", {"sentence_transformers": MagicMock()}):
            reranker = CrossEncoderReranker(model_name="test-model")
            assert reranker.model_name == "test-model"

    def test_rerank_empty_results(self):
        """Test reranking empty results."""
        with patch.dict("sys.modules", {"sentence_transformers": MagicMock()}):
            reranker = CrossEncoderReranker()
            results = reranker.rerank("query", [])

            assert results == []

    def test_rerank_basic(self, sample_results):
        """Test basic reranking."""
        mock_cross_encoder = MagicMock()
        mock_cross_encoder.predict.return_value = [0.9, 0.8, 0.7]

        mock_st = MagicMock()
        mock_st.CrossEncoder.return_value = mock_cross_encoder

        with patch.dict("sys.modules", {"sentence_transformers": mock_st}):
            reranker = CrossEncoderReranker()
            results = reranker.rerank("test query", sample_results, top_k=2)

            assert len(results) == 2
            assert all(hasattr(r, "score") for r in results)
            mock_cross_encoder.predict.assert_called_once()

    def test_rerank_without_top_k(self, sample_results):
        """Test reranking without top_k limit."""
        mock_cross_encoder = MagicMock()
        mock_cross_encoder.predict.return_value = [0.9, 0.8, 0.7]

        mock_st = MagicMock()
        mock_st.CrossEncoder.return_value = mock_cross_encoder

        with patch.dict("sys.modules", {"sentence_transformers": mock_st}):
            reranker = CrossEncoderReranker()
            results = reranker.rerank("test query", sample_results)

            assert len(results) == 3

    def test_rerank_preserves_nodes(self, sample_results):
        """Test that reranking preserves node content."""
        mock_cross_encoder = MagicMock()
        mock_cross_encoder.predict.return_value = [0.9, 0.8, 0.7]

        mock_st = MagicMock()
        mock_st.CrossEncoder.return_value = mock_cross_encoder

        with patch.dict("sys.modules", {"sentence_transformers": mock_st}):
            reranker = CrossEncoderReranker()
            results = reranker.rerank("test query", sample_results)

            # Check that nodes are preserved
            assert results[0].node == sample_results[0].node
            assert results[1].node == sample_results[1].node
            assert results[2].node == sample_results[2].node

    def test_rerank_scores_sorted(self, sample_results):
        """Test that reranked results are sorted by score."""
        mock_cross_encoder = MagicMock()
        mock_cross_encoder.predict.return_value = [0.5, 0.9, 0.7]

        mock_st = MagicMock()
        mock_st.CrossEncoder.return_value = mock_cross_encoder

        with patch.dict("sys.modules", {"sentence_transformers": mock_st}):
            reranker = CrossEncoderReranker()
            results = reranker.rerank("test query", sample_results)

            # Check scores are in descending order
            for i in range(len(results) - 1):
                assert results[i].score >= results[i + 1].score

    def test_rerank_batch(self, sample_results):
        """Test batch reranking."""
        mock_cross_encoder = MagicMock()
        mock_cross_encoder.predict.return_value = [0.9, 0.8]

        mock_st = MagicMock()
        mock_st.CrossEncoder.return_value = mock_cross_encoder

        with patch.dict("sys.modules", {"sentence_transformers": mock_st}):
            reranker = CrossEncoderReranker()

            queries = ["query1", "query2"]
            results_list = [sample_results[:2], sample_results[:2]]

            reranked = reranker.rerank_batch(queries, results_list, top_k=2)

            assert len(reranked) == 2
            assert all(len(r) == 2 for r in reranked)

    def test_rerank_batch_mismatch(self, sample_results):
        """Test batch reranking with mismatched lengths."""
        mock_st = MagicMock()

        with patch.dict("sys.modules", {"sentence_transformers": mock_st}):
            reranker = CrossEncoderReranker()

            queries = ["query1", "query2"]
            results_list = [sample_results]  # Only one result list

            with pytest.raises(ValueError, match="Number of queries"):
                reranker.rerank_batch(queries, results_list)

    def test_lazy_model_loading(self, sample_results):
        """Test that model is loaded lazily."""
        mock_cross_encoder = MagicMock()
        mock_cross_encoder.predict.return_value = [0.9]

        mock_st = MagicMock()
        mock_st.CrossEncoder.return_value = mock_cross_encoder

        with patch.dict("sys.modules", {"sentence_transformers": mock_st}):
            reranker = CrossEncoderReranker()

            # Model should not be loaded yet
            assert reranker._model is None

            # After rerank, model should be loaded
            reranker.rerank("query", sample_results[:1])
            assert reranker._model is not None
            mock_st.CrossEncoder.assert_called_once()

    def test_rerank_pairs_format(self, sample_results):
        """Test that query-document pairs are formatted correctly."""
        mock_cross_encoder = MagicMock()
        mock_cross_encoder.predict.return_value = [0.9, 0.8, 0.7]

        mock_st = MagicMock()
        mock_st.CrossEncoder.return_value = mock_cross_encoder

        with patch.dict("sys.modules", {"sentence_transformers": mock_st}):
            reranker = CrossEncoderReranker()
            query = "test query"
            reranker.rerank(query, sample_results)

            # Check that predict was called with correct pairs
            call_args = mock_cross_encoder.predict.call_args
            pairs = call_args[0][0]

            assert len(pairs) == 3
            for pair in pairs:
                assert len(pair) == 2
                assert pair[0] == query
                assert isinstance(pair[1], str)

    def test_batch_size_passed(self, sample_results):
        """Test that batch size is passed to predict."""
        mock_cross_encoder = MagicMock()
        mock_cross_encoder.predict.return_value = [0.9]

        mock_st = MagicMock()
        mock_st.CrossEncoder.return_value = mock_cross_encoder

        with patch.dict("sys.modules", {"sentence_transformers": mock_st}):
            reranker = CrossEncoderReranker(batch_size=8)
            reranker.rerank("query", sample_results[:1])

            call_kwargs = mock_cross_encoder.predict.call_args[1]
            assert call_kwargs.get("batch_size") == 8
            assert call_kwargs.get("show_progress_bar") is False
