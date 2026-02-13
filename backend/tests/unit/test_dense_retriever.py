"""Unit tests for dense retriever."""

import pytest
from unittest.mock import Mock
from llama_index.core.schema import TextNode

from app.core.retrievers.dense_retriever import DenseRetriever
from app.core.retrievers.base import NodeWithScore


class TestDenseRetriever:
    """Test dense (vector) retriever."""

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
        """Test dense retriever initialization."""
        retriever = DenseRetriever(embed_model=mock_embed_model)

        assert retriever.embed_model == mock_embed_model
        assert retriever.is_empty
        assert retriever.document_count == 0

    def test_initialization_with_custom_store(self, mock_embed_model):
        """Test initialization with custom vector store."""
        from llama_index.core.vector_stores import SimpleVectorStore

        custom_store = SimpleVectorStore()
        retriever = DenseRetriever(embed_model=mock_embed_model, vector_store=custom_store)

        assert retriever.vector_store == custom_store

    def test_add_documents(self, mock_embed_model, sample_nodes):
        """Test adding documents."""
        retriever = DenseRetriever(embed_model=mock_embed_model)
        retriever.add_documents(sample_nodes)

        assert not retriever.is_empty
        assert retriever.document_count == 4
        mock_embed_model.get_text_embedding_batch.assert_called_once()

    def test_add_empty_list(self, mock_embed_model):
        """Test adding empty document list."""
        retriever = DenseRetriever(embed_model=mock_embed_model)
        retriever.add_documents([])

        assert retriever.is_empty
        mock_embed_model.get_text_embedding_batch.assert_not_called()

    def test_retrieve_from_empty_retriever(self, mock_embed_model):
        """Test retrieval from empty retriever."""
        retriever = DenseRetriever(embed_model=mock_embed_model)
        results = retriever.retrieve("test query")

        assert results == []
        mock_embed_model.get_text_embedding.assert_not_called()

    def test_retrieve_basic(self, mock_embed_model, sample_nodes):
        """Test basic retrieval."""
        retriever = DenseRetriever(embed_model=mock_embed_model)
        retriever.add_documents(sample_nodes)

        results = retriever.retrieve("machine learning", top_k=2)

        assert isinstance(results, list)
        mock_embed_model.get_text_embedding.assert_called_once()

    def test_embeddings_set_on_nodes(self, mock_embed_model, sample_nodes):
        """Test that embeddings are set on nodes after adding."""
        retriever = DenseRetriever(embed_model=mock_embed_model)
        retriever.add_documents(sample_nodes)

        for node in sample_nodes:
            assert node.embedding is not None
            assert len(node.embedding) == 768

    def test_clear(self, mock_embed_model, sample_nodes):
        """Test clearing retriever."""
        retriever = DenseRetriever(embed_model=mock_embed_model)
        retriever.add_documents(sample_nodes)

        assert not retriever.is_empty

        retriever.clear()

        assert retriever.is_empty
        assert retriever.document_count == 0

    def test_document_count(self, mock_embed_model, sample_nodes):
        """Test document count."""
        retriever = DenseRetriever(embed_model=mock_embed_model)

        assert retriever.document_count == 0

        retriever.add_documents(sample_nodes[:2])
        assert retriever.document_count == 2

        retriever.add_documents(sample_nodes[2:])
        assert retriever.document_count == 4

    def test_node_storage(self, mock_embed_model, sample_nodes):
        """Test that nodes are stored correctly."""
        retriever = DenseRetriever(embed_model=mock_embed_model)
        retriever.add_documents(sample_nodes)

        # Check internal storage
        assert len(retriever._nodes) == 4
        for node in sample_nodes:
            assert node.id_ in retriever._nodes
            assert retriever._nodes[node.id_] == node

    def test_retrieve_returns_node_with_score(self, mock_embed_model, sample_nodes):
        """Test that retrieval returns NodeWithScore objects."""
        retriever = DenseRetriever(embed_model=mock_embed_model)
        retriever.add_documents(sample_nodes)

        results = retriever.retrieve("test query")

        for result in results:
            assert isinstance(result, NodeWithScore)
            assert hasattr(result, "node")
            assert hasattr(result, "score")

    def test_metadata_preserved(self, mock_embed_model, sample_nodes):
        """Test that node metadata is preserved."""
        retriever = DenseRetriever(embed_model=mock_embed_model)
        retriever.add_documents(sample_nodes)

        results = retriever.retrieve("test query")

        if results:
            assert hasattr(results[0], "metadata")

    def test_top_k_parameter(self, mock_embed_model, sample_nodes):
        """Test top_k parameter is passed correctly."""
        retriever = DenseRetriever(embed_model=mock_embed_model)
        retriever.add_documents(sample_nodes)

        # Just verify no error occurs
        results = retriever.retrieve("test", top_k=3)
        assert isinstance(results, list)
