"""Tests for BM25 multi-index cache."""

import pytest
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path
import json
import pickle


class TestCacheType:
    """Test CacheType enum."""

    def test_cache_type_values(self):
        """Test that CacheType has expected values."""
        from app.core.cache.bm25_multi_cache import CacheType

        assert CacheType.CONTENT.value == "content"
        assert CacheType.METADATA.value == "metadata"
        assert CacheType.CITATION.value == "citation"


class TestBM25MultiIndexCache:
    """Test cases for BM25MultiIndexCache."""

    def test_init_creates_directories(self, tmp_path):
        """Test that initialization creates subdirectories."""
        from app.core.cache.bm25_multi_cache import BM25MultiIndexCache, CacheType

        cache = BM25MultiIndexCache(cache_dir=str(tmp_path / ".bm25_cache"))

        # Check subdirectories exist
        for cache_type in CacheType:
            assert (Path(cache.cache_dir) / cache_type.value).exists()

    def test_cache_version(self):
        """Test that cache uses v2 format."""
        from app.core.cache.bm25_multi_cache import BM25MultiIndexCache

        assert BM25MultiIndexCache.CACHE_VERSION == "2.0"

    def test_compute_content_hash_nodes(self):
        """Test hash computation for TextNodes."""
        from app.core.cache.bm25_multi_cache import BM25MultiIndexCache
        from llama_index.core.schema import TextNode

        cache = BM25MultiIndexCache(cache_dir="/tmp/test")

        node1 = TextNode(id_="1", text="content1")
        node2 = TextNode(id_="2", text="content2")

        hash1 = cache._compute_content_hash([node1, node2])
        hash2 = cache._compute_content_hash([node1, node2])

        # Same content should produce same hash
        assert hash1 == hash2
        assert len(hash1) == 32  # MD5 hex length

    def test_compute_content_hash_metadata(self):
        """Test hash computation for metadata dicts."""
        from app.core.cache.bm25_multi_cache import BM25MultiIndexCache

        cache = BM25MultiIndexCache(cache_dir="/tmp/test")

        items = [
            {"file_name": "doc1.pdf", "page_number": 1},
            {"file_name": "doc2.pdf", "page_number": 2},
        ]

        hash1 = cache._compute_content_hash(items)
        hash2 = cache._compute_content_hash(items)

        assert hash1 == hash2

    def test_get_cache_paths(self):
        """Test cache path generation."""
        from app.core.cache.bm25_multi_cache import BM25MultiIndexCache, CacheType

        cache = BM25MultiIndexCache(cache_dir="/tmp/test")

        index_path, meta_path = cache._get_cache_paths(CacheType.CONTENT, "abc123")

        assert "content" in str(index_path)
        assert "v2_content_index_abc123.pkl" in str(index_path)
        assert "v2_content_abc123_metadata.json" in str(meta_path)

    @patch("app.core.cache.bm25_multi_cache.pickle")
    def test_save_and_load_content_index(self, mock_pickle, tmp_path):
        """Test saving and loading content index."""
        from app.core.cache.bm25_multi_cache import BM25MultiIndexCache, CacheType
        from app.core.retrievers.sparse_retriever import BM25Retriever
        from llama_index.core.schema import TextNode

        cache = BM25MultiIndexCache(cache_dir=str(tmp_path / ".bm25_cache"))

        # Create mock retriever
        mock_retriever = Mock(spec=BM25Retriever)
        mock_retriever.is_empty = False
        mock_retriever._tokenized_corpus = [["token1"], ["token2"]]

        node1 = TextNode(id_="1", text="content1", metadata={"page_number": 1})
        node2 = TextNode(id_="2", text="content2", metadata={"page_number": 2})
        mock_retriever._nodes = [node1, node2]

        # Mock pickle to actually serialize
        def mock_dump(data, f):
            f.write(pickle.dumps(data))

        def mock_load(f):
            return pickle.loads(f.read())

        mock_pickle.dump = mock_dump
        mock_pickle.load = mock_load

        # Save
        cache.save_content_index(mock_retriever, [node1, node2], k1=1.5, b=0.75)

        # Verify files created
        content_dir = tmp_path / ".bm25_cache" / "content"
        assert any(content_dir.iterdir())

    def test_save_metadata_index(self, tmp_path):
        """Test saving metadata-only index."""
        from app.core.cache.bm25_multi_cache import BM25MultiIndexCache, CacheType

        cache = BM25MultiIndexCache(cache_dir=str(tmp_path / ".bm25_cache"))

        metadata_items = [
            {"file_name": "doc1.pdf", "page_number": 1},
            {"file_name": "doc2.pdf", "page_number": 2},
        ]

        cache.save_metadata_index(metadata_items, k1=1.5, b=0.75)

        # Verify files created
        metadata_dir = tmp_path / ".bm25_cache" / "metadata"
        files = list(metadata_dir.iterdir())
        assert len(files) > 0

    def test_load_nonexistent_cache(self, tmp_path):
        """Test loading cache that doesn't exist."""
        from app.core.cache.bm25_multi_cache import BM25MultiIndexCache
        from llama_index.core.schema import TextNode

        cache = BM25MultiIndexCache(cache_dir=str(tmp_path / ".bm25_cache"))

        node = TextNode(id_="1", text="content")
        result = cache.load_content_index([node], k1=1.5, b=0.75)

        assert result is None

    def test_clear_cache_specific_type(self, tmp_path):
        """Test clearing specific cache type."""
        from app.core.cache.bm25_multi_cache import BM25MultiIndexCache, CacheType

        cache = BM25MultiIndexCache(cache_dir=str(tmp_path / ".bm25_cache"))

        # Create dummy files
        content_dir = tmp_path / ".bm25_cache" / "content"
        (content_dir / "v2_content_index_test.pkl").touch()

        result = cache.clear_cache(CacheType.CONTENT)

        assert result["content"] == 1

    def test_clear_cache_all_types(self, tmp_path):
        """Test clearing all cache types."""
        from app.core.cache.bm25_multi_cache import BM25MultiIndexCache

        cache = BM25MultiIndexCache(cache_dir=str(tmp_path / ".bm25_cache"))

        # Create dummy files in each type
        for cache_type in ["content", "metadata", "citation"]:
            type_dir = tmp_path / ".bm25_cache" / cache_type
            (type_dir / f"v2_{cache_type}_index_test.pkl").touch()
            (type_dir / f"v2_{cache_type}_test_metadata.json").touch()

        result = cache.clear_cache()

        assert result["content"] == 2  # pkl + json
        assert result["metadata"] == 2
        assert result["citation"] == 2

    def test_get_cache_stats(self, tmp_path):
        """Test cache statistics."""
        from app.core.cache.bm25_multi_cache import BM25MultiIndexCache

        cache = BM25MultiIndexCache(cache_dir=str(tmp_path / ".bm25_cache"))

        # Create dummy files with content
        content_dir = tmp_path / ".bm25_cache" / "content"
        test_file = content_dir / "v2_content_index_test.pkl"
        test_file.write_bytes(b"test content" * 100)  # 1100 bytes

        stats = cache.get_cache_stats()

        assert stats["cache_version"] == "2.0"
        assert stats["types"]["content"]["index_count"] == 1
        assert stats["types"]["content"]["size_bytes"] == 1100
        assert stats["total"]["index_count"] == 1

    def test_migrate_from_v1(self, tmp_path):
        """Test migration from v1 cache format."""
        from app.core.cache.bm25_multi_cache import BM25MultiIndexCache

        cache_dir = tmp_path / ".bm25_cache"
        cache_dir.mkdir()

        # Create v1 style files
        (cache_dir / "bm25_abc123.pkl").touch()
        (cache_dir / "bm25_abc123_metadata.json").touch()

        cache = BM25MultiIndexCache(cache_dir=str(cache_dir))
        migrated = cache.migrate_from_v1()

        assert migrated == 1
        # Check files moved to content directory
        content_dir = cache_dir / "content"
        assert (content_dir / "v1_content_bm25_abc123.pkl").exists()


class TestCacheIntegration:
    """Integration tests for cache operations."""

    def test_full_cache_lifecycle(self, tmp_path):
        """Test complete save-load-clear cycle."""
        from app.core.cache.bm25_multi_cache import BM25MultiIndexCache, CacheType
        from app.core.retrievers.sparse_retriever import BM25Retriever
        from llama_index.core.schema import TextNode

        cache_dir = str(tmp_path / ".bm25_cache")
        cache = BM25MultiIndexCache(cache_dir=cache_dir)

        # 1. Create and save content index
        mock_retriever = Mock(spec=BM25Retriever)
        mock_retriever.is_empty = False
        mock_retriever._tokenized_corpus = [["token1"]]

        node = TextNode(
            id_="1", text="content", metadata={"file_name": "test.pdf", "page_number": 5}
        )
        mock_retriever._nodes = [node]

        # Use real pickle operations
        cache.save_content_index(mock_retriever, [node])

        # 2. Save metadata index
        metadata = [{"file_name": "test.pdf", "page_number": 5}]
        cache.save_metadata_index(metadata)

        # 3. Verify stats
        stats = cache.get_cache_stats()
        assert stats["types"]["content"]["index_count"] == 1
        assert stats["types"]["metadata"]["index_count"] == 1

        # 4. Clear and verify
        cache.clear_cache()
        stats = cache.get_cache_stats()
        assert stats["total"]["index_count"] == 0
