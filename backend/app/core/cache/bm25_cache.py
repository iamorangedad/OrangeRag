"""BM25 index cache manager for persisting and loading BM25 indices."""

import pickle
import hashlib
import json
from pathlib import Path
from typing import List, Optional, Tuple
from datetime import datetime

from llama_index.core.schema import TextNode

# Avoid circular import - import BM25Retriever lazily inside methods
# from app.core.retrievers.sparse_retriever import BM25Retriever


class BM25IndexCache:
    """
    Cache manager for BM25 indices to avoid rebuilding on every startup.

    The cache stores:
    - Tokenized corpus
    - BM25 parameters (k1, b)
    - Node metadata (without embeddings)
    - Index version for cache invalidation

    Cache invalidation happens when:
    - Documents change (detected via content hash)
    - BM25 parameters change
    - Index version changes (software update)
    """

    CACHE_VERSION = "1.0"

    def __init__(self, cache_dir: str = ".bm25_cache"):
        """
        Initialize cache manager.

        Args:
            cache_dir: Directory to store cache files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _compute_content_hash(self, nodes: List[TextNode]) -> str:
        """
        Compute hash of document contents for cache validation.

        Args:
            nodes: List of document nodes

        Returns:
            MD5 hash of concatenated document contents
        """
        content = "".join(node.text for node in sorted(nodes, key=lambda n: n.id_))
        return hashlib.md5(content.encode()).hexdigest()

    def _get_cache_path(self, content_hash: str) -> Path:
        """Get cache file path for given content hash."""
        return self.cache_dir / f"bm25_{content_hash}.pkl"

    def _get_metadata_path(self, content_hash: str) -> Path:
        """Get metadata file path for given content hash."""
        return self.cache_dir / f"bm25_{content_hash}_metadata.json"

    def save(
        self, retriever: "BM25Retriever", nodes: List[TextNode], k1: float = 1.5, b: float = 0.75
    ) -> None:
        """
        Save BM25 index to cache.

        Args:
            retriever: BM25 retriever with built index
            nodes: Document nodes
            k1: BM25 k1 parameter
            b: BM25 b parameter
        """
        if retriever.is_empty:
            return

        content_hash = self._compute_content_hash(nodes)
        cache_path = self._get_cache_path(content_hash)
        metadata_path = self._get_metadata_path(content_hash)

        # Prepare cache data
        cache_data = {
            "tokenized_corpus": retriever._tokenized_corpus,
            "nodes": [
                {"id": node.id_, "text": node.text, "metadata": node.metadata}
                for node in retriever._nodes
            ],
            "k1": k1,
            "b": b,
            "cache_version": self.CACHE_VERSION,
        }

        # Save index data
        with open(cache_path, "wb") as f:
            pickle.dump(cache_data, f)

        # Save metadata
        metadata = {
            "content_hash": content_hash,
            "node_count": len(nodes),
            "k1": k1,
            "b": b,
            "created_at": datetime.now().isoformat(),
            "cache_version": self.CACHE_VERSION,
        }
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    def load(
        self, nodes: List[TextNode], k1: float = 1.5, b: float = 0.75
    ) -> Optional[Tuple["BM25Retriever", List[TextNode]]]:
        """
        Load BM25 index from cache if valid.

        Args:
            nodes: Document nodes to check against cache
            k1: BM25 k1 parameter
            b: BM25 b parameter

        Returns:
            Tuple of (retriever, loaded_nodes) if cache hit, None otherwise
        """
        if not nodes:
            return None

        content_hash = self._compute_content_hash(nodes)
        cache_path = self._get_cache_path(content_hash)
        metadata_path = self._get_metadata_path(content_hash)

        # Check if cache files exist
        if not cache_path.exists() or not metadata_path.exists():
            return None

        try:
            # Load metadata and validate
            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            # Check cache version
            if metadata.get("cache_version") != self.CACHE_VERSION:
                return None

            # Check BM25 parameters
            if metadata.get("k1") != k1 or metadata.get("b") != b:
                return None

            # Check node count
            if metadata.get("node_count") != len(nodes):
                return None

            # Load cache data
            with open(cache_path, "rb") as f:
                cache_data = pickle.load(f)

            # Verify cache version in data
            if cache_data.get("cache_version") != self.CACHE_VERSION:
                return None

            # Verify BM25 parameters in data
            if cache_data.get("k1") != k1 or cache_data.get("b") != b:
                return None

            # Reconstruct retriever - lazy import to avoid circular dependency
            from app.core.retrievers.sparse_retriever import BM25Retriever

            retriever = BM25Retriever(k1=k1, b=b)
            retriever._tokenized_corpus = cache_data["tokenized_corpus"]

            # Reconstruct nodes
            loaded_nodes = []
            for node_data in cache_data["nodes"]:
                node = TextNode(
                    id_=node_data["id"], text=node_data["text"], metadata=node_data["metadata"]
                )
                loaded_nodes.append(node)

            retriever._nodes = loaded_nodes

            # Rebuild BM25 index from tokenized corpus
            if retriever._tokenized_corpus:
                from rank_bm25 import BM25Okapi

                retriever._bm25 = BM25Okapi(retriever._tokenized_corpus, k1=k1, b=b)

            return retriever, loaded_nodes

        except (pickle.PickleError, json.JSONDecodeError, KeyError, IOError):
            # Cache corrupted or unreadable
            return None

    def clear_cache(self) -> int:
        """
        Clear all cached indices.

        Returns:
            Number of cache files removed
        """
        count = 0
        for cache_file in self.cache_dir.glob("bm25_*.pkl"):
            cache_file.unlink()
            count += 1

        for metadata_file in self.cache_dir.glob("bm25_*_metadata.json"):
            metadata_file.unlink()
            count += 1

        return count

    def get_cache_stats(self) -> dict:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        cache_files = list(self.cache_dir.glob("bm25_*.pkl"))
        total_size = sum(f.stat().st_size for f in cache_files)

        return {
            "cache_dir": str(self.cache_dir),
            "index_count": len(cache_files),
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
        }
