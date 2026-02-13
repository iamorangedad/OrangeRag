"""Extended BM25 cache manager supporting multiple index types.

This module extends the basic BM25 cache to support three types of indices:
- content: Standard BM25 index for Hybrid RAG (sparse retriever)
- metadata: Metadata-only index for MetadataMatcher
- citation: Content index optimized for citation retrieval

Cache structure:
.bm25_cache/
├── v2_content_index_{hash}.pkl
├── v2_content_{hash}_metadata.json
├── v2_metadata_index_{hash}.pkl
├── v2_metadata_{hash}_metadata.json
├── v2_citation_index_{hash}.pkl
└── v2_citation_{hash}_metadata.json
"""

import pickle
import hashlib
import json
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
from datetime import datetime
from enum import Enum

from llama_index.core.schema import TextNode

from app.core.retrievers.sparse_retriever import BM25Retriever
from app.core.cache.bm25_cache import BM25IndexCache


class CacheType(str, Enum):
    """Types of BM25 indices supported."""

    CONTENT = "content"  # Standard content index for Hybrid RAG
    METADATA = "metadata"  # Metadata-only index for MetadataMatcher
    CITATION = "citation"  # Optimized index for citation retrieval


class BM25MultiIndexCache:
    """
    Multi-index cache manager for different BM25 use cases.

    Manages three separate indices:
    1. CONTENT: Standard BM25 for Hybrid RAG sparse retrieval
    2. METADATA: Extracted metadata (filename, page_number, etc.) for matching
    3. CITATION: Content optimized for citation retrieval

    All indices use CACHE_VERSION = "2.0" for v2 metadata support.
    """

    CACHE_VERSION = "2.0"  # Version 2 adds metadata field support

    def __init__(self, cache_dir: str = ".bm25_cache"):
        """
        Initialize multi-index cache manager.

        Args:
            cache_dir: Base directory for all cache files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories for each cache type
        for cache_type in CacheType:
            (self.cache_dir / cache_type.value).mkdir(exist_ok=True)

    def _compute_content_hash(self, items: List[Any]) -> str:
        """
        Compute hash of items for cache validation.

        Args:
            items: List of items (nodes or metadata dicts)

        Returns:
            MD5 hash
        """
        if items and isinstance(items[0], TextNode):
            # For TextNode items
            content = "".join(node.text for node in sorted(items, key=lambda n: n.id_))
        else:
            # For metadata dict items
            content = json.dumps(items, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()

    def _get_cache_paths(self, cache_type: CacheType, content_hash: str) -> Tuple[Path, Path]:
        """
        Get cache file paths for given type and hash.

        Args:
            cache_type: Type of cache
            content_hash: Content hash

        Returns:
            Tuple of (index_path, metadata_path)
        """
        type_dir = self.cache_dir / cache_type.value
        index_path = type_dir / f"v{self.CACHE_VERSION}_{cache_type.value}_index_{content_hash}.pkl"
        metadata_path = (
            type_dir / f"v{self.CACHE_VERSION}_{cache_type.value}_{content_hash}_metadata.json"
        )
        return index_path, metadata_path

    def save_content_index(
        self, retriever: BM25Retriever, nodes: List[TextNode], k1: float = 1.5, b: float = 0.75
    ) -> None:
        """
        Save standard content index (for Hybrid RAG).

        Args:
            retriever: BM25 retriever with built index
            nodes: Document nodes
            k1: BM25 k1 parameter
            b: BM25 b parameter
        """
        self._save_index(CacheType.CONTENT, retriever, nodes, k1, b)

    def save_metadata_index(
        self, metadata_items: List[Dict[str, Any]], k1: float = 1.5, b: float = 0.75
    ) -> None:
        """
        Save metadata-only index (for MetadataMatcher).

        Args:
            metadata_items: List of metadata dicts with 'file_name', 'page_number', etc.
            k1: BM25 k1 parameter
            b: BM25 b parameter
        """
        content_hash = self._compute_content_hash(metadata_items)
        index_path, metadata_path = self._get_cache_paths(CacheType.METADATA, content_hash)

        # Prepare cache data
        cache_data = {
            "metadata_items": metadata_items,
            "k1": k1,
            "b": b,
            "cache_version": self.CACHE_VERSION,
            "cache_type": CacheType.METADATA.value,
        }

        # Save
        with open(index_path, "wb") as f:
            pickle.dump(cache_data, f)

        # Save metadata
        meta = {
            "content_hash": content_hash,
            "item_count": len(metadata_items),
            "k1": k1,
            "b": b,
            "created_at": datetime.now().isoformat(),
            "cache_version": self.CACHE_VERSION,
            "cache_type": CacheType.METADATA.value,
        }
        with open(metadata_path, "w") as f:
            json.dump(meta, f, indent=2)

    def save_citation_index(
        self,
        retriever: BM25Retriever,
        nodes: List[TextNode],
        k1: float = 1.2,  # More conservative for citations
        b: float = 0.75,
    ) -> None:
        """
        Save citation-optimized index (for CitationRetriever).

        Args:
            retriever: BM25 retriever
            nodes: Document nodes
            k1: BM25 k1 parameter (default 1.2 for more conservative matching)
            b: BM25 b parameter
        """
        self._save_index(CacheType.CITATION, retriever, nodes, k1, b)

    def _save_index(
        self,
        cache_type: CacheType,
        retriever: BM25Retriever,
        nodes: List[TextNode],
        k1: float,
        b: float,
    ) -> None:
        """Internal method to save a retriever index."""
        if retriever.is_empty:
            return

        content_hash = self._compute_content_hash(nodes)
        index_path, metadata_path = self._get_cache_paths(cache_type, content_hash)

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
            "cache_type": cache_type.value,
        }

        # Save index
        with open(index_path, "wb") as f:
            pickle.dump(cache_data, f)

        # Save metadata
        meta = {
            "content_hash": content_hash,
            "node_count": len(nodes),
            "k1": k1,
            "b": b,
            "created_at": datetime.now().isoformat(),
            "cache_version": self.CACHE_VERSION,
            "cache_type": cache_type.value,
        }
        with open(metadata_path, "w") as f:
            json.dump(meta, f, indent=2)

    def load_content_index(
        self, nodes: List[TextNode], k1: float = 1.5, b: float = 0.75
    ) -> Optional[Tuple[BM25Retriever, List[TextNode]]]:
        """
        Load standard content index.

        Args:
            nodes: Document nodes
            k1: BM25 k1 parameter
            b: BM25 b parameter

        Returns:
            Tuple of (retriever, loaded_nodes) if cache hit
        """
        return self._load_index(CacheType.CONTENT, nodes, k1, b)

    def load_metadata_index(
        self, metadata_items: List[Dict[str, Any]], k1: float = 1.5, b: float = 0.75
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Load metadata-only index.

        Args:
            metadata_items: Metadata items for hash computation
            k1: BM25 k1 parameter
            b: BM25 b parameter

        Returns:
            List of metadata dicts if cache hit
        """
        content_hash = self._compute_content_hash(metadata_items)
        index_path, metadata_path = self._get_cache_paths(CacheType.METADATA, content_hash)

        if not index_path.exists() or not metadata_path.exists():
            return None

        try:
            # Validate metadata
            with open(metadata_path, "r") as f:
                meta = json.load(f)

            if meta.get("cache_version") != self.CACHE_VERSION:
                return None
            if meta.get("k1") != k1 or meta.get("b") != b:
                return None

            # Load cache
            with open(index_path, "rb") as f:
                cache_data = pickle.load(f)

            return cache_data.get("metadata_items", [])

        except (pickle.PickleError, json.JSONDecodeError, KeyError, IOError):
            return None

    def load_citation_index(
        self, nodes: List[TextNode], k1: float = 1.2, b: float = 0.75
    ) -> Optional[Tuple[BM25Retriever, List[TextNode]]]:
        """
        Load citation-optimized index.

        Args:
            nodes: Document nodes
            k1: BM25 k1 parameter
            b: BM25 b parameter

        Returns:
            Tuple of (retriever, loaded_nodes) if cache hit
        """
        return self._load_index(CacheType.CITATION, nodes, k1, b)

    def _load_index(
        self, cache_type: CacheType, nodes: List[TextNode], k1: float, b: float
    ) -> Optional[Tuple[BM25Retriever, List[TextNode]]]:
        """Internal method to load a retriever index."""
        if not nodes:
            return None

        content_hash = self._compute_content_hash(nodes)
        index_path, metadata_path = self._get_cache_paths(cache_type, content_hash)

        if not index_path.exists() or not metadata_path.exists():
            return None

        try:
            # Validate metadata
            with open(metadata_path, "r") as f:
                meta = json.load(f)

            if meta.get("cache_version") != self.CACHE_VERSION:
                return None
            if meta.get("k1") != k1 or meta.get("b") != b:
                return None
            if meta.get("node_count") != len(nodes):
                return None

            # Load cache
            with open(index_path, "rb") as f:
                cache_data = pickle.load(f)

            if cache_data.get("cache_version") != self.CACHE_VERSION:
                return None

            # Reconstruct retriever
            retriever = BM25Retriever(k1=k1, b=b)
            retriever._tokenized_corpus = cache_data["tokenized_corpus"]

            # Reconstruct nodes with full metadata
            loaded_nodes = []
            for node_data in cache_data["nodes"]:
                node = TextNode(
                    id_=node_data["id"],
                    text=node_data["text"],
                    metadata=node_data["metadata"],  # Includes page_number, etc.
                )
                loaded_nodes.append(node)

            retriever._nodes = loaded_nodes

            # Rebuild BM25
            if retriever._tokenized_corpus:
                from rank_bm25 import BM25Okapi

                retriever._bm25 = BM25Okapi(retriever._tokenized_corpus, k1=k1, b=b)

            return retriever, loaded_nodes

        except (pickle.PickleError, json.JSONDecodeError, KeyError, IOError):
            return None

    def clear_cache(self, cache_type: Optional[CacheType] = None) -> Dict[str, int]:
        """
        Clear cached indices.

        Args:
            cache_type: Specific type to clear, or None for all

        Returns:
            Dict mapping cache type to number of files removed
        """
        results = {}

        types_to_clear = [cache_type] if cache_type else list(CacheType)

        for ct in types_to_clear:
            count = 0
            type_dir = self.cache_dir / ct.value

            if type_dir.exists():
                for f in type_dir.glob(f"v*_index_*.pkl"):
                    f.unlink()
                    count += 1
                for f in type_dir.glob(f"v*_*_metadata.json"):
                    f.unlink()
                    count += 1

            results[ct.value] = count

        return results

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics for all types.

        Returns:
            Dict with stats per cache type and totals
        """
        stats = {"cache_dir": str(self.cache_dir), "cache_version": self.CACHE_VERSION, "types": {}}

        total_files = 0
        total_size = 0

        for cache_type in CacheType:
            type_dir = self.cache_dir / cache_type.value
            if not type_dir.exists():
                continue

            files = list(type_dir.glob(f"v*_index_*.pkl"))
            size = sum(f.stat().st_size for f in files)

            stats["types"][cache_type.value] = {
                "index_count": len(files),
                "size_bytes": size,
                "size_mb": round(size / (1024 * 1024), 2),
            }

            total_files += len(files)
            total_size += size

        stats["total"] = {
            "index_count": total_files,
            "size_bytes": total_size,
            "size_mb": round(total_size / (1024 * 1024), 2),
        }

        return stats

    def migrate_from_v1(self) -> int:
        """
        Migrate v1 cache files to v2 format.

        v1 files don't have separate subdirectories and use old naming.
        v2 files are organized by type in subdirectories.

        Returns:
            Number of v1 files migrated
        """
        migrated = 0

        # Look for v1 files (old naming without v2_ prefix)
        for old_file in self.cache_dir.glob("bm25_*.pkl"):
            # Skip if already in subdirectory
            if old_file.parent != self.cache_dir:
                continue

            # Move to content directory as v1 content
            content_dir = self.cache_dir / CacheType.CONTENT.value
            content_dir.mkdir(exist_ok=True)

            # Rename with v1 prefix
            new_name = f"v1_content_{old_file.stem}.pkl"
            new_path = content_dir / new_name

            try:
                old_file.rename(new_path)
                migrated += 1
            except Exception:
                pass

        # Also migrate metadata files
        for old_meta in self.cache_dir.glob("bm25_*_metadata.json"):
            if old_meta.parent != self.cache_dir:
                continue

            content_dir = self.cache_dir / CacheType.CONTENT.value
            content_dir.mkdir(exist_ok=True)

            new_name = f"v1_content_{old_meta.stem}.json"
            new_path = content_dir / new_name

            try:
                old_meta.rename(new_path)
            except Exception:
                pass

        return migrated
