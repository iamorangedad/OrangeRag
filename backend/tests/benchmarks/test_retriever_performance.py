"""Performance benchmark for Hybrid RAG retrievers.

This script benchmarks the performance of dense, sparse, and hybrid
retrieval methods under various conditions.
"""

import time
import random
import string
from typing import List, Tuple
from dataclasses import dataclass

from llama_index.core.schema import TextNode


def generate_random_text(length: int = 100) -> str:
    """Generate random text of given length."""
    words = []
    word_list = [
        "machine",
        "learning",
        "artificial",
        "intelligence",
        "neural",
        "network",
        "deep",
        "learning",
        "data",
        "science",
        "algorithm",
        "model",
        "training",
        "python",
        "programming",
        "code",
        "software",
        "development",
        "computer",
        "technology",
        "innovation",
        "research",
        "analysis",
        "system",
        "application",
    ]

    for _ in range(length):
        words.append(random.choice(word_list))

    return " ".join(words)


def create_test_documents(count: int) -> List[TextNode]:
    """Create test documents."""
    nodes = []
    for i in range(count):
        text = generate_random_text(random.randint(50, 200))
        node = TextNode(id_=f"doc_{i}", text=text, metadata={"index": i, "source": "benchmark"})
        nodes.append(node)
    return nodes


@dataclass
class BenchmarkResult:
    """Benchmark result data class."""

    method: str
    document_count: int
    index_time: float
    query_time: float
    avg_query_time: float
    cache_hit: bool = False


def benchmark_bm25(
    documents: List[TextNode], queries: List[str], use_cache: bool = True
) -> BenchmarkResult:
    """Benchmark BM25 retriever."""
    from app.core.retrievers.sparse_retriever import BM25Retriever

    retriever = BM25Retriever()

    # Index time
    start = time.time()
    retriever.add_documents(documents)
    index_time = time.time() - start

    # Query time
    query_times = []
    for query in queries:
        start = time.time()
        retriever.retrieve(query, top_k=10)
        query_times.append(time.time() - start)

    avg_query_time = sum(query_times) / len(query_times)

    return BenchmarkResult(
        method="BM25",
        document_count=len(documents),
        index_time=index_time,
        query_time=sum(query_times),
        avg_query_time=avg_query_time,
    )


def benchmark_hybrid(
    documents: List[TextNode], queries: List[str], embed_model=None
) -> BenchmarkResult:
    """Benchmark Hybrid retriever."""
    from app.core.retrievers.hybrid_retriever import HybridRetriever

    # Mock embedding model for testing
    if embed_model is None:
        from unittest.mock import Mock

        embed_model = Mock()
        embed_model.get_text_embedding.return_value = [0.1] * 768
        embed_model.get_text_embedding_batch.return_value = [[0.1] * 768] * len(documents)

    retriever = HybridRetriever(embed_model=embed_model)

    # Index time
    start = time.time()
    retriever.add_documents(documents)
    index_time = time.time() - start

    # Query time
    query_times = []
    for query in queries:
        start = time.time()
        retriever.retrieve(query, top_k=10)
        query_times.append(time.time() - start)

    avg_query_time = sum(query_times) / len(query_times)

    return BenchmarkResult(
        method="Hybrid",
        document_count=len(documents),
        index_time=index_time,
        query_time=sum(query_times),
        avg_query_time=avg_query_time,
    )


def benchmark_cached_bm25(
    documents: List[TextNode], queries: List[str]
) -> Tuple[BenchmarkResult, BenchmarkResult]:
    """Benchmark BM25 with cache (first load and cached load)."""
    from app.core.retrievers.sparse_retriever import BM25Retriever
    import tempfile
    import shutil

    # Create temporary cache directory
    cache_dir = tempfile.mkdtemp()

    try:
        # First run - build index and cache
        retriever1 = BM25Retriever(cache_enabled=True, cache_dir=cache_dir)
        start = time.time()
        retriever1.add_documents(documents)
        first_index_time = time.time() - start

        # First query
        query_times = []
        for query in queries:
            start = time.time()
            retriever1.retrieve(query, top_k=10)
            query_times.append(time.time() - start)

        first_result = BenchmarkResult(
            method="BM25 (First Load)",
            document_count=len(documents),
            index_time=first_index_time,
            query_time=sum(query_times),
            avg_query_time=sum(query_times) / len(query_times),
            cache_hit=False,
        )

        # Second run - load from cache
        retriever2 = BM25Retriever(cache_enabled=True, cache_dir=cache_dir)
        start = time.time()
        retriever2.add_documents(documents)
        cached_index_time = time.time() - start

        query_times = []
        for query in queries:
            start = time.time()
            retriever2.retrieve(query, top_k=10)
            query_times.append(time.time() - start)

        cached_result = BenchmarkResult(
            method="BM25 (Cached)",
            document_count=len(documents),
            index_time=cached_index_time,
            query_time=sum(query_times),
            avg_query_time=sum(query_times) / len(query_times),
            cache_hit=True,
        )

        return first_result, cached_result

    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)


def run_benchmarks():
    """Run all benchmarks."""
    print("=" * 80)
    print("Hybrid RAG Performance Benchmark")
    print("=" * 80)

    # Test configurations
    doc_counts = [100, 500, 1000]
    query_count = 50

    results = []

    for doc_count in doc_counts:
        print(f"\n{'=' * 80}")
        print(f"Testing with {doc_count} documents, {query_count} queries")
        print(f"{'=' * 80}")

        # Generate test data
        documents = create_test_documents(doc_count)
        queries = [generate_random_text(5) for _ in range(query_count)]

        # Benchmark BM25
        print("\n1. BM25 Retriever...")
        bm25_result = benchmark_bm25(documents, queries)
        results.append(bm25_result)
        print(f"   Index time: {bm25_result.index_time:.3f}s")
        print(f"   Avg query time: {bm25_result.avg_query_time * 1000:.2f}ms")

        # Benchmark BM25 with cache
        print("\n2. BM25 with Cache...")
        first_result, cached_result = benchmark_cached_bm25(documents, queries)
        results.append(first_result)
        results.append(cached_result)
        print(f"   First load index time: {first_result.index_time:.3f}s")
        print(f"   Cached load index time: {cached_result.index_time:.3f}s")
        print(f"   Cache speedup: {first_result.index_time / cached_result.index_time:.2f}x")
        print(f"   Avg query time: {cached_result.avg_query_time * 1000:.2f}ms")

        # Benchmark Hybrid
        print("\n3. Hybrid Retriever...")
        hybrid_result = benchmark_hybrid(documents, queries)
        results.append(hybrid_result)
        print(f"   Index time: {hybrid_result.index_time:.3f}s")
        print(f"   Avg query time: {hybrid_result.avg_query_time * 1000:.2f}ms")

    # Print summary
    print("\n" + "=" * 80)
    print("Benchmark Summary")
    print("=" * 80)
    print(f"{'Method':<25} {'Docs':>8} {'Index(s)':>10} {'Query(ms)':>12} {'Cache':>8}")
    print("-" * 80)

    for result in results:
        cache_str = "HIT" if result.cache_hit else "MISS"
        print(
            f"{result.method:<25} {result.document_count:>8} "
            f"{result.index_time:>10.3f} {result.avg_query_time * 1000:>12.2f} "
            f"{cache_str:>8}"
        )

    print("\n" + "=" * 80)


if __name__ == "__main__":
    run_benchmarks()
