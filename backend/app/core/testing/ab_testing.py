"""A/B testing framework for evaluating Hybrid RAG configurations."""

import json
import time
import uuid
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path


@dataclass
class TestResult:
    """Result of a single test run."""

    test_id: str
    variant_name: str
    query: str
    latency_ms: float
    num_results: int
    scores: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class TestVariant:
    """Configuration variant for A/B testing."""

    name: str
    config: Dict[str, Any]
    description: str = ""


class ABTestFramework:
    """
    Framework for A/B testing Hybrid RAG configurations.

    This framework allows you to:
    - Define multiple configuration variants
    - Run queries against all variants
    - Collect and compare metrics
    - Generate evaluation reports

    Example:
        ```python
        framework = ABTestFramework()

        # Define variants
        variant_a = TestVariant(
            name="baseline",
            config={"dense_top_k": 10, "sparse_top_k": 10, "fusion_mode": "rrf"},
            description="Standard RRF fusion"
        )

        variant_b = TestVariant(
            name="weighted",
            config={"dense_top_k": 10, "sparse_top_k": 10, "fusion_mode": "weighted", "dense_weight": 0.7},
            description="Weighted fusion favoring dense"
        )

        framework.add_variant(variant_a)
        framework.add_variant(variant_b)

        # Run tests
        results = framework.run_test(query="your query", retriever_factory=your_factory)

        # Generate report
        report = framework.generate_report()
        ```
    """

    def __init__(self, output_dir: str = "ab_test_results"):
        """
        Initialize A/B test framework.

        Args:
            output_dir: Directory to store test results
        """
        self.variants: List[TestVariant] = []
        self.results: List[TestResult] = []
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._test_id = str(uuid.uuid4())[:8]

    def add_variant(self, variant: TestVariant) -> None:
        """
        Add a test variant.

        Args:
            variant: Test variant configuration
        """
        self.variants.append(variant)

    def run_test(
        self,
        query: str,
        retriever_factory: Callable[[Dict[str, Any]], Any],
        reranker_factory: Optional[Callable[[], Any]] = None,
    ) -> Dict[str, TestResult]:
        """
        Run a test query against all variants.

        Args:
            query: Test query
            retriever_factory: Factory function that creates retriever from config
            reranker_factory: Optional factory for reranker

        Returns:
            Dictionary mapping variant names to results
        """
        if not self.variants:
            raise ValueError("No variants configured. Use add_variant() first.")

        results = {}

        for variant in self.variants:
            # Create retriever with variant config
            retriever = retriever_factory(variant.config)

            # Apply reranker if specified
            if reranker_factory:
                reranker = reranker_factory()
                if hasattr(retriever, "set_reranker"):
                    retriever.set_reranker(reranker)

            # Run retrieval and measure latency
            start_time = time.time()
            retrieved_results = retriever.retrieve(query)
            latency_ms = (time.time() - start_time) * 1000

            # Collect scores
            scores = [r.score for r in retrieved_results]

            # Create result
            result = TestResult(
                test_id=self._test_id,
                variant_name=variant.name,
                query=query,
                latency_ms=latency_ms,
                num_results=len(retrieved_results),
                scores=scores,
                metadata={
                    "config": variant.config,
                    "description": variant.description,
                },
            )

            self.results.append(result)
            results[variant.name] = result

        return results

    def run_test_suite(
        self,
        queries: List[str],
        retriever_factory: Callable[[Dict[str, Any]], Any],
        reranker_factory: Optional[Callable[[], Any]] = None,
    ) -> Dict[str, List[TestResult]]:
        """
        Run a suite of test queries against all variants.

        Args:
            queries: List of test queries
            retriever_factory: Factory function that creates retriever from config
            reranker_factory: Optional factory for reranker

        Returns:
            Dictionary mapping variant names to lists of results
        """
        all_results = {variant.name: [] for variant in self.variants}

        for i, query in enumerate(queries):
            print(f"Running test {i + 1}/{len(queries)}: {query[:50]}...")
            results = self.run_test(query, retriever_factory, reranker_factory)

            for variant_name, result in results.items():
                all_results[variant_name].append(result)

        return all_results

    def calculate_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate metrics for each variant.

        Returns:
            Dictionary with metrics per variant
        """
        metrics = {}

        for variant in self.variants:
            variant_results = [r for r in self.results if r.variant_name == variant.name]

            if not variant_results:
                continue

            # Calculate average metrics
            avg_latency = sum(r.latency_ms for r in variant_results) / len(variant_results)
            avg_num_results = sum(r.num_results for r in variant_results) / len(variant_results)
            avg_score = sum(
                sum(r.scores) / len(r.scores) if r.scores else 0 for r in variant_results
            ) / len(variant_results)

            metrics[variant.name] = {
                "avg_latency_ms": round(avg_latency, 2),
                "avg_num_results": round(avg_num_results, 2),
                "avg_score": round(avg_score, 4),
                "total_queries": len(variant_results),
            }

        return metrics

    def compare_variants(self) -> Dict[str, Any]:
        """
        Compare variants against each other.

        Returns:
            Comparison results with winner determination
        """
        metrics = self.calculate_metrics()

        if len(metrics) < 2:
            return {"error": "Need at least 2 variants to compare"}

        # Find best variant for each metric
        comparisons = {}

        # Compare latency (lower is better)
        if metrics:
            fastest = min(metrics.items(), key=lambda x: x[1]["avg_latency_ms"])
            comparisons["fastest"] = {
                "variant": fastest[0],
                "latency_ms": fastest[1]["avg_latency_ms"],
            }

        # Compare scores (higher is better)
        if metrics:
            highest_score = max(metrics.items(), key=lambda x: x[1]["avg_score"])
            comparisons["highest_score"] = {
                "variant": highest_score[0],
                "avg_score": highest_score[1]["avg_score"],
            }

        # Overall recommendation (weighted combination)
        # Lower latency is good, higher score is good
        variant_scores = {}
        for name, m in metrics.items():
            # Normalize scores (simplified)
            score = m["avg_score"] * 100 - m["avg_latency_ms"] / 1000
            variant_scores[name] = score

        if variant_scores:
            recommended = max(variant_scores.items(), key=lambda x: x[1])
            comparisons["recommended"] = {
                "variant": recommended[0],
                "composite_score": round(recommended[1], 2),
            }

        return {
            "metrics": metrics,
            "comparisons": comparisons,
            "variants_tested": list(metrics.keys()),
        }

    def generate_report(self) -> str:
        """
        Generate a human-readable test report.

        Returns:
            Report string
        """
        comparison = self.compare_variants()
        metrics = comparison.get("metrics", {})

        report_lines = [
            "=" * 80,
            "A/B Test Report",
            "=" * 80,
            f"Test ID: {self._test_id}",
            f"Timestamp: {datetime.now().isoformat()}",
            f"Total Queries: {len(set(r.query for r in self.results))}",
            f"Variants Tested: {len(self.variants)}",
            "",
            "Variant Configurations:",
            "-" * 40,
        ]

        for variant in self.variants:
            report_lines.append(f"\n{variant.name}:")
            report_lines.append(f"  Description: {variant.description}")
            report_lines.append(f"  Config: {json.dumps(variant.config, indent=2)}")

        report_lines.extend(
            [
                "",
                "Performance Metrics:",
                "-" * 40,
            ]
        )

        for name, m in metrics.items():
            report_lines.extend(
                [
                    f"\n{name}:",
                    f"  Average Latency: {m['avg_latency_ms']:.2f}ms",
                    f"  Average Results: {m['avg_num_results']:.2f}",
                    f"  Average Score: {m['avg_score']:.4f}",
                    f"  Total Queries: {m['total_queries']}",
                ]
            )

        report_lines.extend(
            [
                "",
                "Comparison Results:",
                "-" * 40,
            ]
        )

        comparisons = comparison.get("comparisons", {})
        for metric, result in comparisons.items():
            report_lines.append(f"  {metric.replace('_', ' ').title()}: {result['variant']}")

        report_lines.extend(
            [
                "",
                "=" * 80,
            ]
        )

        return "\n".join(report_lines)

    def save_results(self, filename: Optional[str] = None) -> str:
        """
        Save test results to file.

        Args:
            filename: Optional filename (default: auto-generated)

        Returns:
            Path to saved file
        """
        if filename is None:
            filename = f"ab_test_{self._test_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        filepath = self.output_dir / filename

        data = {
            "test_id": self._test_id,
            "timestamp": datetime.now().isoformat(),
            "variants": [
                {"name": v.name, "config": v.config, "description": v.description}
                for v in self.variants
            ],
            "results": [asdict(r) for r in self.results],
            "metrics": self.calculate_metrics(),
            "comparison": self.compare_variants(),
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        return str(filepath)

    def load_results(self, filepath: str) -> None:
        """
        Load test results from file.

        Args:
            filepath: Path to results file
        """
        with open(filepath, "r") as f:
            data = json.load(f)

        self._test_id = data["test_id"]
        self.results = [TestResult(**r) for r in data["results"]]

        # Reconstruct variants
        self.variants = [
            TestVariant(name=v["name"], config=v["config"], description=v.get("description", ""))
            for v in data["variants"]
        ]
