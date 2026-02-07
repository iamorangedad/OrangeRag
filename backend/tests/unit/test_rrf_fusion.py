"""Unit tests for RRF fusion algorithm."""

import pytest
from llama_index.core.schema import TextNode

from app.core.fusion.rrf_fusion import (
    reciprocal_rank_fusion,
    weighted_reciprocal_rank_fusion,
)
from app.core.retrievers.base import NodeWithScore


class TestReciprocalRankFusion:
    """Test RRF fusion algorithm."""

    @pytest.fixture
    def create_node(self):
        """Helper to create NodeWithScore."""

        def _create(node_id: str, text: str, score: float = 0.0):
            node = TextNode(id_=node_id, text=text)
            return NodeWithScore(node=node, score=score)

        return _create

    def test_empty_results(self):
        """Test fusion with empty results."""
        results = reciprocal_rank_fusion([])
        assert results == []

    def test_single_result_list(self, create_node):
        """Test fusion with single result list."""
        results = [
            create_node("1", "doc1", 0.9),
            create_node("2", "doc2", 0.8),
            create_node("3", "doc3", 0.7),
        ]

        fused = reciprocal_rank_fusion([results])

        assert len(fused) == 3
        # Order should be preserved
        assert fused[0].id_ == "1"
        assert fused[1].id_ == "2"
        assert fused[2].id_ == "3"

    def test_two_result_lists_no_overlap(self, create_node):
        """Test fusion with two non-overlapping result lists."""
        list1 = [
            create_node("1", "doc1", 0.9),
            create_node("2", "doc2", 0.8),
        ]
        list2 = [
            create_node("3", "doc3", 0.85),
            create_node("4", "doc4", 0.75),
        ]

        fused = reciprocal_rank_fusion([list1, list2])

        assert len(fused) == 4
        # All documents should be present
        ids = {r.id_ for r in fused}
        assert ids == {"1", "2", "3", "4"}

    def test_two_result_lists_with_overlap(self, create_node):
        """Test fusion with overlapping result lists."""
        list1 = [
            create_node("1", "doc1", 0.9),
            create_node("2", "doc2", 0.8),
            create_node("3", "doc3", 0.7),
        ]
        list2 = [
            create_node("2", "doc2", 0.85),  # Overlapping
            create_node("1", "doc1", 0.75),  # Overlapping
            create_node("4", "doc4", 0.9),
        ]

        fused = reciprocal_rank_fusion([list1, list2])

        assert len(fused) == 4
        # Document 2 should be ranked high (appears in both)
        # Document 1 should also be ranked high (appears in both)

    def test_rrf_score_calculation(self, create_node):
        """Test RRF score calculation."""
        # Doc1: rank 0 in list1, rank 1 in list2
        # Doc2: rank 1 in list1, rank 0 in list2
        list1 = [
            create_node("1", "doc1", 0.9),
            create_node("2", "doc2", 0.8),
        ]
        list2 = [
            create_node("2", "doc2", 0.85),
            create_node("1", "doc1", 0.75),
        ]

        k = 60.0
        fused = reciprocal_rank_fusion([list1, list2], k=k)

        # Both documents appear in both lists
        # Doc1: 1/(60+1) + 1/(60+2) = 1/61 + 1/62
        # Doc2: 1/(60+2) + 1/(60+1) = 1/62 + 1/61
        # They should have the same score

        doc1 = next(r for r in fused if r.id_ == "1")
        doc2 = next(r for r in fused if r.id_ == "2")

        assert abs(doc1.score - doc2.score) < 0.0001

    def test_k_parameter_effect(self, create_node):
        """Test different k values."""
        list1 = [create_node("1", "doc1", 0.9)]
        list2 = [create_node("1", "doc1", 0.8)]

        fused_k60 = reciprocal_rank_fusion([list1, list2], k=60.0)
        fused_k10 = reciprocal_rank_fusion([list1, list2], k=10.0)

        # Same document, different k should produce different scores
        assert fused_k60[0].score != fused_k10[0].score

    def test_multiple_lists(self, create_node):
        """Test fusion with multiple result lists."""
        lists = [
            [create_node("1", "doc1", 0.9)],
            [create_node("1", "doc1", 0.8)],
            [create_node("1", "doc1", 0.7)],
        ]

        fused = reciprocal_rank_fusion(lists)

        assert len(fused) == 1
        # Score should be sum of contributions from all lists
        assert fused[0].score > 0

    def test_result_ordering(self, create_node):
        """Test that results are properly ordered by RRF score."""
        # List1: A(rank 0), B(rank 1), C(rank 2)
        # List2: C(rank 0), B(rank 1)
        # Expected: B should be highest (appears in both, avg rank ~1)
        list1 = [
            create_node("A", "docA", 0.9),
            create_node("B", "docB", 0.8),
            create_node("C", "docC", 0.7),
        ]
        list2 = [
            create_node("C", "docC", 0.95),
            create_node("B", "docB", 0.85),
        ]

        fused = reciprocal_rank_fusion([list1, list2])

        # Check that results are sorted by score descending
        for i in range(len(fused) - 1):
            assert fused[i].score >= fused[i + 1].score


class TestWeightedReciprocalRankFusion:
    """Test weighted RRF fusion algorithm."""

    @pytest.fixture
    def create_node(self):
        """Helper to create NodeWithScore."""

        def _create(node_id: str, text: str, score: float = 0.0):
            node = TextNode(id_=node_id, text=text)
            return NodeWithScore(node=node, score=score)

        return _create

    def test_equal_weights(self, create_node):
        """Test with equal weights."""
        list1 = [create_node("1", "doc1", 0.9)]
        list2 = [create_node("1", "doc1", 0.8)]

        weighted = weighted_reciprocal_rank_fusion([list1, list2], [0.5, 0.5])
        standard = reciprocal_rank_fusion([list1, list2])

        # With equal weights, results should be proportional
        assert len(weighted) == len(standard)

    def test_unequal_weights(self, create_node):
        """Test with different weights."""
        list1 = [create_node("1", "doc1", 0.9)]
        list2 = [create_node("2", "doc2", 0.9)]

        # Give more weight to list1
        fused = weighted_reciprocal_rank_fusion([list1, list2], [0.8, 0.2])

        assert len(fused) == 2
        # Doc1 should have higher score due to higher weight
        doc1_score = next(r.score for r in fused if r.id_ == "1")
        doc2_score = next(r.score for r in fused if r.id_ == "2")
        assert doc1_score > doc2_score

    def test_zero_weight(self, create_node):
        """Test with zero weight for one list."""
        list1 = [create_node("1", "doc1", 0.9)]
        list2 = [create_node("2", "doc2", 0.9)]

        # Give zero weight to list2
        fused = weighted_reciprocal_rank_fusion([list1, list2], [1.0, 0.0])

        # Only list1 should contribute
        assert len(fused) == 1
        assert fused[0].id_ == "1"

    def test_weight_mismatch_error(self, create_node):
        """Test error when weights don't match results."""
        list1 = [create_node("1", "doc1", 0.9)]
        list2 = [create_node("2", "doc2", 0.8)]

        with pytest.raises(ValueError, match="Number of weights"):
            weighted_reciprocal_rank_fusion([list1, list2], [0.5])

    def test_weights_sum_greater_than_one(self, create_node):
        """Test with weights sum > 1."""
        list1 = [create_node("1", "doc1", 0.9)]
        list2 = [create_node("1", "doc1", 0.8)]

        # Weights sum to 2.0
        fused = weighted_reciprocal_rank_fusion([list1, list2], [1.0, 1.0])

        # Should still work, just with higher scores
        assert len(fused) == 1
        assert fused[0].score > 0
