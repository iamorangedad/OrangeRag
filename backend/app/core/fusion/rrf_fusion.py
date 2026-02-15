"""Reciprocal Rank Fusion (RRF) algorithm implementation."""

from typing import List
from collections import defaultdict

from app.core.retrievers.base import NodeWithScore


def reciprocal_rank_fusion(
    results_list: List[List[NodeWithScore]], k: float = 60.0
) -> List[NodeWithScore]:
    """
    Reciprocal Rank Fusion (RRF) algorithm.

    RRF combines multiple ranked lists by computing a fusion score for each document:
    score = sum(1 / (k + rank_i)) for each result list i

    This approach is simple yet effective and doesn't require training.
    The constant k (default 60) helps dampen the impact of high rankings
    from individual retrievers.

    Reference:
        Cormack, G. V., Clarke, C. L., & Buettcher, S. (2009).
        "Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods"
        SIGIR 2009.

    Args:
        results_list: List of retrieval results from different retrievers
        k: RRF constant (default: 60.0)
           Higher values reduce the impact of top rankings

    Returns:
        Fused list of nodes with RRF scores, sorted by score descending
    """
    if not results_list:
        return []

    # Dictionary to accumulate RRF scores for each document
    # Key: node_id, Value: (accumulated_score, node)
    rrf_scores: dict[str, tuple[float, NodeWithScore]] = defaultdict(lambda: (0.0, None))

    # Process each result list
    for results in results_list:
        for rank, node_with_score in enumerate(results):
            node_id = node_with_score.id_

            # Calculate RRF score contribution from this rank
            # rank is 0-indexed, so add 1 to make it 1-indexed
            rrf_score = 1.0 / (k + rank + 1)

            # Accumulate score
            current_score, _ = rrf_scores[node_id]
            rrf_scores[node_id] = (current_score + rrf_score, node_with_score)

    # Sort by RRF score descending
    sorted_results = sorted(
        rrf_scores.items(),
        key=lambda x: x[1][0],  # Sort by accumulated score
        reverse=True,
    )

    # Build final result list
    fused_results = []
    for node_id, (score, node_with_score) in sorted_results:
        # Create new NodeWithScore with RRF score
        fused_node = NodeWithScore(node=node_with_score.node, score=score)
        fused_results.append(fused_node)

    return fused_results


def weighted_reciprocal_rank_fusion(
    results_list: List[List[NodeWithScore]], weights: List[float], k: float = 60.0
) -> List[NodeWithScore]:
    """
    Weighted RRF that allows different weights for each retriever.

    Args:
        results_list: List of retrieval results
        weights: List of weights corresponding to each result list
        k: RRF constant (default: 60.0)

    Returns:
        Fused list of nodes with weighted RRF scores

    Raises:
        ValueError: If weights length doesn't match results_list length
    """
    if len(results_list) != len(weights):
        raise ValueError(
            f"Number of weights ({len(weights)}) must match "
            f"number of result lists ({len(results_list)})"
        )

    if not results_list:
        return []

    # Dictionary to accumulate weighted RRF scores
    rrf_scores: dict[str, tuple[float, NodeWithScore]] = defaultdict(lambda: (0.0, None))

    # Process each result list with its weight
    for results, weight in zip(results_list, weights):
        for rank, node_with_score in enumerate(results):
            node_id = node_with_score.id_

            # Calculate weighted RRF score
            rrf_score = weight * (1.0 / (k + rank + 1))

            # Accumulate score
            current_score, _ = rrf_scores[node_id]
            rrf_scores[node_id] = (current_score + rrf_score, node_with_score)

    # Sort and build results
    sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1][0], reverse=True)

    # Filter out zero-score results and build final list
    return [
        NodeWithScore(node=node.node, score=score)
        for _, (score, node) in sorted_results
        if score > 0
    ]
