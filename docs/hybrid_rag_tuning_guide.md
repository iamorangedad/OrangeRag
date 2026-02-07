# Hybrid RAG Configuration Tuning Guide

This guide provides recommendations for tuning Hybrid RAG parameters based on your use case and performance requirements.

## Overview

Hybrid RAG combines dense (vector) and sparse (BM25) retrieval methods. The key parameters control:
- Retrieval depth from each method
- Fusion algorithm behavior
- Caching strategy

## Core Parameters

### 1. Retrieval Configuration

#### dense_top_k / sparse_top_k
- **Default**: 10
- **Range**: 5-50
- **Description**: Number of documents retrieved from each method before fusion

**Recommendations**:
- **Small document collections (<1000 docs)**: Use 5-10
  - Faster retrieval with less noise
- **Medium collections (1000-10000 docs)**: Use 10-20
  - Balanced recall and precision
- **Large collections (>10000 docs)**: Use 20-50
  - Ensures good coverage before fusion

#### final_top_k
- **Default**: 5
- **Range**: 3-20
- **Description**: Number of documents after RRF fusion to send to LLM

**Recommendations**:
- **Concise answers**: 3-5 documents
  - Reduces context window usage and cost
- **Comprehensive answers**: 5-10 documents
  - Better coverage of relevant information
- **Research/Analysis**: 10-20 documents
  - Maximum recall for complex queries

### 2. Fusion Configuration

#### fusion_mode
- **Default**: "rrf" (Reciprocal Rank Fusion)
- **Options**: "rrf", "weighted"

**rrf mode**:
- Equal weighting for both retrieval methods
- Good default for most use cases
- Simple and robust

**weighted mode**:
- Allows custom weighting with `dense_weight` and `sparse_weight`
- Use when one method is more reliable for your domain

#### dense_weight / sparse_weight (weighted mode only)
- **Default**: 0.5 / 0.5
- **Range**: 0.0-1.0 (sum should be 1.0)

**When to adjust**:
- **More semantic similarity**: dense_weight=0.7, sparse_weight=0.3
  - Good for conceptual/conversational queries
- **More keyword matching**: dense_weight=0.3, sparse_weight=0.7
  - Good for technical/specific term queries
- **Balanced**: dense_weight=0.5, sparse_weight=0.5
  - Good default for mixed query types

#### rrf_k
- **Default**: 60.0
- **Range**: 10-100
- **Description**: RRF damping constant (higher = less aggressive ranking)

**Recommendations**:
- **Small collections**: k=10-30
  - More aggressive rank boosting for top results
- **Medium collections**: k=40-60 (default)
  - Balanced behavior (paper recommendation)
- **Large collections**: k=60-100
  - Smoother ranking curve, less sensitive to position

### 3. BM25 Parameters

#### bm25_k1
- **Default**: 1.5
- **Range**: 0.5-2.0
- **Description**: Term frequency saturation

**Effect**:
- **Lower (0.5-1.0)**: Documents with more keyword matches don't get much higher scores
- **Higher (1.5-2.0)**: Documents with more keyword matches get significantly higher scores

**Recommendations**:
- **Short documents**: k1=0.5-1.0
- **Medium documents**: k1=1.2-1.5 (default)
- **Long documents**: k1=1.5-2.0

#### bm25_b
- **Default**: 0.75
- **Range**: 0.0-1.0
- **Description**: Document length normalization

**Effect**:
- **0.0**: No length normalization (long docs favored)
- **0.75**: Moderate normalization (default)
- **1.0**: Full normalization (short docs favored)

**Recommendations**:
- **Similar length documents**: b=0.5-0.75
- **Variable length documents**: b=0.75-1.0

### 4. Cache Configuration

#### bm25_cache_enabled
- **Default**: true
- **Description**: Enable caching of BM25 indices to disk

**Benefits**:
- 10-50x faster subsequent loads
- Reduced CPU usage on startup

**When to disable**:
- Documents change frequently
- Disk space is limited
- Maximum privacy required (no disk persistence)

#### bm25_cache_dir
- **Default**: Uses chroma_dir/bm25_cache
- **Description**: Directory for cache files

**Recommendations**:
- Use fast SSD storage for large collections
- Ensure adequate disk space (typically 10-50% of document size)

## Use Case Presets

### 1. General Purpose (Default)
```
dense_top_k=10
sparse_top_k=10
final_top_k=5
fusion_mode=rrf
rrf_k=60.0
bm25_k1=1.5
bm25_b=0.75
bm25_cache_enabled=true
```

### 2. Technical Documentation
```
dense_top_k=15
sparse_top_k=20
final_top_k=8
fusion_mode=weighted
dense_weight=0.3
sparse_weight=0.7
rrf_k=60.0
bm25_k1=1.8
bm25_b=0.8
bm25_cache_enabled=true
```

### 3. Conversational AI
```
dense_top_k=20
sparse_top_k=15
final_top_k=5
fusion_mode=weighted
dense_weight=0.7
sparse_weight=0.3
rrf_k=60.0
bm25_k1=1.2
bm25_b=0.75
bm25_cache_enabled=true
```

### 4. High-Performance (Low Latency)
```
dense_top_k=5
sparse_top_k=5
final_top_k=3
fusion_mode=rrf
rrf_k=30.0
bm25_k1=1.5
bm25_b=0.75
bm25_cache_enabled=true
```

### 5. Maximum Recall
```
dense_top_k=50
sparse_top_k=50
final_top_k=20
fusion_mode=rrf
rrf_k=80.0
bm25_k1=1.5
bm25_b=0.75
bm25_cache_enabled=true
```

## Environment Variables

All parameters can be configured via environment variables:

```bash
# Retrieval
export DENSE_TOP_K=10
export SPARSE_TOP_K=10
export FINAL_TOP_K=5

# Fusion
export FUSION_MODE=rrf
export DENSE_WEIGHT=0.5
export SPARSE_WEIGHT=0.5
export RRF_K=60.0

# BM25
export BM25_K1=1.5
export BM25_B=0.75

# Cache
export BM25_CACHE_ENABLED=true
export BM25_CACHE_DIR=/path/to/cache
```

## Performance Tuning Tips

1. **Start with defaults** - They work well for most cases
2. **Adjust top_k values** - Most impactful for performance/recall tradeoff
3. **Enable caching** - Provides significant speedup for repeated loads
4. **Profile with benchmarks** - Use the benchmark script to measure impact
5. **Tune fusion weights** - Only if you have clear domain requirements

## Monitoring

Monitor these metrics to validate your configuration:
- **Retrieval latency** - Should be <200ms for most queries
- **Cache hit rate** - Should be >90% for repeated workloads
- **Recall** - Fraction of relevant documents found
- **Precision** - Fraction of retrieved documents that are relevant

## Troubleshooting

### Slow Retrieval
- Reduce `dense_top_k` and `sparse_top_k`
- Enable caching (`bm25_cache_enabled=true`)
- Consider async retrieval (`aretrieve` method)

### Low Recall
- Increase `dense_top_k` and `sparse_top_k`
- Increase `final_top_k`
- Try weighted fusion with adjusted weights

### Irrelevant Results
- Reduce `final_top_k`
- Adjust fusion weights based on query type
- Tune BM25 parameters for your document lengths
