# Hybrid RAG Kubernetes Quick Reference

## Quick Start

### 1. Basic Deployment

```bash
# Deploy with default Hybrid RAG settings
kubectl apply -f deployment/deployment.yaml
```

### 2. Check Deployment

```bash
# Watch pods
kubectl get pods -n doc-chat-system -w

# Check logs
kubectl logs -n doc-chat-system -l app=doc-chat --tail=50

# Verify Hybrid RAG is active
kubectl logs -n doc-chat-system -l app=doc-chat | grep "HybridChat"
```

## Common Configurations

### Configuration 1: Default (Balanced)

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: doc-chat-config
  namespace: doc-chat-system
data:
  # Basic
  VECTOR_STORE_TYPE: "chroma"
  USE_CHROMA: "true"
  
  # Hybrid RAG - Balanced
  ENABLE_HYBRID_SEARCH: "true"
  DENSE_TOP_K: "10"
  SPARSE_TOP_K: "10"
  FINAL_TOP_K: "5"
  FUSION_MODE: "rrf"
  BM25_CACHE_ENABLED: "true"
  
  # Optional features disabled
  ENABLE_RERANK: "false"
  ENABLE_QUERY_EXPANSION: "false"
```

### Configuration 2: Technical Docs (Keyword Heavy)

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: doc-chat-config
  namespace: doc-chat-system
data:
  # Technical documentation - favor keyword matching
  ENABLE_HYBRID_SEARCH: "true"
  FUSION_MODE: "weighted"
  DENSE_WEIGHT: "0.3"
  SPARSE_WEIGHT: "0.7"
  SPARSE_TOP_K: "20"
  ENABLE_QUERY_EXPANSION: "true"
  QUERY_EXPANSION_TYPE: "synonym"
```

### Configuration 3: Conversational (Semantic Heavy)

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: doc-chat-config
  namespace: doc-chat-system
data:
  # Conversational - favor semantic understanding
  ENABLE_HYBRID_SEARCH: "true"
  FUSION_MODE: "weighted"
  DENSE_WEIGHT: "0.7"
  SPARSE_WEIGHT: "0.3"
  DENSE_TOP_K: "20"
  ENABLE_QUERY_EXPANSION: "true"
  QUERY_EXPANSION_TYPE: "keyword"
```

### Configuration 4: High Performance

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: doc-chat-config
  namespace: doc-chat-system
data:
  # Fast response - minimal retrieval
  ENABLE_HYBRID_SEARCH: "true"
  DENSE_TOP_K: "5"
  SPARSE_TOP_K: "5"
  FINAL_TOP_K: "3"
  BM25_CACHE_ENABLED: "true"
  # Disable optional features for speed
  ENABLE_RERANK: "false"
  ENABLE_QUERY_EXPANSION: "false"
```

### Configuration 5: Maximum Quality

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: doc-chat-config
  namespace: doc-chat-system
data:
  # Best results - deeper retrieval with reranking
  ENABLE_HYBRID_SEARCH: "true"
  DENSE_TOP_K: "50"
  SPARSE_TOP_K: "50"
  FINAL_TOP_K: "20"
  ENABLE_RERANK: "true"
  RERANK_MODEL: "cross-encoder/ms-marco-MiniLM-L-6-v2"
  ENABLE_QUERY_EXPANSION: "true"
  QUERY_EXPANSION_TYPE: "multi"
  QUERY_EXPANSION_MAX: "5"
```

## Environment Variables Quick Reference

```bash
# Apply configuration changes
kubectl apply -f - <<EOF
apiVersion: v1
kind: ConfigMap
metadata:
  name: doc-chat-config
  namespace: doc-chat-system
data:
  # Core Hybrid RAG
  ENABLE_HYBRID_SEARCH: "true"
  DENSE_TOP_K: "10"
  SPARSE_TOP_K: "10"
  FINAL_TOP_K: "5"
  RRF_K: "60.0"
  FUSION_MODE: "rrf"
  DENSE_WEIGHT: "0.5"
  SPARSE_WEIGHT: "0.5"
  
  # Cache
  BM25_CACHE_ENABLED: "true"
  BM25_CACHE_DIR: "/app/chroma_db/bm25_cache"
  
  # Optional Features
  ENABLE_RERANK: "false"
  RERANK_MODEL: "cross-encoder/ms-marco-MiniLM-L-6-v2"
  ENABLE_QUERY_EXPANSION: "false"
  QUERY_EXPANSION_TYPE: "synonym"
  QUERY_EXPANSION_MAX: "3"
EOF

# Restart deployment to apply
kubectl rollout restart deployment/doc-chat -n doc-chat-system
```

## Resource Requirements

### Minimum (Testing)

```yaml
resources:
  requests:
    memory: "2Gi"
    cpu: "1000m"
  limits:
    memory: "4Gi"
    cpu: "2000m"
```

### Recommended (Production)

```yaml
resources:
  requests:
    memory: "4Gi"
    cpu: "2000m"
  limits:
    memory: "8Gi"
    cpu: "4000m"
```

### With Reranking Enabled

```yaml
resources:
  requests:
    memory: "6Gi"  # Extra for model
    cpu: "2000m"
  limits:
    memory: "12Gi"  # Extra for model
    cpu: "4000m"
```

## Storage Requirements

```yaml
# PVC sizes for different document volumes

# Small (< 1000 docs)
resources:
  requests:
    storage: 5Gi  # uploads
    storage: 10Gi # chroma + cache

# Medium (1000-10000 docs)
resources:
  requests:
    storage: 10Gi  # uploads
    storage: 50Gi  # chroma + cache

# Large (> 10000 docs)
resources:
  requests:
    storage: 50Gi   # uploads
    storage: 200Gi  # chroma + cache
```

## Troubleshooting Commands

```bash
# Check if Hybrid RAG is working
kubectl logs -n doc-chat-system -l app=doc-chat | grep -i "hybrid"

# Check BM25 cache
kubectl exec -n doc-chat-system deployment/doc-chat -- ls -la /app/chroma_db/bm25_cache/

# Check cache statistics
kubectl exec -n doc-chat-system deployment/doc-chat -- python -c "
from app.core.cache.bm25_cache import BM25IndexCache
cache = BM25IndexCache('/app/chroma_db/bm25_cache')
print(cache.get_cache_stats())
"

# Test retrieval
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "test query"}'

# Check memory usage
kubectl top pod -n doc-chat-system

# Check disk usage
kubectl exec -n doc-chat-system deployment/doc-chat -- df -h
```

## Performance Tuning

### Check Current Performance

```bash
# View logs for timing information
kubectl logs -n doc-chat-system -l app=doc-chat | grep -E "retrieval|latency|time"
```

### Tune Based on Load

**High Query Volume:**
```yaml
DENSE_TOP_K: "5"        # Reduce
SPARSE_TOP_K: "5"       # Reduce
FINAL_TOP_K: "3"        # Reduce
BM25_CACHE_ENABLED: "true"  # Critical
ENABLE_RERANK: "false"  # Disable
```

**High Recall Needed:**
```yaml
DENSE_TOP_K: "50"       # Increase
SPARSE_TOP_K: "50"      # Increase
FINAL_TOP_K: "20"       # Increase
ENABLE_RERANK: "true"   # Enable
```

**Mixed Workload:**
```yaml
FUSION_MODE: "weighted"
DENSE_WEIGHT: "0.6"
SPARSE_WEIGHT: "0.4"
```

## Migration Checklist

- [ ] Update ConfigMap with Hybrid RAG variables
- [ ] Mount Chroma PVC to doc-chat pod
- [ ] Verify sufficient disk space for cache
- [ ] Check memory limits (add 500MB-1GB for Hybrid RAG)
- [ ] Deploy and verify logs show "HybridChat"
- [ ] Test queries and verify cache is created
- [ ] Monitor performance metrics

## Useful Aliases

```bash
# Add to .bashrc or .zshrc
alias kdoc='kubectl -n doc-chat-system'
alias kdoclogs='kubectl logs -n doc-chat-system -l app=doc-chat --tail=100 -f'
alias kdochybrid='kubectl logs -n doc-chat-system -l app=doc-chat | grep -i hybrid'
alias kdoccache='kubectl exec -n doc-chat-system deployment/doc-chat -- ls -la /app/chroma_db/bm25_cache/'
```
