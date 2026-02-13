# Kubernetes Deployment Guide

This guide covers deploying the Smart Document Assistant on Kubernetes with the new modular architecture.

## Architecture Overview

The application now supports multiple vector store backends through a pluggable architecture:

```
┌─────────────────────────────────────────────────────────────┐
│                    Kubernetes Cluster                       │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                  Namespace: doc-chat-system         │   │
│  │                                                     │   │
│  │  ┌──────────────┐         ┌──────────────┐         │   │
│  │  │  doc-chat    │────────▶│   chroma     │         │   │
│  │  │  (Backend)   │         │(Vector Store)│         │   │
│  │  │  Port: 8000  │         │  Port: 8000  │         │   │
│  │  └──────────────┘         └──────────────┘         │   │
│  │         │                       │                  │   │
│  │         ▼                       ▼                  │   │
│  │  ┌──────────────┐         ┌──────────────┐         │   │
│  │  │ uploads-pvc  │         │ chroma-pvc   │         │   │
│  │  │   (5Gi)      │         │   (10Gi)     │         │   │
│  │  └──────────────┘         └──────────────┘         │   │
│  │                                                     │   │
│  └─────────────────────────────────────────────────────┘   │
│                            │                                │
│                            ▼                                │
│              External Ollama Service                       │
│              (http://10.0.0.55:11434)                      │
└─────────────────────────────────────────────────────────────┘
```

## Deployment Scenarios

### Scenario 1: Simple Mode (No Chroma)
Use this for testing or when persistence is not required.

```yaml
# ConfigMap overrides
VECTOR_STORE_TYPE: "simple"
USE_CHROMA: "false"
```

**Pros:**
- No ChromaDB pod required
- Faster startup
- Lower resource usage

**Cons:**
- Vector embeddings lost on pod restart
- In-memory only

### Scenario 2: Chroma Mode with Hybrid RAG (Recommended for Production)
Use this for persistent vector storage with Hybrid RAG enabled.

```yaml
# ConfigMap overrides
VECTOR_STORE_TYPE: "chroma"
USE_CHROMA: "true"
CHROMA_HOST: "chroma"
CHROMA_PORT: "8000"

# Hybrid RAG Configuration
ENABLE_HYBRID_SEARCH: "true"
DENSE_TOP_K: "10"
SPARSE_TOP_K: "10"
FINAL_TOP_K: "5"
FUSION_MODE: "rrf"
BM25_CACHE_ENABLED: "true"
```

**Pros:**
- Persistent embeddings survive restarts
- Supports large document collections
- Better query performance
- **Hybrid RAG**: Combines semantic + keyword search for better recall
- **BM25 Caching**: 10-50x faster subsequent loads
- **Optional Reranking**: Fine-grained relevance scoring

**Cons:**
- Requires additional ChromaDB pod
- Higher resource usage
- BM25 cache requires additional disk space (~10-50% of document size)

## Quick Deployment

### 1. Build Docker Image

```bash
# Build the image
docker build -t doc-chat:latest .

# Tag for registry (optional)
docker tag doc-chat:latest your-registry/doc-chat:v1.0.0
docker push your-registry/doc-chat:v1.0.0
```

### 2. Update Deployment Configuration

Edit `deployment/deployment.yaml` to customize:

#### a. Ollama Configuration
```yaml
data:
  OLLAMA_BASE_URL: "http://your-ollama-host:11434"
  MODEL_NAME: "qwen3:4b"  # or your preferred model
  EMBEDDING_MODEL: "nomic-embed-text"
```

#### b. Vector Store Selection
```yaml
data:
  # For Chroma (persistent)
  VECTOR_STORE_TYPE: "chroma"
  USE_CHROMA: "true"
  
  # For Simple mode (in-memory)
  # VECTOR_STORE_TYPE: "simple"
  # USE_CHROMA: "false"
```

#### c. CORS Settings
```yaml
data:
  # For development (allows all)
  ALLOWED_ORIGINS: ""
  
  # For production (specific origins)
  ALLOWED_ORIGINS: "https://yourdomain.com,https://app.yourdomain.com"
```

#### d. Resource Limits
```yaml
resources:
  requests:
    memory: "2Gi"    # Minimum recommended
    cpu: "1000m"
  limits:
    memory: "4Gi"    # Adjust based on document count
    cpu: "2000m"
```

### 3. Apply Deployment

```bash
kubectl apply -f deployment/deployment.yaml
```

### 4. Verify Deployment

```bash
# Check all resources
kubectl get all -n doc-chat-system

# Check pods
kubectl get pods -n doc-chat-system

# Check logs
kubectl logs -n doc-chat-system -l app=doc-chat --tail=50
kubectl logs -n doc-chat-system -l app=chroma --tail=20

# Check PVCs
kubectl get pvc -n doc-chat-system

# Test service
kubectl port-forward -n doc-chat-system svc/doc-chat 8080:80
curl http://localhost:8080/health
```

## Configuration Reference

### ConfigMap: doc-chat-config

| Key | Description | Default |
|-----|-------------|---------|
| `OLLAMA_BASE_URL` | Ollama service URL | `http://10.0.0.55:11434` |
| `MODEL_NAME` | Default LLM model | `qwen3:4b` |
| `EMBEDDING_MODEL` | Default embedding model | `nomic-embed-text` |
| `VECTOR_STORE_TYPE` | Vector store type (`simple` or `chroma`) | `chroma` |
| `USE_CHROMA` | Legacy flag (overrides VECTOR_STORE_TYPE if `true`) | `true` |
| `CHROMA_HOST` | ChromaDB service hostname | `chroma` |
| `CHROMA_PORT` | ChromaDB service port | `8000` |
| `CHROMA_COLLECTION` | Chroma collection name | `documents` |
| `UPLOAD_DIR` | Document storage path | `/app/uploads` |
| `CHROMA_DIR` | Chroma persistence path | `/app/chroma_db` |
| `STATIC_DIR` | Static files path | `/app/static` |
| `ALLOWED_ORIGINS` | CORS allowed origins | `` (empty = all) |
| `DEBUG` | Debug mode | `false` |

### Hybrid RAG Configuration (New)

| Key | Description | Default |
|-----|-------------|---------|
| `ENABLE_HYBRID_SEARCH` | Enable Hybrid RAG retrieval | `true` |
| `DENSE_TOP_K` | Number of results from dense retrieval | `10` |
| `SPARSE_TOP_K` | Number of results from sparse retrieval | `10` |
| `FINAL_TOP_K` | Number of final results after fusion | `5` |
| `RRF_K` | RRF fusion constant | `60.0` |
| `DENSE_WEIGHT` | Weight for dense retrieval (weighted mode) | `0.5` |
| `SPARSE_WEIGHT` | Weight for sparse retrieval (weighted mode) | `0.5` |
| `FUSION_MODE` | Fusion algorithm (`rrf` or `weighted`) | `rrf` |
| `BM25_CACHE_ENABLED` | Enable BM25 index caching | `true` |
| `BM25_CACHE_DIR` | BM25 cache directory | `/app/chroma_db/bm25_cache` |
| `ENABLE_RERANK` | Enable Cross-Encoder reranking | `false` |
| `RERANK_MODEL` | Reranker model name | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| `ENABLE_QUERY_EXPANSION` | Enable query expansion | `false` |
| `QUERY_EXPANSION_TYPE` | Expansion type (`synonym`, `keyword`, `hyde`, `multi`) | `synonym` |
| `QUERY_EXPANSION_MAX` | Maximum expanded queries | `3` |

## Hybrid RAG Configuration Guide

### What is Hybrid RAG?

Hybrid RAG combines **dense retrieval** (vector embeddings) and **sparse retrieval** (BM25 keywords) to provide better search results:

- **Dense**: Good for semantic similarity and context understanding
- **Sparse**: Good for exact keyword matching and technical terms
- **RRF Fusion**: Intelligently combines both without training

### Basic Configuration

For most use cases, the defaults work well:

```yaml
ENABLE_HYBRID_SEARCH: "true"
DENSE_TOP_K: "10"
SPARSE_TOP_K: "10"
FINAL_TOP_K: "5"
FUSION_MODE: "rrf"
BM25_CACHE_ENABLED: "true"
```

### Advanced Configuration

#### 1. Tuning for Technical Documentation

When searching technical docs with specific terminology:

```yaml
# Favor keyword matching for technical terms
FUSION_MODE: "weighted"
DENSE_WEIGHT: "0.3"
SPARSE_WEIGHT: "0.7"
SPARSE_TOP_K: "20"
ENABLE_QUERY_EXPANSION: "true"
QUERY_EXPANSION_TYPE: "synonym"
```

#### 2. Tuning for Conversational Queries

For chatbot-style conversational queries:

```yaml
# Favor semantic understanding
FUSION_MODE: "weighted"
DENSE_WEIGHT: "0.7"
SPARSE_WEIGHT: "0.3"
DENSE_TOP_K: "20"
ENABLE_QUERY_EXPANSION: "true"
QUERY_EXPANSION_TYPE: "keyword"
```

#### 3. High-Performance Mode

For low-latency requirements:

```yaml
# Reduce retrieval depth
DENSE_TOP_K: "5"
SPARSE_TOP_K: "5"
FINAL_TOP_K: "3"
BM25_CACHE_ENABLED: "true"
ENABLE_RERANK: "false"
ENABLE_QUERY_EXPANSION: "false"
```

#### 4. Maximum Quality Mode

For research/analysis requiring best results:

```yaml
# Deep retrieval with reranking
DENSE_TOP_K: "50"
SPARSE_TOP_K: "50"
FINAL_TOP_K: "20"
ENABLE_RERANK: "true"
RERANK_MODEL: "cross-encoder/ms-marco-MiniLM-L-6-v2"
ENABLE_QUERY_EXPANSION: "true"
QUERY_EXPANSION_TYPE: "multi"
```

### Cache Configuration

The BM25 cache significantly speeds up subsequent deployments:

```yaml
BM25_CACHE_ENABLED: "true"
BM25_CACHE_DIR: "/app/chroma_db/bm25_cache"
```

**Cache Benefits:**
- 10-50x faster index loading on restarts
- Automatic invalidation when documents change
- Stored in persistent volume (survives pod restarts)

**Storage Requirements:**
- Cache size: ~10-50% of document text size
- Ensure sufficient PVC space for both documents and cache

### Query Expansion Types

| Type | Description | Best For |
|------|-------------|----------|
| `synonym` | Replaces words with synonyms | General queries |
| `keyword` | Adds formal/informal variations | How-to queries |
| `hyde` | Generates hypothetical documents | Complex research queries |
| `multi` | Combines multiple strategies | Mixed query types |

### Reranking

Enable reranking for better result quality (adds 100-500ms latency):

```yaml
ENABLE_RERANK: "true"
RERANK_MODEL: "cross-encoder/ms-marco-MiniLM-L-6-v2"
```

**Note:** Requires `sentence-transformers` Python package.

### Persistent Volume Claims

| PVC Name | Purpose | Size | Access Mode |
|----------|---------|------|-------------|
| `uploads-pvc` | Document storage | 5Gi | ReadWriteOnce |
| `chroma-pvc` | Vector embeddings | 10Gi | ReadWriteOnce |

## Advanced Configuration

### Using External ChromaDB

If you have an existing ChromaDB instance:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: doc-chat-config
data:
  VECTOR_STORE_TYPE: "chroma"
  CHROMA_HOST: "external-chroma.example.com"  # External host
  CHROMA_PORT: "8000"
```

Remove the Chroma deployment from the YAML if using external service.

### Node Affinity

Current deployment uses soft affinity (preferred but not required):

```yaml
affinity:
  nodeAffinity:
    preferredDuringSchedulingIgnoredDuringExecution:
    - weight: 100
      preference:
        matchExpressions:
        - key: kubernetes.io/hostname
          operator: In
          values:
          - "ubuntu"  # Change to your preferred node
```

To use hard affinity (required):

```yaml
affinity:
  nodeAffinity:
    requiredDuringSchedulingIgnoredDuringExecution:
      nodeSelectorTerms:
      - matchExpressions:
        - key: kubernetes.io/hostname
          operator: In
          values:
          - "ubuntu"
```

### Resource Tuning

**Small Deployment (Testing):**
```yaml
resources:
  requests:
    memory: "1Gi"
    cpu: "500m"
  limits:
    memory: "2Gi"
    cpu: "1000m"
```

**Large Deployment (Production):**
```yaml
resources:
  requests:
    memory: "4Gi"
    cpu: "2000m"
  limits:
    memory: "8Gi"
    cpu: "4000m"
```

## Troubleshooting

### Pod Status Issues

```bash
# Check pod status
kubectl get pods -n doc-chat-system

# Describe pod for events
kubectl describe pod -n doc-chat-system -l app=doc-chat

# View logs
kubectl logs -n doc-chat-system -l app=doc-chat
```

### Common Issues

#### 1. ImagePullBackOff
```bash
# Check if image exists locally (for IfNotPresent)
docker images | grep doc-chat

# Or push to registry and update image reference
kubectl set image deployment/doc-chat doc-chat=your-registry/doc-chat:v1.0.0 -n doc-chat-system
```

#### 2. CrashLoopBackOff
```bash
# Check logs for errors
kubectl logs -n doc-chat-system -l app=doc-chat --previous

# Common causes:
# - Ollama not accessible
# - Missing required models
# - Insufficient resources
```

#### 3. Chroma Connection Failed
The application will fallback to simple mode automatically. Check logs:
```bash
kubectl logs -n doc-chat-system -l app=doc-chat | grep -i "chroma\|vector"
```

#### 4. Hybrid RAG Not Working
Check if HybridChatService is initialized:
```bash
kubectl logs -n doc-chat-system -l app=doc-chat | grep -i "hybrid"
# Expected: "[HybridChat] Initialized HybridChatService"
```

If missing, check:
- ConfigMap has `ENABLE_HYBRID_SEARCH: "true"`
- Documents are uploaded and indexed
- ChromaDB connection is working (if using Chroma mode)

#### 5. BM25 Cache Issues
If cache is not being created:
```bash
# Check cache directory permissions
kubectl exec -n doc-chat-system deployment/doc-chat -- ls -la /app/chroma_db/

# Check logs for cache operations
kubectl logs -n doc-chat-system -l app=doc-chat | grep -i "cache"
```

Common causes:
- Chroma PVC not mounted to doc-chat pod
- Disk space exhausted
- Permission issues

#### 6. Reranking Not Working
If reranking is enabled but not working:
```bash
# Check if sentence-transformers is installed
kubectl exec -n doc-chat-system deployment/doc-chat -- pip list | grep sentence

# Check logs for reranker errors
kubectl logs -n doc-chat-system -l app=doc-chat | grep -i "reranker\|rerank"
```

Fix:
```bash
# Install optional dependency (requires rebuild)
# Add to Dockerfile:
# RUN pip install sentence-transformers
```

#### 7. PVC Pending
```bash
# Check storage class
kubectl get storageclass

# Check PVC events
kubectl describe pvc uploads-pvc -n doc-chat-system
```

### Health Checks

```bash
# Port forward to test locally
kubectl port-forward -n doc-chat-system svc/doc-chat 8080:80

# Test health endpoint
curl http://localhost:8080/health

# Expected response:
# {
#   "status": "healthy",
#   "vector_store": "chroma",
#   "ollama_url": "http://10.0.0.55:11434",
#   "documents_count": 0
# }

# Test model list
curl http://localhost:8080/models
```

## Upgrading from Old Version

### Changes in New Architecture

1. **Environment Variables:**
   - Added `VECTOR_STORE_TYPE` (replaces `USE_CHROMA` as primary config)
   - Added `CHROMA_COLLECTION`, `APP_NAME`, `APP_VERSION`
   - Added storage path configs: `UPLOAD_DIR`, `CHROMA_DIR`, `STATIC_DIR`
   - **NEW**: Hybrid RAG configuration variables

2. **File Structure:**
   - New modular structure in `/app/` directory
   - `PYTHONPATH` set to `/app`

3. **Dockerfile Changes:**
   - Now copies `main.py` and `app/` separately
   - Production-grade CMD without `--reload`

### Migration Steps

1. Update ConfigMap with new variables
2. Rebuild Docker image with new Dockerfile
3. Redeploy with updated YAML
4. PVCs will retain existing data

### Migrating to Hybrid RAG

If upgrading from non-Hybrid version:

1. **Update ConfigMap** with Hybrid RAG settings (see Configuration Reference)

2. **Mount Chroma PVC** to doc-chat pod for BM25 cache:
   ```yaml
   volumeMounts:
   - name: chroma-data
     mountPath: /app/chroma_db
   ```

3. **First deployment** will build BM25 index (may take 1-5 minutes depending on document count)

4. **Verify Hybrid RAG** is working:
   ```bash
   kubectl logs -n doc-chat-system -l app=doc-chat | grep -i "hybrid"
   # Should see: "[HybridChat] Initialized HybridChatService"
   # And: "Indexed X nodes in hybrid retriever"
   ```

5. **Check cache** is created:
   ```bash
   kubectl exec -n doc-chat-system deployment/doc-chat -- ls -la /app/chroma_db/bm25_cache/
   ```

### Performance Comparison

| Metric | Old (Dense Only) | Hybrid RAG | Improvement |
|--------|------------------|------------|-------------|
| Recall@10 | Baseline | +15-25% | Better coverage |
| Keyword match | Poor | Excellent | Technical terms |
| Semantic match | Good | Good | Context understanding |
| Index load (cached) | 1x | 10-50x | Faster restarts |

## Scaling Considerations

### Horizontal Scaling

**Current Limitations:**
- Simple mode: Cannot scale horizontally (in-memory state)
- Chroma mode: Can scale backend, but ChromaDB is single-instance

**Recommended Architecture for High Availability:**
```
┌──────────────────────────────────────────┐
│           Load Balancer                  │
└──────────────┬───────────────────────────┘
               │
    ┌──────────┴──────────┐
    ▼                     ▼
┌─────────┐          ┌─────────┐
│doc-chat-1│          │doc-chat-2│
└────┬────┘          └────┬────┘
     │                    │
     └────────┬───────────┘
              ▼
       ┌────────────┐
       │   Chroma   │
       │   (HA)     │
       └────────────┘
```

### Vertical Scaling

Adjust resource limits based on:
- Number of documents
- Document sizes
- Concurrent users
- Model sizes

**Memory Calculation:**
- Base: 1GB
- Per 1000 documents: ~500MB
- ChromaDB: 512MB base + vector storage
- **Hybrid RAG overhead**: +200-500MB
  - BM25 index: ~50% of document text size
  - Cache metadata: ~10MB per 1000 documents
  - Reranker (if enabled): +500MB-1GB model size

## Security Best Practices

### 1. Network Policies
```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: doc-chat-network-policy
  namespace: doc-chat-system
spec:
  podSelector:
    matchLabels:
      app: doc-chat
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: chroma
  - to:
    - ipBlock:
        cidr: 10.0.0.55/32  # Ollama host
```

### 2. Secrets Management
Use Kubernetes Secrets for sensitive data:
```bash
kubectl create secret generic doc-chat-secrets \
  --from-literal=ollama-api-key=your-key \
  -n doc-chat-system
```

### 3. RBAC
Create service accounts with minimal permissions.

## Monitoring

### Prometheus Metrics (Future Enhancement)

Add to deployment for metrics endpoint:
```yaml
annotations:
  prometheus.io/scrape: "true"
  prometheus.io/port: "8000"
  prometheus.io/path: "/metrics"
```

### Log Aggregation

Configure logging format:
```yaml
env:
- name: LOG_LEVEL
  value: "INFO"  # DEBUG, INFO, WARNING, ERROR
- name: LOG_FORMAT
  value: "json"  # or "text"
```

## Cleanup

```bash
# Remove all resources
kubectl delete -f deployment/deployment.yaml

# Remove PVCs (WARNING: Data will be lost!)
kubectl delete pvc uploads-pvc chroma-pvc -n doc-chat-system

# Remove namespace
kubectl delete namespace doc-chat-system
```

---

For more information, see [README.md](../README.md) and [ARCHITECTURE.md](../backend/ARCHITECTURE.md).
