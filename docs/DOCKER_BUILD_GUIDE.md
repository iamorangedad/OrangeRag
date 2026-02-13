# Docker Build Guide for Hybrid RAG

This guide covers building and running the Hybrid RAG Docker container.

## Quick Start

```bash
# Build the image
docker build -t doc-chat:latest .

# Run with default Hybrid RAG configuration
docker run -d \
  --name doc-chat \
  -p 8000:8000 \
  -v $(pwd)/uploads:/app/uploads \
  -v $(pwd)/chroma_db:/app/chroma_db \
  -e ENABLE_HYBRID_SEARCH=true \
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
  doc-chat:latest
```

## Build Options

### 1. Basic Build (Standard)

```bash
docker build -t doc-chat:latest .
```

**Features:**
- Hybrid RAG with dense + sparse retrieval
- BM25 caching enabled
- No reranking (faster, smaller image)

**Image Size:** ~1.5-2GB

### 2. Build with Reranking Support (Optional)

For best retrieval quality, add Cross-Encoder reranking:

```bash
# Create a requirements-rerank.txt with additional dependencies
cat > backend/requirements-rerank.txt << EOF
-r requirements.txt
sentence-transformers>=2.2.0
EOF

# Build with modified Dockerfile
docker build -f Dockerfile.rerank -t doc-chat:rerank .
```

Or modify requirements.txt temporarily:
```bash
# Add sentence-transformers to requirements.txt
echo "sentence-transformers>=2.2.0" >> backend/requirements.txt

# Build
docker build -t doc-chat:full .

# Restore original
git checkout backend/requirements.txt
```

**Features:**
- All Hybrid RAG features
- Cross-Encoder reranking support
- Better result quality (adds 100-500ms latency)

**Image Size:** ~2.5-3GB

## Running the Container

### Basic Usage

```bash
docker run -d \
  --name doc-chat \
  -p 8000:8000 \
  -v $(pwd)/uploads:/app/uploads \
  -v $(pwd)/chroma_db:/app/chroma_db \
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
  doc-chat:latest
```

### With Hybrid RAG Tuning

```bash
docker run -d \
  --name doc-chat \
  -p 8000:8000 \
  -v $(pwd)/uploads:/app/uploads \
  -v $(pwd)/chroma_db:/app/chroma_db \
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
  -e ENABLE_HYBRID_SEARCH=true \
  -e DENSE_TOP_K=10 \
  -e SPARSE_TOP_K=10 \
  -e FINAL_TOP_K=5 \
  -e FUSION_MODE=rrf \
  -e BM25_CACHE_ENABLED=true \
  doc-chat:latest
```

### High Performance Mode

```bash
docker run -d \
  --name doc-chat-hybrid \
  -p 8000:8000 \
  -v $(pwd)/uploads:/app/uploads \
  -v $(pwd)/chroma_db:/app/chroma_db \
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
  -e ENABLE_HYBRID_SEARCH=true \
  -e DENSE_TOP_K=5 \
  -e SPARSE_TOP_K=5 \
  -e FINAL_TOP_K=3 \
  -e BM25_CACHE_ENABLED=true \
  --memory="4g" \
  --cpus="2.0" \
  doc-chat:latest
```

### Maximum Quality Mode

```bash
docker run -d \
  --name doc-chat-quality \
  -p 8000:8000 \
  -v $(pwd)/uploads:/app/uploads \
  -v $(pwd)/chroma_db:/app/chroma_db \
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
  -e ENABLE_HYBRID_SEARCH=true \
  -e DENSE_TOP_K=50 \
  -e SPARSE_TOP_K=50 \
  -e FINAL_TOP_K=20 \
  -e ENABLE_RERANK=true \
  -e ENABLE_QUERY_EXPANSION=true \
  -e QUERY_EXPANSION_TYPE=multi \
  --memory="8g" \
  --cpus="4.0" \
  doc-chat:full
```

## Docker Compose

### docker-compose.yml

```yaml
version: '3.8'

services:
  doc-chat:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OLLAMA_BASE_URL=http://host.docker.internal:11434
      - ENABLE_HYBRID_SEARCH=true
      - DENSE_TOP_K=10
      - SPARSE_TOP_K=10
      - FINAL_TOP_K=5
      - BM25_CACHE_ENABLED=true
    volumes:
      - ./uploads:/app/uploads
      - ./chroma_db:/app/chroma_db
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s
    restart: unless-stopped
```

### Run with Compose

```bash
# Start
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down

# Stop and remove volumes (WARNING: Data loss)
docker-compose down -v
```

## Volume Management

### Important Directories

| Host Path | Container Path | Purpose |
|-----------|----------------|---------|
| `./uploads` | `/app/uploads` | Document storage |
| `./chroma_db` | `/app/chroma_db` | Vector DB + BM25 cache |

### Backup Strategy

```bash
# Backup data
tar -czf backup-$(date +%Y%m%d).tar.gz uploads/ chroma_db/

# Restore data
tar -xzf backup-20240207.tar.gz
```

### Cache Management

```bash
# View BM25 cache size
docker exec doc-chat du -sh /app/chroma_db/bm25_cache

# Clear cache (if needed)
docker exec doc-chat rm -rf /app/chroma_db/bm25_cache/*

# Recreate cache on next startup
```

## Environment Variables

### Required

| Variable | Description | Example |
|----------|-------------|---------|
| `OLLAMA_BASE_URL` | Ollama service URL | `http://host.docker.internal:11434` |

### Hybrid RAG Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `ENABLE_HYBRID_SEARCH` | Enable Hybrid RAG | `true` |
| `DENSE_TOP_K` | Dense retrieval results | `10` |
| `SPARSE_TOP_K` | Sparse retrieval results | `10` |
| `FINAL_TOP_K` | Final results after fusion | `5` |
| `FUSION_MODE` | Fusion algorithm | `rrf` |
| `BM25_CACHE_ENABLED` | Enable BM25 caching | `true` |

### Optional Features

| Variable | Description | Default |
|----------|-------------|---------|
| `ENABLE_RERANK` | Enable Cross-Encoder reranking | `false` |
| `RERANK_DEVICE` | Reranker device (`cuda`/`cpu`/`mps`) | `cuda` |
| `ENABLE_QUERY_EXPANSION` | Enable query expansion | `false` |

## Health Checks

The container includes a built-in health check:

```bash
# Check container health
docker ps

# Expected: (healthy) after ~60s

# Manual health check
curl http://localhost:8000/health
```

**Note:** First startup may take longer (30-120s) due to:
1. Loading embedding models
2. Building BM25 index from documents
3. Creating initial cache

Subsequent startups are faster due to BM25 cache.

## Monitoring

### Container Stats

```bash
# Resource usage
docker stats doc-chat

# Logs
docker logs -f doc-chat

# Filter Hybrid RAG logs
docker logs doc-chat | grep -i "hybrid\|bm25\|cache"
```

### Performance Metrics

Watch for these log patterns:

```
[HybridChat] Indexed X nodes in hybrid retriever
[HybridChat] Hybrid retrieval completed in Xs
[HybridChat] Total request time: Xs
```

## Troubleshooting

### Container Won't Start

```bash
# Check logs
docker logs doc-chat

# Common issues:
# 1. Ollama not accessible
# 2. Port 8000 already in use
# 3. Permission issues with volumes
```

### Permission Denied

```bash
# Fix volume permissions
sudo chown -R $USER:$USER uploads/ chroma_db/

# Or run with user ID
docker run --user $(id -u):$(id -g) ...
```

### Memory Issues

```bash
# If OOM killed, increase memory
docker run --memory="8g" ...

# Or reduce retrieval depth
-e DENSE_TOP_K=5 -e SPARSE_TOP_K=5
```

### BM25 Cache Not Created

```bash
# Check directory permissions
docker exec doc-chat ls -la /app/chroma_db/

# Check logs
docker logs doc-chat | grep -i cache
```

## Multi-Stage Build (Advanced)

For smaller production images:

```dockerfile
# Build stage
FROM python:3.11-slim as builder

WORKDIR /app
RUN apt-get update && apt-get install -y gcc

COPY backend/requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.11-slim

WORKDIR /app

# Copy only necessary files
COPY --from=builder /root/.local /root/.local
COPY backend/main.py .
COPY backend/app/ ./app/
COPY frontend/index.html ./static/

RUN mkdir -p uploads chroma_db static

ENV PATH=/root/.local/bin:$PATH
ENV PYTHONPATH=/app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build:
```bash
docker build -f Dockerfile.multistage -t doc-chat:slim .
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Build and Push

on:
  push:
    branches: [ main ]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Build image
      run: docker build -t doc-chat:${{ github.sha }} .
    
    - name: Test image
      run: |
        docker run -d --name test -p 8000:8000 doc-chat:${{ github.sha }}
        sleep 30
        curl -f http://localhost:8000/health || exit 1
        docker stop test
    
    - name: Push to registry
      run: |
        echo ${{ secrets.DOCKER_PASSWORD }} | docker login -u ${{ secrets.DOCKER_USERNAME }} --password-stdin
        docker tag doc-chat:${{ github.sha }} doc-chat:latest
        docker push doc-chat:latest
```

## Best Practices

1. **Always use volumes** for persistent data
2. **Set memory limits** to prevent OOM kills
3. **Enable health checks** in orchestration
4. **Use BM25 cache** for faster restarts
5. **Monitor logs** for Hybrid RAG performance
6. **Tag images** with version numbers

## Security

```bash
# Run as non-root user (optional)
docker run --user 1000:1000 ...

# Read-only root filesystem
docker run --read-only \
  --tmpfs /tmp:noexec,nosuid,size=100m \
  ...

# Drop capabilities
docker run --cap-drop=ALL ...
```
