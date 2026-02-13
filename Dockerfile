FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
# gcc: Required for building Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Note: For optional Cross-Encoder reranking support, add to requirements.txt:
# sentence-transformers>=2.2.0
# This will increase image size by ~500MB-1GB

# Copy and install Python dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code with new modular structure
COPY backend/main.py .
COPY backend/app/ ./app/

# Copy frontend static files
COPY frontend/index.html ./static/

# Create necessary directories
# uploads: Document storage
# chroma_db: Vector database and BM25 cache storage
# static: Frontend files
# Note: BM25 cache directory will be created at runtime under chroma_db
RUN mkdir -p uploads chroma_db static && \
    chmod 755 uploads chroma_db static

# Set environment variables
ENV PYTHONPATH=/app
ENV UPLOAD_DIR=/app/uploads
ENV CHROMA_DIR=/app/chroma_db
ENV STATIC_DIR=/app/static

EXPOSE 8000

# Health check for container orchestration
# Hybrid RAG startup may take longer initially due to BM25 index building
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Use production-grade server without --reload
# Note: For Hybrid RAG with high query volume, consider increasing workers
# based on available CPU cores: --workers $(nproc)
# However, be mindful of memory usage (each worker loads BM25 index)
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
