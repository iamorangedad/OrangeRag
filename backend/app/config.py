"""Application configuration management."""

import os
from functools import lru_cache
from typing import List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Ollama Configuration
    ollama_base_url: str = Field(default="http://10.0.0.55:11434", alias="OLLAMA_BASE_URL")
    default_model_name: str = Field(default="qwen3:4b", alias="MODEL_NAME")
    default_embedding_model: str = Field(default="nomic-embed-text", alias="EMBEDDING_MODEL")

    # Vector Store Configuration
    vector_store_type: str = Field(default="simple", alias="VECTOR_STORE_TYPE")

    # ChromaDB Configuration
    use_chroma: bool = Field(default=False, alias="USE_CHROMA")
    chroma_host: str = Field(default="chroma", alias="CHROMA_HOST")
    chroma_port: int = Field(default=8000, alias="CHROMA_PORT")
    chroma_collection: str = Field(default="documents", alias="CHROMA_COLLECTION")

    # File Storage Configuration
    upload_dir: str = Field(default="uploads", alias="UPLOAD_DIR")
    chroma_dir: str = Field(default="chroma_db", alias="CHROMA_DIR")
    static_dir: str = Field(default="static", alias="STATIC_DIR")

    # CORS Configuration
    allowed_origins: str = Field(default="", alias="ALLOWED_ORIGINS")

    # Application Configuration
    app_name: str = Field(default="Document Chat API", alias="APP_NAME")
    app_version: str = Field(default="1.0.0", alias="APP_VERSION")
    debug: bool = Field(default=False, alias="DEBUG")

    # Hybrid RAG Configuration
    enable_hybrid_search: bool = Field(default=True, alias="ENABLE_HYBRID_SEARCH")
    dense_weight: float = Field(default=0.5, alias="DENSE_WEIGHT")
    sparse_weight: float = Field(default=0.5, alias="SPARSE_WEIGHT")
    rrf_k: float = Field(default=60.0, alias="RRF_K")

    # Retrieval Configuration
    dense_top_k: int = Field(default=10, alias="DENSE_TOP_K")
    sparse_top_k: int = Field(default=10, alias="SPARSE_TOP_K")
    final_top_k: int = Field(default=5, alias="FINAL_TOP_K")

    # Cache Configuration
    bm25_cache_enabled: bool = Field(default=True, alias="BM25_CACHE_ENABLED")
    bm25_cache_dir: Optional[str] = Field(default=None, alias="BM25_CACHE_DIR")

    # Fusion Configuration
    fusion_mode: str = Field(default="rrf", alias="FUSION_MODE")  # "rrf" or "weighted"

    # Reranking Configuration (Phase 3)
    enable_rerank: bool = Field(default=False, alias="ENABLE_RERANK")
    rerank_model: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2", alias="RERANK_MODEL")

    # Query Expansion Configuration (Phase 3)
    enable_query_expansion: bool = Field(default=False, alias="ENABLE_QUERY_EXPANSION")
    query_expansion_type: str = Field(
        default="synonym", alias="QUERY_EXPANSION_TYPE"
    )  # "synonym", "keyword", "hyde", "multi"
    query_expansion_max: int = Field(default=3, alias="QUERY_EXPANSION_MAX")

    @property
    def cors_origins(self) -> List[str]:
        """Parse CORS origins from environment variable."""
        if not self.allowed_origins:
            return ["*"]
        return [origin.strip() for origin in self.allowed_origins.split(",")]

    @property
    def effective_vector_store_type(self) -> str:
        """Get effective vector store type (handles legacy USE_CHROMA flag)."""
        if self.use_chroma and self.vector_store_type == "simple":
            return "chroma"
        return self.vector_store_type

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
