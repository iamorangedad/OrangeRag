"""Reranker modules for improving retrieval results."""

from app.core.reranker.base import BaseReranker

__all__ = ["BaseReranker"]

# Optional: CrossEncoderReranker (requires sentence-transformers)
try:
    from app.core.reranker.cross_encoder import CrossEncoderReranker

    __all__.append("CrossEncoderReranker")
except ImportError:
    pass
