"""Ollama LLM provider implementation."""

import logging
from typing import Optional
import httpx
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.llms import LLM
from llama_index.core.embeddings import BaseEmbedding

from app.core.llm.base import LLMProvider, register_llm
from app.utils.helpers import format_size, get_param_size, get_quantization, requires_gpu

logger = logging.getLogger(__name__)


class OllamaProvider(LLMProvider):
    """Ollama LLM provider."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        default_model: str = "qwen3:4b",
        default_embedding_model: str = "nomic-embed-text",
    ):
        """
        Initialize Ollama provider.

        Args:
            base_url: Ollama API base URL
            default_model: Default LLM model name
            default_embedding_model: Default embedding model name
        """
        self.base_url = base_url.rstrip("/")
        self.default_model = default_model
        self.default_embedding_model = default_embedding_model

    def get_llm(self, model_name: Optional[str] = None, request_timeout: float = 120.0) -> LLM:
        """Get Ollama LLM instance with extended timeout.

        Args:
            model_name: Model name to use
            request_timeout: HTTP timeout in seconds (default 120s for long queries)
        """
        model = model_name or self.default_model
        return Ollama(model=model, base_url=self.base_url, request_timeout=request_timeout)

    def get_embedding_model(
        self, model_name: Optional[str] = None, batch_size: int = 10
    ) -> BaseEmbedding:
        """Get Ollama embedding model instance with batching support."""
        model = model_name or self.default_embedding_model

        # Check if model exists
        if not self.check_model_exists(model):
            error_msg = f'Model "{model}" not found in Ollama. Please run: ollama pull {model}'
            logger.error(f"[Embedding] {error_msg}")
            raise ValueError(error_msg)

        # Use batched embedding to handle large documents
        from app.core.llm.batched_embedding import BatchedOllamaEmbedding

        # Use smaller batch size (5) and longer delays to prevent Ollama overload
        return BatchedOllamaEmbedding(
            model_name=model, base_url=self.base_url, batch_size=5, max_retries=5, retry_delay=2.0
        )

    def check_model_exists(self, model_name: str) -> bool:
        """Check if a model exists in Ollama."""
        try:
            with httpx.Client(timeout=5.0) as client:
                response = client.get(f"{self.base_url}/api/tags")
                if response.status_code == 200:
                    data = response.json()
                    models = [m.get("name", "") for m in data.get("models", [])]
                    return any(model_name in m or m in model_name for m in models)
        except Exception as e:
            logger.warning(f"[ModelCheck] Failed to check model {model_name}: {e}")
        return False

    def list_models(self) -> list[dict]:
        """Fetch available models from Ollama."""
        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.get(f"{self.base_url}/api/tags")
                if response.status_code == 200:
                    data = response.json()
                    models = []
                    for model in data.get("models", []):
                        name = model.get("name", "")
                        size = model.get("size", 0)
                        model_info = {
                            "name": name,
                            "size": format_size(size),
                            "parameter_size": get_param_size(name),
                            "quantization": get_quantization(name),
                            "requires_gpu": requires_gpu(name),
                        }
                        models.append(model_info)
                    return models
        except Exception as e:
            logger.error(f"Failed to fetch models from Ollama: {e}")

        # Return default models if Ollama is not available
        return self._get_default_models()

    def _get_default_models(self) -> list[dict]:
        """Return default model list if Ollama is not available."""
        return [
            {
                "name": self.default_model,
                "size": "~2GB",
                "parameter_size": "3B",
                "quantization": "Default",
                "requires_gpu": False,
            },
            {
                "name": self.default_embedding_model,
                "size": "~150MB",
                "parameter_size": "137M",
                "quantization": "Default",
                "requires_gpu": False,
            },
            {
                "name": "llama3.1:8b",
                "size": "~4.7GB",
                "parameter_size": "8B",
                "quantization": "Default",
                "requires_gpu": False,
            },
            {
                "name": "mistral",
                "size": "~4.1GB",
                "parameter_size": "7B",
                "quantization": "Default",
                "requires_gpu": False,
            },
            {
                "name": "qwen2:7b",
                "size": "~4.4GB",
                "parameter_size": "7B",
                "quantization": "Default",
                "requires_gpu": False,
            },
        ]


# Register the provider
register_llm("ollama", OllamaProvider)
