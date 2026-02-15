"""Document Chat API - Main Entry Point.

This is a refactored version with clean architecture:
- Layered architecture with clear separation of concerns
- Abstract vector store layer supporting multiple backends
- Abstract LLM layer supporting multiple providers
- Service layer for business logic
- API layer for HTTP routes
"""
import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import get_settings
from app.api import api_router
from app.services import ChatService
from app.core.logging_config import setup_logging, get_logger

# Configure structured logging using settings
settings = get_settings()
setup_logging(
    level=settings.log_level,
    log_format=settings.log_format,
    output=settings.log_output,
    file_path=settings.log_file_path
)
logger = get_logger(__name__)


def setup_directories():
    """Create necessary directories."""
    settings = get_settings()
    os.makedirs(settings.upload_dir, exist_ok=True)
    os.makedirs(settings.chroma_dir, exist_ok=True)
    os.makedirs(settings.static_dir, exist_ok=True)


def startup_checks():
    """Perform startup checks and logging."""
    settings = get_settings()
    
    logger.info("=" * 50)
    logger.info("[Startup] Doc-Chat API starting...")
    logger.info(f"[Startup] Ollama base URL: {settings.ollama_base_url}")
    logger.info(f"[Startup] Default model: {settings.default_model_name}")
    logger.info(f"[Startup] Embedding model: {settings.default_embedding_model}")
    logger.info(f"[Startup] Vector store: {settings.effective_vector_store_type}")
    
    if settings.effective_vector_store_type == "chroma":
        logger.info(f"[Startup] Chroma host: {settings.chroma_host}:{settings.chroma_port}")
    
    logger.info(f"[Startup] Upload directory: {settings.upload_dir}")
    logger.info(f"[Startup] Static directory: {settings.static_dir}")
    logger.info("=" * 50)
    
    # Test Ollama connection
    try:
        from app.services import ModelService
        model_service = ModelService()
        models = model_service.list_models()
        logger.info(f"[Startup] Successfully connected to Ollama, found {len(models)} models")
        
        # Check default models
        if not model_service.check_model_exists(settings.default_model_name):
            logger.warning(
                f"[Startup] Default model '{settings.default_model_name}' NOT FOUND! "
                f"Run: ollama pull {settings.default_model_name}"
            )
        else:
            logger.info(f"[Startup] Default model '{settings.default_model_name}' is ready")
        
        if not model_service.check_model_exists(settings.default_embedding_model):
            logger.warning(
                f"[Startup] Embedding model '{settings.default_embedding_model}' NOT FOUND! "
                f"Run: ollama pull {settings.default_embedding_model}"
            )
        else:
            logger.info(f"[Startup] Embedding model '{settings.default_embedding_model}' is ready")
            
    except Exception as e:
        logger.warning(f"[Startup] Could not connect to Ollama: {e}")


def start_cleanup_task():
    """Start background cleanup task for expired conversations."""
    import threading
    import time
    
    def cleanup_loop():
        chat_service = ChatService()
        while True:
            time.sleep(3600)  # Check every hour
            chat_service.cleanup_expired_conversations(max_age_hours=24)
    
    thread = threading.Thread(target=cleanup_loop, daemon=True)
    thread.start()
    logger.info("[Startup] Started conversation cleanup thread")


def start_indexing_service():
    """Initialize the async indexing service."""
    from app.services import get_indexing_service
    indexing_service = get_indexing_service()
    logger.info("[Startup] Async indexing service initialized")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    setup_directories()
    startup_checks()
    start_cleanup_task()
    start_indexing_service()
    yield
    # Shutdown
    logger.info("[Shutdown] Doc-Chat API shutting down...")
    from app.services import get_indexing_service
    indexing_service = get_indexing_service()
    indexing_service.shutdown()


def create_application() -> FastAPI:
    """Create and configure FastAPI application."""
    settings = get_settings()
    
    app = FastAPI(
        title=settings.app_name,
        description="Document Chat System with LlamaIndex - Refactored Version",
        version=settings.app_version,
        lifespan=lifespan
    )
    
    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization"],
    )
    
    # Mount static files
    app.mount("/static", StaticFiles(directory=settings.static_dir), name="static")
    
    # Include all API routes
    app.include_router(api_router)
    
    return app


# Create application instance
app = create_application()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
