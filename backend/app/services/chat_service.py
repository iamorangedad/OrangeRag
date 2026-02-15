"""Chat service for conversation and indexing."""
import os
import uuid
import time
import logging
import traceback
from typing import Optional, Dict, Any

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings

from app.config import get_settings
from app.core.vector_store import create_vector_store, VectorStoreProvider
from app.services.model_service import ModelService
from app.core.logging_config import get_logger, log_performance, log_error

logger = get_logger(__name__)


class ChatService:
    """Service for chat operations and document indexing."""
    
    def __init__(
        self,
        upload_dir: str = None,
        vector_store_provider: VectorStoreProvider = None,
        model_service: ModelService = None
    ):
        """
        Initialize chat service.
        
        Args:
            upload_dir: Directory containing uploaded documents
            vector_store_provider: Vector store provider instance
            model_service: Model service instance
        """
        settings = get_settings()
        self.upload_dir = upload_dir or settings.upload_dir
        self.model_service = model_service or ModelService()
        
        # Initialize vector store
        if vector_store_provider is None:
            store_type = settings.effective_vector_store_type
            if store_type == "chroma":
                vector_store_provider = create_vector_store(
                    "chroma",
                    host=settings.chroma_host,
                    port=settings.chroma_port,
                    collection=settings.chroma_collection
                )
            else:
                vector_store_provider = create_vector_store(
                    "simple",
                    persist_dir=settings.chroma_dir
                )
        
        self.vector_store_provider = vector_store_provider
        
        # Conversation management
        self.conversation_history: Dict[str, Any] = {}
        self.conversation_timestamps: Dict[str, float] = {}
    
    def create_conversation(self) -> str:
        """Create a new conversation and return its ID."""
        conv_id = str(uuid.uuid4())
        self.conversation_timestamps[conv_id] = time.time()
        return conv_id
    
    def get_or_create_query_engine(
        self,
        conversation_id: str,
        model_name: Optional[str] = None,
        embedding_model: Optional[str] = None
    ) -> Any:
        """
        Get or create a query engine for a conversation.
        
        Args:
            conversation_id: Conversation ID
            model_name: LLM model name
            embedding_model: Embedding model name
            
        Returns:
            Query engine instance
        """
        if conversation_id in self.conversation_history:
            # Update timestamp
            self.conversation_timestamps[conversation_id] = time.time()
            return self.conversation_history[conversation_id]
        
        # Create new index
        logger.info(f"[Chat] Creating new index for conversation: {conversation_id}")
        
        # Check if documents exist
        if not os.path.exists(self.upload_dir) or not os.listdir(self.upload_dir):
            raise ValueError("No documents uploaded. Please upload documents first.")
        
        # Configure models
        settings = get_settings()
        llm_model = model_name or settings.default_model_name
        embed_model = embedding_model or settings.default_embedding_model
        
        # Set up LLM with extended timeout for long queries
        logger.info(f"[Chat] Setting up LLM: {llm_model}")
        try:
            # Use 180s timeout for long documents and complex queries
            Settings.llm = self.model_service.get_provider().get_llm(llm_model, request_timeout=180.0)
            logger.info("[Chat] LLM setup complete with 180s timeout")
        except Exception as e:
            logger.error(f"[Chat] Failed to setup LLM: {e}")
            raise
        
        # Get embedding model (don't set to Settings - pass directly to index)
        logger.info(f"[Chat] Setting up embedding model: {embed_model}")
        try:
            embed_model_instance = self.model_service.get_provider().get_embedding_model(embed_model)
            logger.info("[Chat] Embedding model setup complete")
        except Exception as e:
            logger.error(f"[Chat] Failed to setup embedding model: {e}")
            import traceback
            logger.error(f"[Chat] Embedding traceback: {traceback.format_exc()}")
            raise
        
        # Load documents
        documents = SimpleDirectoryReader(self.upload_dir).load_data()
        logger.info(f"[Chat] Loaded {len(documents)} documents")
        
        # Create index with vector store and explicit embed model
        vector_store = self.vector_store_provider.get_vector_store()
        index = VectorStoreIndex.from_documents(
            documents,
            vector_store=vector_store,
            embed_model=embed_model_instance,
            show_progress=True
        )
        
        # Create query engine
        query_engine = index.as_query_engine()
        self.conversation_history[conversation_id] = query_engine
        self.conversation_timestamps[conversation_id] = time.time()
        
        return query_engine
    
    def chat(
        self,
        message: str,
        conversation_id: Optional[str] = None,
        model_name: Optional[str] = None,
        embedding_model: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Process a chat message.
        
        Args:
            message: User message
            conversation_id: Optional conversation ID
            model_name: Optional LLM model name
            embedding_model: Optional embedding model name
            
        Returns:
            Dict with response and conversation_id
        """
        start_time = time.time()
        
        # Create or use conversation ID
        conv_id = conversation_id or self.create_conversation()
        logger.info(f"[Chat] New request - conv_id: {conv_id}, model: {model_name or 'default'}")
        
        try:
            # Get query engine
            query_engine = self.get_or_create_query_engine(
                conv_id,
                model_name,
                embedding_model
            )
            
            # Execute query
            query_start = time.time()
            response = query_engine.query(message)
            query_time = time.time() - query_start
            
            log_performance(
                logger, "Chat", "query_execution",
                query_time * 1000,  # Convert to ms
                conversation_id=conv_id
            )
            
            total_time = time.time() - start_time
            log_performance(
                logger, "Chat", "total_request",
                total_time * 1000,  # Convert to ms
                conversation_id=conv_id
            )
            
            return {
                "response": str(response),
                "conversation_id": conv_id
            }
            
        except Exception as e:
            log_error(logger, "Chat", e, conversation_id=conv_id)
            raise
    
    def clear_conversation(self, conversation_id: str) -> bool:
        """
        Clear a conversation from history.
        
        Args:
            conversation_id: Conversation ID to clear
            
        Returns:
            bool: True if cleared successfully
        """
        if conversation_id in self.conversation_history:
            del self.conversation_history[conversation_id]
        if conversation_id in self.conversation_timestamps:
            del self.conversation_timestamps[conversation_id]
        return True
    
    def clear_all_conversations(self) -> None:
        """Clear all conversation history."""
        self.conversation_history.clear()
        self.conversation_timestamps.clear()
        logger.info("[Chat] All conversations cleared")
    
    def cleanup_expired_conversations(self, max_age_hours: int = 24) -> int:
        """
        Clean up conversations older than specified hours.
        
        Args:
            max_age_hours: Maximum age in hours
            
        Returns:
            int: Number of conversations cleaned up
        """
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        expired = []
        
        for conv_id, last_access in list(self.conversation_timestamps.items()):
            if current_time - last_access > max_age_seconds:
                expired.append(conv_id)
        
        for conv_id in expired:
            self.clear_conversation(conv_id)
        
        if expired:
            logger.info(f"[Chat] Cleaned up {len(expired)} expired conversations")
        
        return len(expired)
