"""Hybrid chat service combining dense and sparse retrieval with RRF fusion."""

import os
import uuid
import time
import logging
from typing import Optional, Dict, Any, List

from llama_index.core.schema import TextNode
from llama_index.core.embeddings import BaseEmbedding

from app.config import get_settings
from app.services.chat_service import ChatService
from app.services.model_service import ModelService
from app.core.retrievers.hybrid_retriever import HybridRetriever
from app.core.retrievers.base import NodeWithScore

logger = logging.getLogger(__name__)


class HybridChatService(ChatService):
    """
    Chat service with Hybrid RAG support.

    This service extends the base ChatService to use HybridRetriever,
    which combines dense (vector) and sparse (BM25) retrieval with
    Reciprocal Rank Fusion (RRF).

    The hybrid approach provides better recall and accuracy by leveraging
    both semantic understanding and exact keyword matching.
    """

    def __init__(self, upload_dir: str = None, model_service: ModelService = None):
        """
        Initialize hybrid chat service.

        Args:
            upload_dir: Directory containing uploaded documents
            model_service: Model service instance
        """
        # Don't call parent __init__ to avoid vector store initialization
        settings = get_settings()
        self.upload_dir = upload_dir or settings.upload_dir
        self.model_service = model_service or ModelService()

        # Conversation management
        self.conversation_history: Dict[str, Any] = {}
        self.conversation_timestamps: Dict[str, float] = {}

        # Hybrid retriever (will be initialized per conversation)
        self._hybrid_retrievers: Dict[str, HybridRetriever] = {}

        logger.info("[HybridChat] Initialized HybridChatService")

    def _load_documents(self) -> List[TextNode]:
        """
        Load documents from upload directory.

        Returns:
            List of text nodes
        """
        from llama_index.core import SimpleDirectoryReader

        if not os.path.exists(self.upload_dir):
            return []

        try:
            documents = SimpleDirectoryReader(self.upload_dir).load_data()
            # Convert documents to nodes
            nodes = []
            for doc in documents:
                node = TextNode(id_=str(uuid.uuid4()), text=doc.text, metadata=doc.metadata)
                nodes.append(node)

            logger.info(f"[HybridChat] Loaded {len(nodes)} document nodes")
            return nodes

        except Exception as e:
            logger.error(f"[HybridChat] Failed to load documents: {e}")
            return []

    def get_or_create_hybrid_retriever(
        self,
        conversation_id: str,
        model_name: Optional[str] = None,
        embedding_model: Optional[str] = None,
    ) -> HybridRetriever:
        """
        Get or create a hybrid retriever for a conversation.

        Args:
            conversation_id: Conversation ID
            model_name: LLM model name
            embedding_model: Embedding model name

        Returns:
            HybridRetriever instance
        """
        if conversation_id in self._hybrid_retrievers:
            # Update timestamp
            self.conversation_timestamps[conversation_id] = time.time()
            return self._hybrid_retrievers[conversation_id]

        logger.info(f"[HybridChat] Creating hybrid retriever for conversation: {conversation_id}")

        # Check if documents exist
        if not os.path.exists(self.upload_dir) or not os.listdir(self.upload_dir):
            raise ValueError("No documents uploaded. Please upload documents first.")

        # Configure models
        settings = get_settings()
        embed_model_name = embedding_model or settings.default_embedding_model

        # Get embedding model
        logger.info(f"[HybridChat] Setting up embedding model: {embed_model_name}")
        try:
            embed_model: BaseEmbedding = self.model_service.get_provider().get_embedding_model(
                embed_model_name
            )
            logger.info("[HybridChat] Embedding model setup complete")
        except Exception as e:
            logger.error(f"[HybridChat] Failed to setup embedding model: {e}")
            raise

        # Create hybrid retriever
        retriever = HybridRetriever(
            embed_model=embed_model,
            dense_top_k=settings.dense_top_k,
            sparse_top_k=settings.sparse_top_k,
            final_top_k=settings.final_top_k,
            rrf_k=settings.rrf_k,
        )

        # Load and index documents
        nodes = self._load_documents()
        if nodes:
            retriever.add_documents(nodes)
            logger.info(f"[HybridChat] Indexed {len(nodes)} nodes in hybrid retriever")
        else:
            raise ValueError("No documents could be loaded for indexing")

        # Store retriever
        self._hybrid_retrievers[conversation_id] = retriever
        self.conversation_timestamps[conversation_id] = time.time()

        return retriever

    def chat(
        self,
        message: str,
        conversation_id: Optional[str] = None,
        model_name: Optional[str] = None,
        embedding_model: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Process a chat message using hybrid retrieval.

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
        logger.info(
            f"[HybridChat] New request - conv_id: {conv_id}, model: {model_name or 'default'}"
        )

        try:
            # Get hybrid retriever
            retriever = self.get_or_create_hybrid_retriever(conv_id, model_name, embedding_model)

            # Retrieve context using hybrid search
            retrieve_start = time.time()
            results: List[NodeWithScore] = retriever.retrieve(message)
            retrieve_time = time.time() - retrieve_start

            logger.info(
                f"[HybridChat] Hybrid retrieval completed in {retrieve_time:.2f}s, "
                f"found {len(results)} results"
            )

            # Build context from retrieved nodes
            context_parts = []
            for idx, result in enumerate(results, 1):
                context_parts.append(f"[{idx}] {result.text}")

            context = "\n\n".join(context_parts)

            # Build prompt with context
            prompt = self._build_prompt(message, context)

            # Generate response using LLM
            llm_start = time.time()
            response = self._generate_response(prompt, model_name)
            llm_time = time.time() - llm_start

            total_time = time.time() - start_time
            logger.info(f"[HybridChat] LLM generation completed in {llm_time:.2f}s")
            logger.info(f"[HybridChat] Total request time: {total_time:.2f}s - conv_id: {conv_id}")

            return {"response": response, "conversation_id": conv_id}

        except Exception as e:
            logger.error(f"[HybridChat] Error: {e}")
            import traceback

            logger.error(f"[HybridChat] Traceback: {traceback.format_exc()}")
            raise

    def _build_prompt(self, query: str, context: str) -> str:
        """
        Build a prompt with retrieved context.

        Args:
            query: User query
            context: Retrieved context

        Returns:
            Formatted prompt
        """
        return f"""Based on the following context, please answer the question.

Context:
{context}

Question: {query}

Please provide a comprehensive answer based only on the context provided above. If the context doesn't contain enough information to answer the question, please say so.

Answer:"""

    def _generate_response(self, prompt: str, model_name: Optional[str] = None) -> str:
        """
        Generate response using LLM.

        Args:
            prompt: Complete prompt with context
            model_name: Optional model name

        Returns:
            Generated response text
        """
        settings = get_settings()
        llm_model = model_name or settings.default_model_name

        # Get LLM
        llm = self.model_service.get_provider().get_llm(llm_model, request_timeout=180.0)

        # Generate response
        response = llm.complete(prompt)

        return str(response)

    def clear_conversation(self, conversation_id: str) -> bool:
        """
        Clear a conversation from history.

        Args:
            conversation_id: Conversation ID to clear

        Returns:
            bool: True if cleared successfully
        """
        super().clear_conversation(conversation_id)

        if conversation_id in self._hybrid_retrievers:
            del self._hybrid_retrievers[conversation_id]
            logger.info(
                f"[HybridChat] Cleared hybrid retriever for conversation: {conversation_id}"
            )

        return True

    def clear_all_conversations(self) -> None:
        """Clear all conversation history."""
        super().clear_all_conversations()
        self._hybrid_retrievers.clear()
        logger.info("[HybridChat] All hybrid retrievers cleared")
