"""Hybrid chat service with three-layer retrieval architecture and citation support.

This service implements the three-layer architecture:
1. Metadata Matching (BM25 on metadata) - validates document/page references
2. Citation Retrieval (BM25 on content) - retrieves exact text snippets for citations
3. Hybrid RAG (Dense + Sparse) - generates answers with semantic understanding

Key Features:
- Strict mode: When metadata doesn't match, don't force an answer
- Citation support: Provides exact source references [1], [2], etc.
- Metadata validation: Verifies if user queries reference valid documents/pages
"""

import os
import uuid
import time
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field

from llama_index.core.schema import TextNode
from llama_index.core.embeddings import BaseEmbedding

from app.config import get_settings
from app.services.chat_service import ChatService
from app.services.model_service import ModelService
from app.core.retrievers.hybrid_retriever import HybridRetriever
from app.core.retrievers.base import NodeWithScore
from app.core.citation import (
    CitationRetriever,
    CitationRetrieverManager,
    CitationCandidate,
    create_citation_filter_from_match,
)
from app.core.metadata import MetadataMatchResult
from app.core.prompt import StrategicPromptBuilder, PromptContext
from app.core.logging_config import get_logger, log_performance, log_error

logger = get_logger(__name__)


@dataclass
class ChatResponse:
    """Enhanced chat response with citation support."""

    response: str
    conversation_id: str
    metadata_match: Dict[str, Any] = field(default_factory=dict)
    citations: List[Dict[str, Any]] = field(default_factory=list)
    retrieved_count: int = 0


class HybridChatService(ChatService):
    """
    Chat service with three-layer retrieval architecture.

    Architecture:
        Query → MetadataMatcher → [MATCHED/NONE]
                  ↓
        MATCHED: CitationRetriever (filtered)
        NONE: CitationRetriever (global) or skip
                  ↓
        Hybrid RAG (Dense + Sparse)
                  ↓
        Strategic Prompt Builder
                  ↓
        LLM Response with Citations

    Features:
    - Metadata validation using BM25 on document metadata
    - Independent citation retrieval using BM25 on content
    - Hybrid RAG for answer generation
    - Strict mode: Don't force answers when metadata doesn't match
    """

    def __init__(self, upload_dir: str = None, model_service: ModelService = None):
        """
        Initialize hybrid chat service with citation support.

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

        # Three-layer retrieval components (per conversation)
        self._hybrid_retrievers: Dict[str, HybridRetriever] = {}
        self._citation_retrievers = CitationRetrieverManager(
            cache_dir=settings.bm25_cache_dir or os.path.join(settings.chroma_dir, "bm25_cache")
        )

        # Query expander (initialized based on config)
        self._query_expander = None
        self._init_query_expander()

        # Reranker (global instance shared across all conversations)
        self._reranker = None
        self._init_reranker()

        # Prompt builder
        self._prompt_builder = StrategicPromptBuilder(
            strict_mode=True,
            max_citation_length=300,
            max_hybrid_context_length=2000,
        )

        logger.info("[HybridChat] Initialized HybridChatService with citation support")

    def _init_query_expander(self) -> None:
        """Initialize query expander based on configuration."""
        settings = get_settings()

        if not settings.enable_query_expansion:
            return

        try:
            from app.core.query.expansion import (
                SynonymExpander,
                KeywordExpander,
                HyDEExpander,
                MultiExpander,
            )

            expansion_type = settings.query_expansion_type.lower()

            if expansion_type == "synonym":
                self._query_expander = SynonymExpander(max_expansions=settings.query_expansion_max)
            elif expansion_type == "keyword":
                self._query_expander = KeywordExpander()
            elif expansion_type == "hyde":
                # HyDE requires LLM, will be initialized lazily
                self._query_expander = HyDEExpander(llm=None)
            elif expansion_type == "multi":
                self._query_expander = MultiExpander()
            else:
                logger.warning(f"[HybridChat] Unknown query expansion type: {expansion_type}")
                return

            logger.info(f"[HybridChat] Initialized {expansion_type} query expander")

        except Exception as e:
            logger.warning(f"[HybridChat] Failed to initialize query expander: {e}")

    def _init_reranker(self) -> None:
        """Initialize reranker based on configuration (global instance)."""
        settings = get_settings()

        if not settings.enable_rerank:
            return

        try:
            from app.core.reranker import CrossEncoderReranker

            self._reranker = CrossEncoderReranker(
                model_name=settings.rerank_model,
                device=settings.rerank_device if settings.rerank_device else None,
            )
            logger.info(
                f"[HybridChat] Initialized CrossEncoderReranker with model: {settings.rerank_model}. "
                "Model will be loaded on first use (lazy loading)."
            )

        except ImportError as e:
            logger.warning(
                f"[HybridChat] Failed to import CrossEncoderReranker: {e}. "
                "Please install sentence-transformers to enable reranking."
            )
        except Exception as e:
            logger.warning(f"[HybridChat] Failed to initialize reranker: {e}")

    def _load_documents(self) -> List[TextNode]:
        """
        Load documents from upload directory.

        Returns:
            List of text nodes
        """
        from app.core.document_processing import UniversalDocumentLoader

        if not os.path.exists(self.upload_dir):
            return []

        try:
            loader = UniversalDocumentLoader(extract_pdf_metadata=True)
            documents = []

            # Load all files in upload directory
            for filename in os.listdir(self.upload_dir):
                file_path = os.path.join(self.upload_dir, filename)
                if os.path.isfile(file_path):
                    try:
                        docs = loader.load_data(file_path)
                        documents.extend(docs)
                        logger.info(f"[HybridChat] Loaded {len(docs)} pages from {filename}")
                    except Exception as e:
                        logger.warning(f"[HybridChat] Failed to load {filename}: {e}")

            # Convert to nodes
            from llama_index.core.node_parser import SentenceSplitter

            node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)
            nodes = node_parser.get_nodes_from_documents(documents)

            # Enhance metadata
            for i, node in enumerate(nodes):
                node.metadata["chunk_index"] = i
                node.metadata["total_chunks"] = len(nodes)

            logger.info(
                f"[HybridChat] Loaded {len(nodes)} document nodes from {len(documents)} pages"
            )
            return nodes

        except Exception as e:
            logger.error(f"[HybridChat] Failed to load documents: {e}")
            import traceback

            logger.error(f"[HybridChat] Traceback: {traceback.format_exc()}")
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

        # Create hybrid retriever with cache support
        retriever = HybridRetriever(
            embed_model=embed_model,
            dense_top_k=settings.dense_top_k,
            sparse_top_k=settings.sparse_top_k,
            final_top_k=settings.final_top_k,
            rrf_k=settings.rrf_k,
            cache_enabled=settings.bm25_cache_enabled,
            cache_dir=settings.bm25_cache_dir or os.path.join(settings.chroma_dir, "bm25_cache"),
        )

        # Configure fusion weights and mode
        retriever.set_fusion_weights(
            dense_weight=settings.dense_weight,
            sparse_weight=settings.sparse_weight,
            mode=settings.fusion_mode,
        )

        # Set global reranker if enabled (shared across all conversations)
        if self._reranker:
            retriever.set_reranker(self._reranker)
            logger.info("[HybridChat] Set global reranker for hybrid retriever")

        # Load and index documents
        nodes = self._load_documents()
        if nodes:
            retriever.add_documents(nodes)
            logger.info(f"[HybridChat] Indexed {len(nodes)} nodes in hybrid retriever")

            # Also build citation index
            citation_retriever = self._citation_retrievers.get_or_create_retriever(conversation_id)
            citation_retriever.build_index(nodes)
            logger.info(f"[HybridChat] Built citation index for conversation: {conversation_id}")
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
    ) -> ChatResponse:
        """
        Process a chat message using three-layer retrieval architecture.

        Three-layer process:
        1. Metadata matching (BM25 on metadata) - validates document/page references
        2. Citation retrieval (BM25 on content) - retrieves exact text snippets
        3. Hybrid RAG (Dense + Sparse) - generates answers

        Args:
            message: User message
            conversation_id: Optional conversation ID
            model_name: Optional LLM model name
            embedding_model: Optional embedding model name

        Returns:
            ChatResponse with response, citations, and metadata match info
        """
        start_time = time.time()

        # Create or use conversation ID
        conv_id = conversation_id or self.create_conversation()
        logger.info(
            f"[HybridChat] New request - conv_id: {conv_id}, model: {model_name or 'default'}"
        )

        try:
            # Step 1: Get retrievers
            retriever = self.get_or_create_hybrid_retriever(conv_id, model_name, embedding_model)
            citation_retriever = self._citation_retrievers.get_retriever(conv_id)

            if not citation_retriever:
                raise ValueError("Citation retriever not initialized")

            # Step 2: Apply query expansion if enabled
            search_query = message
            if self._query_expander:
                expanded_queries = self._query_expander.expand(message)
                if len(expanded_queries) > 1:
                    logger.info(
                        f"[HybridChat] Query expanded to {len(expanded_queries)} variations"
                    )
                    search_query = expanded_queries[0]

            # Step 3: Metadata Matching (Layer 1)
            logger.info(f"[HybridChat] Step 1: Metadata matching")
            metadata_match = retriever.match_metadata(search_query)
            logger.info(
                f"[HybridChat] Metadata match status: {metadata_match.status}, "
                f"confidence: {metadata_match.confidence:.2f}"
            )

            # Step 4: Citation Retrieval (Layer 2)
            logger.info(f"[HybridChat] Step 2: Citation retrieval")
            citation_start = time.time()

            if metadata_match.status in ["exact", "partial"]:
                # Use filtered citation retrieval
                filter_obj = create_citation_filter_from_match(
                    metadata_match.matched_docs, metadata_match.matched_pages
                )
                citations = citation_retriever.retrieve(
                    search_query, metadata_filter=filter_obj, top_k=5
                )
            else:
                # Use global citation retrieval
                citations = citation_retriever.retrieve(search_query, top_k=5)

            citation_time = time.time() - citation_start
            log_performance(
                logger, "HybridChat", "citation_retrieval",
                citation_time * 1000,  # Convert to ms
                conversation_id=conv_id,
                extra={"citations_found": len(citations)}
            )

            # Step 5: Hybrid RAG Retrieval (Layer 3)
            logger.info(f"[HybridChat] Step 3: Hybrid RAG retrieval")
            retrieve_start = time.time()
            hybrid_results: List[NodeWithScore] = retriever.retrieve(search_query)
            retrieve_time = time.time() - retrieve_start
            log_performance(
                logger, "HybridChat", "hybrid_retrieval",
                retrieve_time * 1000,  # Convert to ms
                conversation_id=conv_id,
                extra={"results_found": len(hybrid_results)}
            )

            # Step 6: Build prompt using strategic builder
            logger.info(f"[HybridChat] Step 4: Building prompt")
            prompt_context = PromptContext(
                query=message,
                metadata_match=metadata_match,
                citations=citations,
                hybrid_results=hybrid_results,
            )
            prompt = self._prompt_builder.build_prompt(prompt_context)

            # Step 7: Generate response using LLM
            logger.info(f"[HybridChat] Step 5: Generating response")
            llm_start = time.time()
            response_text = self._generate_response(prompt, model_name)
            llm_time = time.time() - llm_start
            log_performance(
                logger, "HybridChat", "llm_generation",
                llm_time * 1000,  # Convert to ms
                conversation_id=conv_id
            )

            total_time = time.time() - start_time
            log_performance(
                logger, "HybridChat", "total_request",
                total_time * 1000,  # Convert to ms
                conversation_id=conv_id
            )

            # Build response
            response = ChatResponse(
                response=response_text,
                conversation_id=conv_id,
                metadata_match=metadata_match.to_dict(),
                citations=[c.to_dict() for c in citations],
                retrieved_count=len(hybrid_results),
            )

            return response

        except Exception as e:
            log_error(logger, "HybridChat", e, conversation_id=conv_id)
            raise

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

        # Also clear citation retriever
        self._citation_retrievers.clear_conversation(conversation_id)

        return True

    def clear_all_conversations(self) -> None:
        """Clear all conversation history."""
        super().clear_all_conversations()
        self._hybrid_retrievers.clear()
        self._citation_retrievers.clear_all()
        logger.info("[HybridChat] All retrievers cleared")
