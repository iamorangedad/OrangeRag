"""Pydantic models for API requests and responses."""

from typing import List, Optional, Dict
from pydantic import BaseModel, Field
from enum import Enum


class TaskStatusEnum(str, Enum):
    """Task status enumeration."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""

    message: str = Field(..., description="User message")
    conversation_id: Optional[str] = Field(None, description="Conversation ID for context")
    model: Optional[str] = Field(None, description="LLM model name to use")
    embedding_model: Optional[str] = Field(None, description="Embedding model name to use")


class CitationInfo(BaseModel):
    """Citation information for sources."""

    index: int = Field(..., description="Citation index number [1], [2], etc.")
    text: str = Field(..., description="Cited text snippet (truncated)")
    source: str = Field(..., description="Source document filename")
    page: Optional[int] = Field(None, description="Page number if available")
    page_label: Optional[str] = Field(None, description="Page label (e.g., 'Page 5')")
    section: Optional[str] = Field(None, description="Section title if available")
    score: float = Field(..., description="BM25 relevance score")


class MetadataMatchInfo(BaseModel):
    """Metadata matching result information."""

    status: str = Field(..., description="Match status: exact, partial, or none")
    matched_docs: List[str] = Field(
        default_factory=list, description="List of matched document names"
    )
    matched_pages: Dict[str, List[int]] = Field(
        default_factory=dict, description="Matched pages per document"
    )
    confidence: float = Field(..., description="Match confidence score 0.0-1.0")
    message: str = Field(..., description="Human readable match message")


class ChatResponse(BaseModel):
    """Response model for chat endpoint with citation support."""

    response: str = Field(..., description="AI response")
    conversation_id: str = Field(..., description="Conversation ID")
    metadata_match: MetadataMatchInfo = Field(
        default_factory=lambda: MetadataMatchInfo(
            status="none", confidence=0.0, message="No metadata matching performed"
        ),
        description="Metadata matching result",
    )
    citations: List[CitationInfo] = Field(
        default_factory=list, description="List of citations used in the response"
    )
    retrieved_count: int = Field(default=0, description="Number of documents retrieved for context")


class ModelInfo(BaseModel):
    """Model information."""

    name: str = Field(..., description="Model name")
    size: str = Field(..., description="Model size")
    parameter_size: str = Field(..., description="Parameter count")
    quantization: str = Field(..., description="Quantization type")
    requires_gpu: bool = Field(..., description="Whether GPU is required")


class RecommendedModel(BaseModel):
    """Recommended model information."""

    name: str = Field(..., description="Model name")
    description: str = Field(..., description="Model description")
    context_window: str = Field(..., description="Context window size")
    gpu_required: bool = Field(..., description="Whether GPU is required")


class EmbeddingModelInfo(BaseModel):
    """Embedding model information."""

    name: str = Field(..., description="Model name")
    description: str = Field(..., description="Model description")
    dimensions: int = Field(..., description="Embedding dimensions")
    gpu_required: bool = Field(..., description="Whether GPU is required")


class UploadResponse(BaseModel):
    """Response model for file upload."""

    message: str = Field(..., description="Success message")
    filename: str = Field(..., description="Uploaded filename")
    size: int = Field(..., description="File size in bytes")
    task_id: Optional[str] = Field(None, description="Async indexing task ID")


class DocumentListResponse(BaseModel):
    """Response model for document list."""

    documents: List[str] = Field(..., description="List of document filenames")


class DeleteResponse(BaseModel):
    """Response model for delete operation."""

    message: str = Field(..., description="Success message")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status")
    vector_store: str = Field(..., description="Vector store type")
    ollama_url: str = Field(..., description="Ollama service URL")
    documents_count: int = Field(..., description="Number of uploaded documents")


class ModelsListResponse(BaseModel):
    """Response model for models list."""

    models: List[ModelInfo] = Field(..., description="List of available models")
    count: int = Field(..., description="Total count of models")


class RecommendedModelsResponse(BaseModel):
    """Response model for recommended models."""

    recommended_models: List[RecommendedModel] = Field(..., description="Recommended LLM models")
    embedding_models: List[EmbeddingModelInfo] = Field(
        ..., description="Recommended embedding models"
    )


class IndexingTaskResponse(BaseModel):
    """Response model for indexing task."""

    task_id: str = Field(..., description="Task ID")
    filename: str = Field(..., description="Document filename")
    status: TaskStatusEnum = Field(..., description="Task status")
    progress: int = Field(..., description="Progress percentage (0-100)")
    message: str = Field(..., description="Status message")
    created_at: Optional[str] = Field(None, description="Creation timestamp")
    completed_at: Optional[str] = Field(None, description="Completion timestamp")
    error: Optional[str] = Field(None, description="Error message if failed")
    total_pages: int = Field(0, description="Total number of pages/chunks")
    processed_pages: int = Field(0, description="Number of pages/chunks processed")
    chunk_count: int = Field(0, description="Total chunk count for large documents")


class TaskListResponse(BaseModel):
    """Response model for task list."""

    tasks: List[IndexingTaskResponse] = Field(..., description="List of indexing tasks")
    pending_count: int = Field(..., description="Number of pending tasks")
    processing_count: int = Field(..., description="Number of processing tasks")


class ProcessingStatusResponse(BaseModel):
    """Response for processing status summary."""

    has_pending_tasks: bool = Field(..., description="Whether there are pending tasks")
    has_processing_tasks: bool = Field(..., description="Whether there are processing tasks")
    pending_count: int = Field(..., description="Number of pending tasks")
    processing_count: int = Field(..., description="Number of processing tasks")
    tasks: List[IndexingTaskResponse] = Field(..., description="Recent tasks")


class DocumentVectorStatus(BaseModel):
    """Vector database status for a single document."""

    filename: str = Field(..., description="Document filename")
    is_uploaded: bool = Field(..., description="Whether file is uploaded")
    is_indexed: bool = Field(..., description="Whether file is indexed in vector store")
    indexing_status: str = Field(
        ..., description="Indexing status: pending/processing/completed/failed/none"
    )
    progress: int = Field(..., description="Progress percentage (0-100), 0 if not indexed")
    chunk_count: int = Field(..., description="Number of chunks/segments created, 0 if not indexed")
    message: str = Field(..., description="Status message")


class VectorKnowledgeBaseStatusResponse(BaseModel):
    """Response for vector knowledge base status."""

    total_documents: int = Field(..., description="Total number of uploaded documents")
    indexed_documents: int = Field(..., description="Number of documents fully indexed")
    processing_documents: int = Field(..., description="Number of documents being processed")
    pending_documents: int = Field(..., description="Number of documents pending indexing")
    failed_documents: int = Field(..., description="Number of documents failed indexing")
    documents: List[DocumentVectorStatus] = Field(..., description="List of all document statuses")
