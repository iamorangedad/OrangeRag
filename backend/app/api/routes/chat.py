"""Chat routes."""

from fastapi import APIRouter, HTTPException

from app.services.hybrid_chat_service import HybridChatService
from app.models.schemas import ChatRequest, ChatResponse, CitationInfo, MetadataMatchInfo

router = APIRouter()
chat_service = HybridChatService()


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Send a chat message with three-layer retrieval and citation support.

    This endpoint uses a three-layer architecture:
    1. Metadata Matching - validates document/page references
    2. Citation Retrieval - retrieves exact text snippets
    3. Hybrid RAG - generates answers with semantic understanding

    When metadata doesn't match (e.g., document not found),
    the system clearly states this rather than forcing an answer.
    """
    try:
        result = chat_service.chat(
            message=request.message,
            conversation_id=request.conversation_id,
            model_name=request.model,
            embedding_model=request.embedding_model,
        )

        # Convert citations to schema model
        citations = [CitationInfo(**c) for c in result.citations]

        # Convert metadata match to schema model
        metadata_match = MetadataMatchInfo(**result.metadata_match)

        return ChatResponse(
            response=result.response,
            conversation_id=result.conversation_id,
            metadata_match=metadata_match,
            citations=citations,
            retrieved_count=result.retrieved_count,
        )

    except ValueError as e:
        # No documents uploaded
        return ChatResponse(
            response="Please upload documents first, then I can answer your questions.",
            conversation_id=request.conversation_id or chat_service.create_conversation(),
            metadata_match=MetadataMatchInfo(
                status="none", confidence=0.0, message="No documents available"
            ),
            citations=[],
            retrieved_count=0,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")
