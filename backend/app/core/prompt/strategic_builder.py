"""Strategic prompt builder for hybrid RAG with citation support.

This module builds prompts based on metadata match status and retrieval results.
It supports two modes:
- STRICT: When metadata doesn't match, don't force an answer
- NORMAL: When metadata matches, provide citations
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from app.core.metadata import MetadataMatchResult
from app.core.citation import CitationCandidate
from app.core.retrievers.base import NodeWithScore

logger = logging.getLogger(__name__)


@dataclass
class PromptContext:
    """Context information for prompt building."""

    query: str
    metadata_match: MetadataMatchResult
    citations: List[CitationCandidate]
    hybrid_results: List[NodeWithScore]


class StrategicPromptBuilder:
    """
    Build prompts based on metadata match status.

    Supports two modes:
    1. STRICT: When metadata doesn't match, clearly state it wasn't found
    2. NORMAL: When metadata matches, provide answer with citations
    """

    def __init__(
        self,
        strict_mode: bool = True,
        max_citation_length: int = 300,
        max_hybrid_context_length: int = 2000,
    ):
        """
        Initialize prompt builder.

        Args:
            strict_mode: If True, don't force answers when metadata doesn't match
            max_citation_length: Max length for citation text snippets
            max_hybrid_context_length: Max length for hybrid RAG context
        """
        self.strict_mode = strict_mode
        self.max_citation_length = max_citation_length
        self.max_hybrid_context_length = max_hybrid_context_length

        logger.info(f"[PromptBuilder] Initialized with strict_mode={strict_mode}")

    def build_prompt(self, context: PromptContext) -> str:
        """
        Build appropriate prompt based on metadata match status.

        Args:
            context: PromptContext with all retrieval results

        Returns:
            Complete prompt string for LLM
        """
        match_status = context.metadata_match.status

        if match_status == "none":
            # Metadata didn't match - use strict mode
            return self._build_no_match_prompt(context)
        else:
            # Metadata matched - normal mode with citations
            return self._build_matched_prompt(context)

    def _build_no_match_prompt(self, context: PromptContext) -> str:
        """
        Build prompt when metadata doesn't match (strict mode).

        Clear instructions: Don't make up information about documents/pages
        that don't exist.
        """
        query = context.query
        hybrid_context = self._format_hybrid_context(context.hybrid_results)

        # Note the metadata that was requested but not found
        requested_docs = context.metadata_match.query_metadata.file_names
        requested_pages = context.metadata_match.query_metadata.page_numbers

        missing_info = []
        if requested_docs:
            missing_info.append(f"document(s): {', '.join(requested_docs)}")
        if requested_pages:
            missing_info.append(f"page(s): {', '.join(map(str, requested_pages))}")

        missing_str = "; ".join(missing_info) if missing_info else "specific documents or pages"

        prompt = f"""You are a helpful assistant answering questions based on retrieved context.

⚠️  IMPORTANT: The user's query references {missing_str}, but these were NOT found in the knowledge base.

CRITICAL INSTRUCTIONS:
1. DO NOT make up, invent, or hallucinate information about documents or pages that don't exist.
2. DO NOT pretend to have information from the requested documents/pages.
3. If the context below contains relevant information, you may provide a helpful answer based on it.
4. If the context doesn't contain information about the specific documents/pages mentioned, clearly state that they were not found.
5. Be honest about what you can and cannot answer.

Context from semantic search (may not match the specific documents/pages mentioned):
{hybrid_context}

User Query: {query}

Your response should:
- Acknowledge that the requested documents/pages were not found (if applicable)
- Provide a helpful answer based on available context (if relevant)
- Be clear and honest about limitations

Answer:"""

        return prompt

    def _build_matched_prompt(self, context: PromptContext) -> str:
        """
        Build prompt when metadata matches (normal mode with citations).
        """
        query = context.query
        match_result = context.metadata_match

        # Format citation context
        citation_context = self._format_citation_context(context.citations)

        # Format hybrid RAG context (supplementary)
        hybrid_context = self._format_hybrid_context(context.hybrid_results[:3])  # Top 3 only

        # Build match confirmation
        matched_docs_str = ", ".join(match_result.matched_docs)
        matched_pages_str = ""
        if match_result.matched_pages:
            pages_info = []
            for doc, pages in match_result.matched_pages.items():
                pages_info.append(f"{doc} pages {pages}")
            matched_pages_str = f"\nSpecific pages: {'; '.join(pages_info)}"

        prompt = f"""You are a helpful assistant answering questions based on provided context.

✅ Metadata Match: Successfully matched {matched_docs_str}{matched_pages_str}

PRIMARY CITATION SOURCES (use these as primary references):
{citation_context}

SUPPLEMENTARY CONTEXT (use if additional detail needed):
{hybrid_context}

User Query: {query}

INSTRUCTIONS:
1. Answer based PRIMARILY on the Citation Sources above.
2. Use citations in the format [1], [2], [3] when referencing information.
3. Place citations immediately after the relevant sentence or fact.
4. Use multiple citations if information comes from multiple sources: [1][2]
5. Supplement with Supplementary Context if needed, but prioritize Citation Sources.
6. Be precise about which document and page the information comes from.
7. If the context doesn't contain enough information to answer fully, say so clearly.

Example good citation usage:
"The installation requires Python 3.8 or higher [1]. After installation, you need to configure the settings [2]."

Answer:"""

        return prompt

    def _format_citation_context(self, citations: List[CitationCandidate]) -> str:
        """
        Format citation candidates for prompt context.

        Args:
            citations: List of citation candidates

        Returns:
            Formatted context string
        """
        if not citations:
            return "No specific citations found."

        parts = []
        for citation in citations:
            # Build source info
            source_info = citation.source
            if citation.page:
                source_info += f", Page {citation.page}"
            if citation.section:
                source_info += f", {citation.section}"

            # Truncate text if too long
            text = citation.text
            if len(text) > self.max_citation_length:
                text = text[: self.max_citation_length] + "..."

            part = f"[{citation.index}] {source_info}\n{text}"
            parts.append(part)

        return "\n\n".join(parts)

    def _format_hybrid_context(self, results: List[NodeWithScore]) -> str:
        """
        Format hybrid RAG results for prompt context.

        Args:
            results: List of NodeWithScore from hybrid retrieval

        Returns:
            Formatted context string
        """
        if not results:
            return "No additional context found."

        parts = []
        total_length = 0

        for idx, result in enumerate(results, start=1):
            # Get metadata
            metadata = result.metadata
            source = metadata.get("file_name", "Unknown")
            page = metadata.get("page_number")

            # Build header
            header = f"[{idx}] {source}"
            if page:
                header += f", Page {page}"

            # Truncate if needed
            text = result.text
            remaining = self.max_hybrid_context_length - total_length
            if remaining <= 0:
                break

            if len(text) > remaining:
                text = text[:remaining] + "..."

            part = f"{header}\n{text}"
            parts.append(part)
            total_length += len(part)

        return "\n\n".join(parts)

    def build_system_message(self, metadata_match_status: str) -> str:
        """
        Build system message based on match status.

        Args:
            metadata_match_status: "exact", "partial", or "none"

        Returns:
            System message for LLM
        """
        base_message = "You are a helpful assistant answering questions based on provided context."

        if metadata_match_status == "none" and self.strict_mode:
            return (
                base_message
                + "\n\nIMPORTANT: If specific documents or pages are requested but not found in the context, clearly state this and do not make up information."
            )

        return (
            base_message
            + "\n\nUse citations [1], [2], etc. when referencing information from the context."
        )


class SimplePromptBuilder:
    """
    Simple prompt builder for backward compatibility.

    This maintains the original behavior for non-citation use cases.
    """

    def build_prompt(self, query: str, context: str) -> str:
        """
        Build simple prompt without citations.

        Args:
            query: User query
            context: Retrieved context

        Returns:
            Prompt string
        """
        return f"""Based on the following context, please answer the question.

Context:
{context}

Question: {query}

Please provide a comprehensive answer based only on the context provided above. If the context doesn't contain enough information to answer the question, please say so.

Answer:"""
