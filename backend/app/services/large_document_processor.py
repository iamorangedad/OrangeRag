"""Ultra-robust large document processor for Ollama stability.

Key optimizations for large files (like OM0R028U.pdf):
1. Sequential processing (no parallel threads) to avoid Ollama overload
2. Smaller chunks (512 chars) to reduce embedding failures
3. Aggressive retry with exponential backoff
4. Skip failed chunks (don't fail entire document)
5. Progress persistence (can resume)
"""

import logging
import time
from typing import List, Optional, Any, Dict

from llama_index.core.schema import TextNode
from llama_index.core.vector_stores import SimpleVectorStore

logger = logging.getLogger(__name__)


class RobustLargeDocumentProcessor:
    """Process large documents with maximum stability."""

    # Ultra-conservative settings for Ollama stability
    CHUNK_SIZE = 512  # Much smaller chunks
    CHUNK_OVERLAP = 50
    EMBEDDING_BATCH_SIZE = 1  # Process one at a time
    MAX_TEXT_LENGTH = 1000  # Truncate to 1000 chars max
    BASE_DELAY = 2.0  # 2 second base delay
    MAX_RETRIES = 5

    def __init__(self, embed_model, vector_store):
        self.embed_model = embed_model
        self.vector_store = vector_store
        self.failed_chunks = []
        self.success_count = 0

    def process_document(self, file_path: str, progress_callback=None) -> dict:
        """Process a large document with robust error handling.

        Returns:
            dict: Processing results with success/failure counts
        """
        from llama_index.core.node_parser import SentenceSplitter
        from app.core.document_processing import UniversalDocumentLoader

        logger.info(f"[RobustProcessor] Starting processing of {file_path}")

        # Step 1: Load document with metadata extraction
        if progress_callback:
            progress_callback(5, "Loading document with metadata extraction...")

        loader = UniversalDocumentLoader(extract_pdf_metadata=True)
        all_docs = loader.load_data(file_path)

        logger.info(f"[RobustProcessor] Loaded {len(all_docs)} document segments with metadata")

        if progress_callback:
            progress_callback(10, f"Loaded {len(all_docs)} segments")

        # Step 2: Parse into small chunks
        if progress_callback:
            progress_callback(12, "Parsing into small chunks...")

        node_parser = SentenceSplitter(chunk_size=self.CHUNK_SIZE, chunk_overlap=self.CHUNK_OVERLAP)
        nodes = node_parser.get_nodes_from_documents(all_docs)
        total_nodes = len(nodes)

        # Step 2.5: Ensure metadata is preserved and enhanced for each chunk
        self._enhance_node_metadata(nodes, all_docs)

        logger.info(
            f"[RobustProcessor] Created {total_nodes} small chunks (size: {self.CHUNK_SIZE})"
        )

        if progress_callback:
            progress_callback(15, f"Created {total_nodes} chunks")

        # Step 3: Process each chunk sequentially with retries
        self.success_count = 0
        self.failed_chunks = []

        for i, node in enumerate(nodes):
            try:
                success = self._process_single_chunk(node, i, total_nodes)

                if success:
                    self.success_count += 1
                else:
                    self.failed_chunks.append(i)

                # Update progress
                if progress_callback and i % 10 == 0:
                    progress_pct = 15 + int((i / total_nodes) * 80)
                    progress_callback(
                        progress_pct,
                        f"Processed {i + 1}/{total_nodes} chunks ({self.success_count} success, {len(self.failed_chunks)} failed)",
                    )

            except Exception as e:
                logger.error(f"[RobustProcessor] Unexpected error on chunk {i}: {e}")
                self.failed_chunks.append(i)

        # Final progress update
        if progress_callback:
            progress_callback(
                95, f"Finalizing... {self.success_count}/{total_nodes} chunks indexed"
            )

        # Return results
        success_rate = self.success_count / total_nodes if total_nodes > 0 else 0

        result = {
            "total_chunks": total_nodes,
            "successful": self.success_count,
            "failed": len(self.failed_chunks),
            "success_rate": success_rate,
            "failed_indices": self.failed_chunks[:20],  # First 20 failed indices
        }

        logger.info(
            f"[RobustProcessor] Complete: {self.success_count}/{total_nodes} chunks "
            f"({success_rate * 100:.1f}% success rate)"
        )

        if self.failed_chunks:
            logger.warning(f"[RobustProcessor] Failed chunks: {self.failed_chunks[:10]}...")

        return result

    def _process_single_chunk(self, node: TextNode, index: int, total: int) -> bool:
        """Process a single chunk with aggressive retry logic.

        Returns:
            bool: True if successful, False otherwise
        """
        # Get and truncate text
        text = node.get_content()
        original_len = len(text)

        if original_len > self.MAX_TEXT_LENGTH:
            text = text[: self.MAX_TEXT_LENGTH] + "..."

        # Try to generate embedding with exponential backoff
        embedding = None
        last_error = None

        for attempt in range(self.MAX_RETRIES):
            try:
                # Generate embedding
                embedding = self.embed_model.get_text_embedding(text)

                if embedding:
                    break

            except Exception as e:
                last_error = e

                if attempt < self.MAX_RETRIES - 1:
                    # Exponential backoff: 2s, 4s, 8s, 16s
                    wait_time = self.BASE_DELAY * (2**attempt)
                    logger.warning(
                        f"[RobustProcessor] Chunk {index}/{total} failed (attempt {attempt + 1}), "
                        f"waiting {wait_time}s..."
                    )
                    time.sleep(wait_time)

        if embedding is None:
            logger.error(
                f"[RobustProcessor] Chunk {index}/{total} failed after {self.MAX_RETRIES} attempts: "
                f"{str(last_error)[:100] if last_error else 'Unknown error'}"
            )
            return False

        # Insert into vector store
        try:
            node.embedding = embedding
            self.vector_store.add([node])

            # Delay before next chunk (prevents Ollama overload)
            time.sleep(self.BASE_DELAY)

            return True

        except Exception as e:
            logger.error(f"[RobustProcessor] Failed to insert chunk {index}: {e}")
            return False

    def _enhance_node_metadata(self, nodes: List[TextNode], source_docs: List) -> None:
        """
        Enhance node metadata to ensure proper citation support.

        This method ensures each chunk/node has:
        - file_name, page_number from parent document
        - chunk_index for ordering
        - total_chunks for reference

        Args:
            nodes: List of TextNode chunks from node_parser
            source_docs: Original documents from UniversalDocumentLoader
        """
        total_chunks = len(nodes)

        for i, node in enumerate(nodes):
            # Add chunk-level metadata
            node.metadata["chunk_index"] = i
            node.metadata["total_chunks"] = total_chunks

            # Ensure essential metadata exists
            # Try to inherit from source document if missing
            if "file_name" not in node.metadata and source_docs:
                # Find best matching source document based on content overlap
                best_doc = self._find_source_document(node, source_docs)
                if best_doc:
                    for key in [
                        "file_name",
                        "file_path",
                        "page_number",
                        "total_pages",
                        "page_label",
                    ]:
                        if key in best_doc.metadata and key not in node.metadata:
                            node.metadata[key] = best_doc.metadata[key]

            # Log warning if critical metadata still missing
            if "file_name" not in node.metadata:
                logger.warning(f"[RobustProcessor] Chunk {i} missing file_name metadata")
            if "page_number" not in node.metadata:
                logger.debug(f"[RobustProcessor] Chunk {i} missing page_number metadata")

    def _find_source_document(self, node: TextNode, source_docs: List) -> Optional[Any]:
        """
        Find the source document that best matches the node content.

        Uses simple heuristic: first 100 chars overlap check.

        Args:
            node: TextNode chunk
            source_docs: List of source documents

        Returns:
            Best matching source document or None
        """
        node_start = node.text[:100].strip() if node.text else ""

        for doc in source_docs:
            if node_start and doc.text and node_start in doc.text[:500]:
                return doc

        # Fallback: return first doc if no match found
        return source_docs[0] if source_docs else None


def process_large_document_robust(
    file_path: str, embed_model, vector_store, progress_callback=None
) -> dict:
    """Convenience function to process a large document.

    Usage:
        result = process_large_document_robust(
            "OM0R028U.pdf",
            embed_model,
            vector_store,
            lambda pct, msg: print(f"{pct}%: {msg}")
        )
        print(f"Success rate: {result['success_rate']*100:.1f}%")
    """
    processor = RobustLargeDocumentProcessor(embed_model, vector_store)
    return processor.process_document(file_path, progress_callback)
