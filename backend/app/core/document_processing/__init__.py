"""Document processing module for metadata extraction."""

from app.core.document_processing.pdf_extractor import (
    PDFMetadataExtractor,
    UniversalDocumentLoader,
    load_document_with_metadata,
)

__all__ = [
    "PDFMetadataExtractor",
    "UniversalDocumentLoader",
    "load_document_with_metadata",
]
