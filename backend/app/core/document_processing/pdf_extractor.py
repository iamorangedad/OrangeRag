"""PDF document processor with metadata extraction.

This module provides enhanced PDF loading capabilities that extract page numbers
and other metadata for citation purposes.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from llama_index.core.schema import Document

logger = logging.getLogger(__name__)


class PDFMetadataExtractor:
    """
    Extract PDF content with page number metadata.

    Uses pdfplumber to extract text page by page, preserving page number
    information for accurate citation support.

    Example:
        extractor = PDFMetadataExtractor()
        documents = extractor.load_data("document.pdf")
        # Each document has metadata: file_name, page_number, total_pages
    """

    def __init__(self, extract_page_numbers: bool = True):
        """
        Initialize PDF metadata extractor.

        Args:
            extract_page_numbers: Whether to extract page numbers (default: True)
        """
        self.extract_page_numbers = extract_page_numbers

    def load_data(self, file_path: str) -> List[Document]:
        """
        Load PDF and extract content with page metadata.

        Args:
            file_path: Path to PDF file

        Returns:
            List of Document objects with page metadata

        Raises:
            ImportError: If pdfplumber is not installed
            FileNotFoundError: If file doesn't exist
        """
        try:
            import pdfplumber
        except ImportError:
            logger.error("pdfplumber not installed. Install with: pip install pdfplumber")
            raise ImportError("pdfplumber is required for PDF metadata extraction")

        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        logger.info(f"[PDFExtractor] Loading PDF: {file_path.name}")

        documents = []

        try:
            with pdfplumber.open(file_path) as pdf:
                total_pages = len(pdf.pages)
                logger.info(f"[PDFExtractor] Total pages: {total_pages}")

                for page_num, page in enumerate(pdf.pages, start=1):
                    # Extract text from page
                    text = page.extract_text() or ""

                    if not text.strip():
                        logger.debug(f"[PDFExtractor] Page {page_num} is empty, skipping")
                        continue

                    # Build metadata
                    metadata = {
                        "file_name": file_path.name,
                        "file_path": str(file_path.absolute()),
                        "total_pages": total_pages,
                    }

                    if self.extract_page_numbers:
                        metadata["page_number"] = page_num
                        metadata["page_label"] = f"Page {page_num}"

                    # Create document
                    doc = Document(text=text, metadata=metadata)
                    documents.append(doc)

                    logger.debug(f"[PDFExtractor] Extracted page {page_num}/{total_pages}")

        except Exception as e:
            logger.error(f"[PDFExtractor] Failed to process PDF {file_path}: {e}")
            raise

        logger.info(
            f"[PDFExtractor] Successfully extracted {len(documents)} pages from {file_path.name}"
        )
        return documents


class UniversalDocumentLoader:
    """
    Universal document loader that automatically selects the best extractor.

    - PDF files: Uses PDFMetadataExtractor with page number extraction
    - Other files: Falls back to SimpleDirectoryReader

    Example:
        loader = UniversalDocumentLoader()
        documents = loader.load_data("document.pdf")  # Extracts page numbers
        documents = loader.load_data("document.txt")  # Uses standard loader
    """

    SUPPORTED_EXTENSIONS = {
        ".pdf": PDFMetadataExtractor,
    }

    def __init__(self, extract_pdf_metadata: bool = True):
        """
        Initialize universal document loader.

        Args:
            extract_pdf_metadata: Whether to extract metadata from PDFs (default: True)
        """
        self.extract_pdf_metadata = extract_pdf_metadata
        self._pdf_extractor: Optional[PDFMetadataExtractor] = None

    def load_data(self, file_path: str) -> List[Document]:
        """
        Load document with appropriate extractor.

        Args:
            file_path: Path to document file

        Returns:
            List of Document objects with metadata
        """
        file_path = Path(file_path)
        ext = file_path.suffix.lower()

        # Check if we have a specialized extractor for this file type
        if ext in self.SUPPORTED_EXTENSIONS and self.extract_pdf_metadata:
            extractor_class = self.SUPPORTED_EXTENSIONS[ext]
            extractor = extractor_class()
            logger.info(f"[UniversalLoader] Using {extractor_class.__name__} for {file_path.name}")
            return extractor.load_data(str(file_path))
        else:
            # Fall back to SimpleDirectoryReader
            logger.info(f"[UniversalLoader] Using SimpleDirectoryReader for {file_path.name}")
            return self._load_with_simple_reader(file_path)

    def _load_with_simple_reader(self, file_path: Path) -> List[Document]:
        """
        Load document using SimpleDirectoryReader.

        Args:
            file_path: Path to document file

        Returns:
            List of Document objects
        """
        from llama_index.core import SimpleDirectoryReader

        reader = SimpleDirectoryReader(input_files=[str(file_path)], filename_as_id=True)
        documents = reader.load_data()

        # Add basic metadata for non-PDF files
        for doc in documents:
            if "file_name" not in doc.metadata:
                doc.metadata["file_name"] = file_path.name
            if "file_path" not in doc.metadata:
                doc.metadata["file_path"] = str(file_path.absolute())

        return documents


def load_document_with_metadata(file_path: str) -> List[Document]:
    """
    Convenience function to load document with metadata extraction.

    Args:
        file_path: Path to document file

    Returns:
        List of Document objects with metadata

    Example:
        >>> docs = load_document_with_metadata("report.pdf")
        >>> print(docs[0].metadata)
        {'file_name': 'report.pdf', 'page_number': 1, 'total_pages': 10}
    """
    loader = UniversalDocumentLoader()
    return loader.load_data(file_path)
