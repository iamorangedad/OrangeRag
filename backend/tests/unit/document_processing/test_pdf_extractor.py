"""Tests for PDF metadata extraction module."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path


class TestPDFMetadataExtractor:
    """Test cases for PDFMetadataExtractor."""

    def test_init_default(self):
        """Test default initialization."""
        from app.core.document_processing.pdf_extractor import PDFMetadataExtractor

        extractor = PDFMetadataExtractor()
        assert extractor.extract_page_numbers is True

    def test_init_custom(self):
        """Test initialization with custom parameters."""
        from app.core.document_processing.pdf_extractor import PDFMetadataExtractor

        extractor = PDFMetadataExtractor(extract_page_numbers=False)
        assert extractor.extract_page_numbers is False

    @patch("app.core.document_processing.pdf_extractor.Path")
    def test_load_data_file_not_found(self, mock_path):
        """Test handling of non-existent file."""
        from app.core.document_processing.pdf_extractor import PDFMetadataExtractor

        # Setup mock
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = False
        mock_path.return_value = mock_path_instance

        extractor = PDFMetadataExtractor()

        with pytest.raises(FileNotFoundError):
            extractor.load_data("nonexistent.pdf")

    @patch("app.core.document_processing.pdf_extractor.pdfplumber")
    @patch("app.core.document_processing.pdf_extractor.Path")
    def test_load_data_success(self, mock_path, mock_pdfplumber):
        """Test successful PDF loading with metadata."""
        from app.core.document_processing.pdf_extractor import PDFMetadataExtractor

        # Setup path mock
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path_instance.name = "test.pdf"
        mock_path_instance.absolute.return_value = Path("/path/to/test.pdf")
        mock_path.return_value = mock_path_instance

        # Setup pdfplumber mock
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Page 1 content"

        mock_pdf = MagicMock()
        mock_pdf.pages = [mock_page]
        mock_pdfplumber.open.return_value.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdfplumber.open.return_value.__exit__ = MagicMock(return_value=False)

        extractor = PDFMetadataExtractor()
        documents = extractor.load_data("test.pdf")

        assert len(documents) == 1
        assert documents[0].text == "Page 1 content"
        assert documents[0].metadata["file_name"] == "test.pdf"
        assert documents[0].metadata["page_number"] == 1
        assert documents[0].metadata["total_pages"] == 1

    @patch("app.core.document_processing.pdf_extractor.pdfplumber")
    @patch("app.core.document_processing.pdf_extractor.Path")
    def test_load_data_multiple_pages(self, mock_path, mock_pdfplumber):
        """Test loading PDF with multiple pages."""
        from app.core.document_processing.pdf_extractor import PDFMetadataExtractor

        # Setup
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path_instance.name = "multi.pdf"
        mock_path_instance.absolute.return_value = Path("/path/to/multi.pdf")
        mock_path.return_value = mock_path_instance

        # Create multiple pages
        mock_pages = []
        for i in range(3):
            mock_page = MagicMock()
            mock_page.extract_text.return_value = f"Page {i + 1} content"
            mock_pages.append(mock_page)

        mock_pdf = MagicMock()
        mock_pdf.pages = mock_pages
        mock_pdfplumber.open.return_value.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdfplumber.open.return_value.__exit__ = MagicMock(return_value=False)

        extractor = PDFMetadataExtractor()
        documents = extractor.load_data("multi.pdf")

        assert len(documents) == 3
        for i, doc in enumerate(documents):
            assert doc.metadata["page_number"] == i + 1
            assert doc.metadata["total_pages"] == 3


class TestUniversalDocumentLoader:
    """Test cases for UniversalDocumentLoader."""

    def test_init_default(self):
        """Test default initialization."""
        from app.core.document_processing.pdf_extractor import UniversalDocumentLoader

        loader = UniversalDocumentLoader()
        assert loader.extract_pdf_metadata is True

    @patch("app.core.document_processing.pdf_extractor.PDFMetadataExtractor")
    @patch("app.core.document_processing.pdf_extractor.Path")
    def test_load_pdf_file(self, mock_path, mock_extractor_class):
        """Test loading PDF file uses PDFMetadataExtractor."""
        from app.core.document_processing.pdf_extractor import UniversalDocumentLoader

        # Setup
        mock_path_instance = MagicMock()
        mock_path_instance.suffix.lower.return_value = ".pdf"
        mock_path.return_value = mock_path_instance

        mock_extractor = MagicMock()
        mock_extractor.load_data.return_value = [Mock()]
        mock_extractor_class.return_value = mock_extractor

        loader = UniversalDocumentLoader()
        result = loader.load_data("document.pdf")

        mock_extractor_class.assert_called_once()
        mock_extractor.load_data.assert_called_once_with("document.pdf")
        assert len(result) == 1

    @patch("app.core.document_processing.pdf_extractor.SimpleDirectoryReader")
    @patch("app.core.document_processing.pdf_extractor.Path")
    def test_load_non_pdf_file(self, mock_path, mock_reader_class):
        """Test loading non-PDF file falls back to SimpleDirectoryReader."""
        from app.core.document_processing.pdf_extractor import UniversalDocumentLoader

        # Setup
        mock_path_instance = MagicMock()
        mock_path_instance.suffix.lower.return_value = ".txt"
        mock_path_instance.name = "document.txt"
        mock_path_instance.absolute.return_value = Path("/path/to/document.txt")
        mock_path.return_value = mock_path_instance

        mock_doc = MagicMock()
        mock_doc.metadata = {}

        mock_reader = MagicMock()
        mock_reader.load_data.return_value = [mock_doc]
        mock_reader_class.return_value = mock_reader

        loader = UniversalDocumentLoader()
        result = loader.load_data("document.txt")

        mock_reader_class.assert_called_once()
        assert len(result) == 1
        # Verify metadata was added
        assert result[0].metadata["file_name"] == "document.txt"


class TestMetadataPreservation:
    """Test that metadata is properly preserved through the pipeline."""

    def test_node_metadata_structure(self):
        """Test that TextNode can store the expected metadata fields."""
        from llama_index.core.schema import TextNode

        node = TextNode(
            text="Test content",
            metadata={
                "file_name": "test.pdf",
                "file_path": "/path/to/test.pdf",
                "page_number": 5,
                "total_pages": 100,
                "chunk_index": 10,
                "total_chunks": 50,
            },
        )

        assert node.metadata["file_name"] == "test.pdf"
        assert node.metadata["page_number"] == 5
        assert node.metadata["chunk_index"] == 10
