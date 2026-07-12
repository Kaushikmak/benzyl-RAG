import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

import app.config as config
from indexing.document_loader import (
    _format_unstructured_elements,
    _is_pdf_file,
    _discover_external_files,
    parse_pdf_with_docling,
    parse_non_pdf_with_tika,
    load_documents,
)


class DummyElement:
    def __init__(self, type_name, text):
        self.category = type_name
        self._text = text

    def __str__(self):
        return self._text


class TestDocumentLoader(unittest.TestCase):
    def test_is_pdf_file(self):
        self.assertTrue(_is_pdf_file("report.pdf"))
        self.assertTrue(_is_pdf_file("/path/to/doc.PDF"))
        self.assertFalse(_is_pdf_file("notes.docx"))
        self.assertFalse(_is_pdf_file("data.xlsx"))

    def test_format_unstructured_elements(self):
        elements = [
            DummyElement("Title", "Architecture Overview"),
            DummyElement("NarrativeText", "This document describes the design."),
            DummyElement("Table", "| Col1 | Col2 |\n| ---- | ---- |\n| A    | B    |"),
        ]
        formatted = _format_unstructured_elements(elements)
        self.assertIn("## Architecture Overview", formatted)
        self.assertIn("This document describes the design.", formatted)
        self.assertIn("| Col1 | Col2 |", formatted)

    @patch("indexing.document_loader._fallback_load")
    def test_parse_pdf_with_docling_fallback(self, mock_fallback):
        mock_fallback.return_value = [
            Document(
                page_content="Fallback text",
                metadata={"doc_type": "external", "parser_pipeline": ["fallback_pdf"]},
            )
        ]
        # With docling missing/mocked exception
        docs = parse_pdf_with_docling("non_existent.pdf")
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0].metadata.get("doc_type"), "external")

    @patch("indexing.document_loader._fallback_load")
    def test_parse_non_pdf_with_tika_fallback(self, mock_fallback):
        mock_fallback.return_value = [
            Document(
                page_content="Tika fallback text",
                metadata={"doc_type": "external", "parser_pipeline": ["fallback"]},
            )
        ]
        docs = parse_non_pdf_with_tika("non_existent.docx")
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0].metadata.get("doc_type"), "external")

    def test_discover_external_files_universal(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            file_pdf = os.path.join(tmpdir, "test.pdf")
            file_xlsx = os.path.join(tmpdir, "data.xlsx")
            file_txt = os.path.join(tmpdir, "notes.txt")
            with open(file_pdf, "w") as f:
                f.write("pdf content")
            with open(file_xlsx, "w") as f:
                f.write("xlsx content")
            with open(file_txt, "w") as f:
                f.write("txt content")

            with patch.object(config, "EXTERNAL_DOCS_PATH", tmpdir), \
                 patch.object(config, "INGESTION_UNIVERSAL_LOAD", True):
                discovered = _discover_external_files()
                self.assertEqual(len(discovered), 3)
                self.assertIn(file_pdf, discovered)
                self.assertIn(file_xlsx, discovered)
                self.assertIn(file_txt, discovered)


if __name__ == "__main__":
    unittest.main()
