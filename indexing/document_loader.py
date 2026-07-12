import glob
import logging
import mimetypes
import os
import re
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    UnstructuredFileLoader,
)

import app.config as config

logger = logging.getLogger(__name__)


def _clean_pdf_text(text: str) -> str:
    """Fix common PDF extraction artifacts like hyphens, ligatures, and smashed words."""
    if not text:
        return text
    # Fix words split across lines by hyphens (e.g., "signifi-\ncantly")
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
    # Replace special ligatures that often break spacing
    text = text.replace('ﬁ', 'fi').replace('ﬂ', 'fl').replace('ﬀ', 'ff').replace('ﬃ', 'ffi')
    # Fix missing spaces between lowercase and uppercase letters (if any)
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    return text


def _is_garbage_text(text: str) -> bool:
    """Detects if extracted text is corrupted OCR garbage or empty."""
    text = text.strip()
    if not text:
        return True
        
    # Count how many standard alphabetical characters exist
    alphas = sum(1 for c in text if c.isalpha())
    
    # Rule 1: If less than 40% of the characters are actual letters, it's gibberish
    if alphas / len(text) < 0.40:
        return True
        
    # Rule 2: If there are fewer than 3 actual recognizable words, it's essentially empty
    words = re.findall(r'[a-zA-Z]{3,}', text)
    if len(words) < 3:
        return True
        
    return False


def _is_pdf_file(file_path: str) -> bool:
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return True
    mime_type, _ = mimetypes.guess_type(file_path)
    return mime_type == "application/pdf"


def _format_unstructured_elements(elements: List[Any]) -> str:
    """Format unstructured elements into clean structured markdown/text."""
    if not elements:
        return ""
    sections = []
    for el in elements:
        el_type = getattr(el, "category", None) or el.__class__.__name__
        text = str(el).strip()
        if not text:
            continue
        if el_type in ("Title", "Header"):
            sections.append(f"## {text}")
        elif el_type == "Table":
            # Preserve table representation
            sections.append(f"\n{text}\n")
        else:
            sections.append(text)
    return "\n\n".join(sections)


def parse_pdf_with_docling(file_path: str) -> List[Document]:
    """Parse PDF documents using Docling for layout/tables, then Unstructured partition_md."""
    if not getattr(config, "INGESTION_USE_DOCLING_FOR_PDF", True):
        return _fallback_load(file_path, pipeline=["unstructured_fallback"])

    try:
        from docling.document_converter import DocumentConverter

        converter = DocumentConverter()
        conv_res = converter.convert(file_path)
        markdown_content = conv_res.document.export_to_markdown()

        # --- PRODUCTION FIX: GIBBERISH CHECK ---
        if _is_garbage_text(markdown_content):
            logger.info("Detected scanned/corrupted PDF %s. Routing to hi_res OCR fallback.", file_path)
            return _fallback_load(file_path, pipeline=["unstructured_hi_res"])
        # ---------------------------------------

        # Pass structured Markdown into Unstructured Markdown partitioner
        structured_text = markdown_content
        elements_count = 0
        pipeline = ["docling"]

        if getattr(config, "INGESTION_USE_UNSTRUCTURED", True):
            try:
                from unstructured.partition.md import partition_md

                elements = partition_md(text=markdown_content)
                elements_count = len(elements)
                formatted = _format_unstructured_elements(elements)
                if formatted.strip():
                    structured_text = formatted
                pipeline.append("unstructured_md")
            except Exception as e:
                logger.warning("Unstructured partition_md failed on %s: %s", file_path, e)

        # Apply PDF text cleaning
        structured_text = _clean_pdf_text(structured_text)

        doc = Document(
            page_content=structured_text,
            metadata={
                "doc_type": "external",
                "source": file_path,
                "external_ext": os.path.splitext(file_path)[1].lower(),
                "parser_pipeline": pipeline,
                "mime_type": "application/pdf",
                "elements_count": elements_count,
            },
        )
        return [doc]
    except Exception as e:
        logger.warning("Docling PDF parsing failed for %s (%s). Falling back.", file_path, e)
        return _fallback_load(file_path, pipeline=["fallback_pdf"])


def parse_non_pdf_with_tika(file_path: str) -> List[Document]:
    """Parse non-PDF documents using Apache Tika for metadata/content, then Unstructured."""
    if not getattr(config, "INGESTION_USE_TIKA", True):
        return _fallback_load(file_path, pipeline=["unstructured_fallback"])

    try:
        from tika import parser as tika_parser

        parsed = tika_parser.from_file(file_path)
        metadata: Dict[str, Any] = parsed.get("metadata") or {}
        raw_content: str = parsed.get("content") or ""

        mime_type = metadata.get("Content-Type") or mimetypes.guess_type(file_path)[0] or "application/octet-stream"
        if isinstance(mime_type, list):
            mime_type = mime_type[0]

        structured_text = raw_content.strip()
        elements_count = 0
        pipeline = ["tika"]

        if getattr(config, "INGESTION_USE_UNSTRUCTURED", True) and structured_text:
            try:
                from unstructured.partition.auto import partition

                elements = partition(filename=file_path)
                elements_count = len(elements)
                formatted = _format_unstructured_elements(elements)
                if formatted.strip():
                    structured_text = formatted
                pipeline.append("unstructured")
            except Exception as e:
                logger.debug("Unstructured partition failed for %s: %s", file_path, e)

        if not structured_text:
            return _fallback_load(file_path, pipeline=["fallback"])

        doc = Document(
            page_content=structured_text,
            metadata={
                "doc_type": "external",
                "source": file_path,
                "external_ext": os.path.splitext(file_path)[1].lower(),
                "parser_pipeline": pipeline,
                "mime_type": str(mime_type),
                "author": metadata.get("Author") or metadata.get("meta:author", ""),
                "elements_count": elements_count,
            },
        )
        return [doc]
    except Exception as e:
        logger.warning("Apache Tika parsing failed for %s (%s). Falling back.", file_path, e)
        return _fallback_load(file_path, pipeline=["fallback"])


def _ocr_pdf_via_pdf2image(file_path: str) -> str:
    """Rasterize each PDF page with pdf2image and OCR it directly with pytesseract.

    This exists because tesseract cannot read PDFs directly ("Pdf reading is not
    supported" -- it only accepts raster images), and UnstructuredFileLoader's
    hi_res strategy pulls in unstructured-inference (layout/table detection
    models) on top of pdf2image, which is a much heavier dependency chain that
    can fail independently even when pdf2image + poppler-utils are present.
    This path only needs pdf2image + poppler-utils + pytesseract, matching
    exactly what worked in manual testing (pdftoppm -> tesseract).
    """
    from pdf2image import convert_from_path
    import pytesseract

    pages = convert_from_path(file_path, dpi=300)
    page_texts = []
    for page_image in pages:
        text = pytesseract.image_to_string(page_image)
        if text.strip():
            page_texts.append(text.strip())
    return "\n\n".join(page_texts)


def _fallback_load(file_path: str, pipeline: Optional[List[str]] = None) -> List[Document]:
    """Fallback loader using UnstructuredFileLoader, PyPDF, or plain text."""
    if pipeline is None:
        pipeline = ["unstructured_fallback"]
    try:
        # Strategy forced to hi_res for OCR
        loaded = UnstructuredFileLoader(file_path, strategy="hi_res").load()
        for doc in loaded:
            if file_path.lower().endswith(".pdf"):
                doc.page_content = _clean_pdf_text(doc.page_content)
            doc.metadata["doc_type"] = "external"
            doc.metadata["external_ext"] = os.path.splitext(file_path)[1].lower()
            doc.metadata["source"] = file_path
            doc.metadata["parser_pipeline"] = pipeline
        return loaded
    except Exception as e:
        # This is the hi_res/UnstructuredFileLoader path failing. For PDFs this is
        # usually a missing OCR dependency (pdf2image, poppler-utils) rather than a
        # genuinely unreadable file, so it's logged at error level, not warning.
        logger.error(
            "hi_res OCR loader failed for %s: %s: %s. Falling back to pypdf/raw-text "
            "(these paths cannot OCR image-only PDFs, so check that pdf2image and "
            "poppler-utils are installed).",
            file_path, type(e).__name__, e,
        )
        if file_path.lower().endswith(".pdf"):
            # Try direct rasterize+OCR first. This is what actually works for
            # image-only PDFs (scanned docs, ID cards) -- pypdf below can only
            # extract an existing text layer, which these files don't have.
            try:
                ocr_text = _ocr_pdf_via_pdf2image(file_path)
                if ocr_text.strip() and not _is_garbage_text(ocr_text):
                    return [
                        Document(
                            page_content=_clean_pdf_text(ocr_text),
                            metadata={
                                "doc_type": "external",
                                "external_ext": ".pdf",
                                "source": file_path,
                                "parser_pipeline": ["pdf2image_tesseract_ocr"],
                            },
                        )
                    ]
                logger.warning(
                    "Direct pdf2image+tesseract OCR produced no usable text for %s",
                    file_path,
                )
            except ImportError as imp_err:
                logger.error(
                    "pdf2image/pytesseract not installed, cannot OCR %s: %s. "
                    "Run: pip install pdf2image pytesseract (and ensure "
                    "poppler-utils is installed at the OS level).",
                    file_path, imp_err,
                )
            except Exception as ocr_err:
                logger.error(
                    "Direct pdf2image+tesseract OCR failed for %s: %s: %s",
                    file_path, type(ocr_err).__name__, ocr_err,
                )

            try:
                import pypdf
                reader = pypdf.PdfReader(file_path)
                text = "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
                if text.strip():
                    return [
                        Document(
                            page_content=_clean_pdf_text(text.strip()),
                            metadata={
                                "doc_type": "external",
                                "external_ext": ".pdf",
                                "source": file_path,
                                "parser_pipeline": ["pypdf_fallback"],
                            },
                        )
                    ]
            except Exception as pdf_err:
                logger.warning("PyPDF fallback failed for %s: %s", file_path, pdf_err)

            # Direct OCR, hi_res OCR, and pypdf all failed. Do NOT fall through to
            # the raw-binary-as-text read below -- decoding a PDF's binary bytes as
            # UTF-8 with errors="ignore" always "succeeds" with a non-empty
            # string, but it's just object-stream/page-dictionary garbage. That
            # previously passed through as if it were valid content.
            logger.error(
                "All extraction methods failed for PDF %s (no text layer, hi_res OCR "
                "failed, direct pdf2image OCR failed, pypdf found nothing). "
                "Skipping rather than ingesting raw binary bytes as text.",
                file_path,
            )
            return []
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            if _is_garbage_text(content):
                logger.error(
                    "Raw-text fallback for %s produced unreadable/garbage content "
                    "(likely a binary file with no matching parser). Skipping.",
                    file_path,
                )
                return []
            return [
                Document(
                    page_content=content,
                    metadata={
                        "doc_type": "external",
                        "external_ext": os.path.splitext(file_path)[1].lower(),
                        "source": file_path,
                        "parser_pipeline": ["raw_text_fallback"],
                    },
                )
            ]
        except Exception as read_err:
            logger.error("All document loading failed for %s: %s", file_path, read_err)
            return []


def parse_image_with_ocr(file_path: str) -> List[Document]:
    """Parse image documents (.png, .jpg, etc.) using local Tesseract OCR."""
    import subprocess

    structured_text = ""
    pipeline = ["image_ocr"]
    try:
        res = subprocess.run(
            ["tesseract", file_path, "stdout"],
            capture_output=True,
            text=True,
            check=False,
        )
        if res.returncode == 0 and res.stdout.strip():
            structured_text = res.stdout.strip()
            pipeline.append("tesseract_cli")
    except Exception as e:
        logger.debug("Tesseract OCR CLI failed for %s: %s", file_path, e)

    if not structured_text and getattr(config, "INGESTION_USE_TIKA", True):
        try:
            from tika import parser as tika_parser

            parsed = tika_parser.from_file(file_path)
            raw = parsed.get("content") or ""
            if raw.strip():
                structured_text = raw.strip()
                pipeline.append("tika")
        except Exception:
            pass

    if not structured_text:
        return []

    return [
        Document(
            page_content=structured_text,
            metadata={
                "doc_type": "external",
                "source": file_path,
                "external_ext": os.path.splitext(file_path)[1].lower(),
                "parser_pipeline": pipeline,
                "mime_type": mimetypes.guess_type(file_path)[0] or "image/png",
            },
        )
    ]


def _discover_files(directory: str) -> List[str]:
    if not os.path.isdir(directory):
        return []

    discovered = set()
    for root, _, files in os.walk(directory):
        for file_name in files:
            if file_name.startswith("."):
                continue
            file_path = os.path.join(root, file_name)
            discovered.add(file_path)
    return sorted(list(discovered))


def _discover_external_files() -> List[str]:
    search_dir = getattr(config, "EXTERNAL_DOCS_PATH", getattr(config, "DATA_DIR", "./data"))
    return _discover_files(search_dir)


def _load_files_with_type(file_paths: List[str], doc_type: str = "document", source_folder: str = "data") -> List[Document]:
    documents: List[Document] = []
    for file_path in sorted(file_paths):
        ext = os.path.splitext(file_path)[1].lower()
        if ext in (".md", ".txt"):
            try:
                loader = TextLoader(file_path, encoding="utf-8")
                loaded = loader.load()
                for doc in loaded:
                    doc.metadata["doc_type"] = doc_type
                    doc.metadata["source_folder"] = source_folder
                    doc.metadata["source"] = file_path
                    doc.metadata.setdefault("parser_pipeline", ["text_loader"])
                documents.extend(loaded)
            except Exception as e:
                logger.warning("TextLoader failed for %s (%s), falling back to Tika/Unstructured", file_path, e)
                fallback = parse_non_pdf_with_tika(file_path)
                for doc in fallback:
                    doc.metadata["doc_type"] = doc_type
                    doc.metadata["source_folder"] = source_folder
                documents.extend(fallback)
        elif _is_pdf_file(file_path):
            loaded = parse_pdf_with_docling(file_path)
            for doc in loaded:
                doc.metadata["doc_type"] = doc_type
                doc.metadata["source_folder"] = source_folder
            documents.extend(loaded)
        elif ext in (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"):
            loaded = parse_image_with_ocr(file_path)
            for doc in loaded:
                doc.metadata["doc_type"] = doc_type
                doc.metadata["source_folder"] = source_folder
            documents.extend(loaded)
        else:
            loaded = parse_non_pdf_with_tika(file_path)
            for doc in loaded:
                doc.metadata["doc_type"] = doc_type
                doc.metadata["source_folder"] = source_folder
            documents.extend(loaded)
    return documents


def load_documents() -> List[Document]:
    data_dir = getattr(config, "DATA_DIR", "./data")
    all_files = _discover_files(data_dir)

    logger.info("Loading documents: %s files discovered in %s", len(all_files), data_dir)
    documents = _load_files_with_type(all_files, doc_type="document", source_folder="data")
    logger.info("Loaded %s total universal documents", len(documents))
    return documents