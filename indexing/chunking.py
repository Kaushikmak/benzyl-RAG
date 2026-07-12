import os
import re
import logging
from collections import Counter
from typing import List, Dict, Any, Tuple

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app import config

logger = logging.getLogger(__name__)

STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at",
    "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can", "did", "do",
    "does", "doing", "don", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have",
    "having", "he", "her", "here", "hers", "herself", "him", "himself", "his", "how", "i", "if", "in", "into",
    "is", "it", "its", "itself", "just", "me", "more", "most", "my", "myself", "no", "nor", "not", "now", "of",
    "off", "on", "once", "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own", "s", "same",
    "she", "should", "so", "some", "such", "t", "than", "that", "the", "their", "theirs", "them", "themselves",
    "then", "there", "these", "they", "this", "those", "through", "to", "too", "under", "until", "up", "very",
    "was", "we", "were", "what", "when", "where", "which", "while", "who", "whom", "why", "will", "with",
    "you", "your", "yours", "yourself", "yourselves"
}


class HeadingStackTracker:
    """Tracks Markdown headings hierarchy (# to ######) as a true stack."""
    def __init__(self):
        self.stack: Dict[int, str] = {}

    def process_line(self, line: str) -> bool:
        match = re.match(r"^(#{1,6})\s+(.+)$", line.strip())
        if not match:
            return False
        level = len(match.group(1))
        title = match.group(2).strip()
        self.stack[level] = title
        # Remove any deeper headings
        for l in list(self.stack.keys()):
            if l > level:
                del self.stack[l]
        return True

    def get_breadcrumb(self) -> str:
        if not self.stack:
            return ""
        sorted_levels = sorted(self.stack.keys())
        return " > ".join(self.stack[l] for l in sorted_levels)


def extract_keywords_rake(text: str, top_k: int = 5) -> List[str]:
    """Fast local RAKE/TF-IDF word scoring filtered against stop words."""
    clean_text = re.sub(r"[^a-zA-Z0-9\s-]", " ", text).lower()
    words = [w.strip() for w in clean_text.split() if len(w.strip()) >= 3 and w.strip() not in STOP_WORDS]
    if not words:
        return []
    counts = Counter(words)
    # Return top_k most frequent domain terms
    return [word for word, _ in counts.most_common(top_k)]


def extract_chunk_metadata(chunk_text: str, breadcrumb: str) -> Dict[str, Any]:
    """Precompute summary, keywords, and hypothetical questions for each chunk."""
    keywords = extract_keywords_rake(chunk_text, top_k=5)

    # Clean leading sentences for summary
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", chunk_text.strip()) if s.strip() and not s.strip().startswith("[Heading Context")]
    summary_body = " ".join(sentences[:2]) if sentences else chunk_text[:150]
    if breadcrumb:
        summary = f"[{breadcrumb}] {summary_body}"
    else:
        summary = summary_body

    # Construct hypothetical questions
    topic = breadcrumb.split(" > ")[-1] if breadcrumb else (keywords[0] if keywords else "this document")
    questions = [
        f"What details are explained about {topic}?",
    ]
    if len(keywords) > 1:
        questions.append(f"How does {topic} relate to {keywords[1]}?")
    elif breadcrumb:
        questions.append(f"What is discussed under {breadcrumb}?")

    return {
        "summary": summary[:300],
        "keywords": keywords,
        "hypothetical_questions": questions,
    }


def format_chunk_content(raw_text: str, breadcrumb: str) -> str:
    """Prefix heading breadcrumb to chunk text if present."""
    if breadcrumb and not raw_text.startswith("[Heading Context:"):
        return f"[Heading Context: {breadcrumb}]\n\n{raw_text.strip()}"
    return raw_text.strip()


def process_table_block(table_lines: List[str], breadcrumb: str, max_chars: int = 4000) -> List[Tuple[str, bool]]:
    """Process table block atomically or split row-by-row with column header prepending if > max_chars."""
    if not table_lines:
        return []
    table_text = "\n".join(table_lines).strip()
    if len(table_text) <= max_chars:
        content = format_chunk_content(table_text, breadcrumb)
        return [(content, True)]

    # Massive table: split row-by-row prepending header rows
    header_lines = []
    data_lines = []
    header_found_sep = False
    for line in table_lines:
        if not header_found_sep:
            header_lines.append(line)
            if re.match(r"^\|\s*[-:]+\s*\|", line.strip()):
                header_found_sep = True
        else:
            data_lines.append(line)

    if not header_found_sep:
        # Fallback if no markdown separator line found: use first line as header
        header_lines = table_lines[:1]
        data_lines = table_lines[1:]

    header_text = "\n".join(header_lines)
    sub_chunks: List[Tuple[str, bool]] = []
    current_rows: List[str] = []
    current_len = len(header_text)

    for row in data_lines:
        row_len = len(row) + 1
        if current_rows and (current_len + row_len > max_chars):
            sub_table = header_text + "\n" + "\n".join(current_rows)
            sub_chunks.append((format_chunk_content(sub_table, breadcrumb), True))
            current_rows = [row]
            current_len = len(header_text) + row_len
        else:
            current_rows.append(row)
            current_len += row_len

    if current_rows:
        sub_table = header_text + "\n" + "\n".join(current_rows)
        sub_chunks.append((format_chunk_content(sub_table, breadcrumb), True))

    return sub_chunks


def process_narrative_block(text: str, breadcrumb: str, chunk_size: int, chunk_overlap: int) -> List[Tuple[str, bool]]:
    """Context-aware chunker that respects sentences and words, falling back gracefully."""
    clean_text = text.strip()
    if not clean_text:
        return []

    # Use LangChain's recursive splitter to respect semantic boundaries
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "?", "!", " ", ""],
        length_function=len,
    )
    
    raw_chunks = splitter.split_text(clean_text)
    
    chunks: List[Tuple[str, bool]] = []
    for chunk in raw_chunks:
        # Prepend the heading breadcrumb to each semantic chunk
        formatted_chunk = format_chunk_content(chunk, breadcrumb)
        chunks.append((formatted_chunk, False))

    return chunks


def chunk_documents(documents: List[Document]) -> List[Document]:
    logger.info("Chunking documents using Intelligent Chunking & Metadata Enrichment pipeline")
    max_chars = getattr(config, "TABLE_MAX_CHARS", 4000)
    chunk_size = getattr(config, "CHUNK_SIZE", 600)
    chunk_overlap = getattr(config, "CHUNK_OVERLAP", 100)

    all_chunks: List[Document] = []

    for doc in documents:
        base_meta = dict(doc.metadata)
        source = base_meta.get("source", "")
        filename = os.path.basename(source) if source else "unknown.md"
        doc_name = os.path.splitext(filename)[0]
        base_meta["doc_name"] = doc_name
        base_meta["note_name"] = doc_name
        base_meta["filename"] = filename

        lines = doc.page_content.splitlines()
        tracker = HeadingStackTracker()
        narrative_buffer: List[str] = []
        table_buffer: List[str] = []
        in_table = False

        doc_chunks: List[Tuple[str, bool, str]] = []  # (content, is_table, breadcrumb)

        def flush_narrative():
            nonlocal narrative_buffer
            if narrative_buffer:
                txt = "\n".join(narrative_buffer).strip()
                if txt:
                    for c_txt, is_tbl in process_narrative_block(txt, tracker.get_breadcrumb(), chunk_size, chunk_overlap):
                        doc_chunks.append((c_txt, is_tbl, tracker.get_breadcrumb()))
                narrative_buffer = []

        def flush_table():
            nonlocal table_buffer, in_table
            if table_buffer:
                for c_txt, is_tbl in process_table_block(table_buffer, tracker.get_breadcrumb(), max_chars=max_chars):
                    doc_chunks.append((c_txt, is_tbl, tracker.get_breadcrumb()))
                table_buffer = []
            in_table = False

        for line in lines:
            stripped = line.strip()
            # Check for heading
            if re.match(r"^#{1,6}\s+.+", stripped):
                flush_narrative()
                flush_table()
                tracker.process_line(stripped)
                narrative_buffer.append(line)
                continue

            # Check for markdown table row
            is_table_row = stripped.startswith("|") and stripped.endswith("|") and len(stripped) >= 2
            if is_table_row:
                if not in_table:
                    flush_narrative()
                    in_table = True
                table_buffer.append(line)
            else:
                if in_table:
                    flush_table()
                narrative_buffer.append(line)

        flush_narrative()
        flush_table()

        for content, is_tbl, breadcrumb in doc_chunks:
            meta = dict(base_meta)
            meta["heading_breadcrumb"] = breadcrumb
            meta["is_table"] = is_tbl
            enrichment = extract_chunk_metadata(content, breadcrumb)
            meta.update(enrichment)
            all_chunks.append(Document(page_content=content, metadata=meta))

    logger.info(f"Created {len(all_chunks)} chunks")
    if all_chunks:
        chunks_per_note = Counter(chunk.metadata["note_name"] for chunk in all_chunks)
        logger.info("Chunk statistics:")
        logger.info(f"Min chunks/note: {min(chunks_per_note.values())}")
        logger.info(f"Max chunks/note: {max(chunks_per_note.values())}")
        logger.info(
            f"Average chunks/note: "
            f"{sum(chunks_per_note.values()) / len(chunks_per_note):.2f}"
        )
    return all_chunks