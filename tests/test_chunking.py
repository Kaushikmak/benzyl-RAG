import unittest
from langchain_core.documents import Document

from indexing.chunking import (
    HeadingStackTracker,
    extract_keywords_rake,
    extract_chunk_metadata,
    process_table_block,
    chunk_documents,
)


class TestIntelligentChunking(unittest.TestCase):
    def test_heading_stack_tracker(self):
        tracker = HeadingStackTracker()
        tracker.process_line("# Architecture")
        self.assertEqual(tracker.get_breadcrumb(), "Architecture")

        tracker.process_line("## Databases")
        self.assertEqual(tracker.get_breadcrumb(), "Architecture > Databases")

        tracker.process_line("### PGVector")
        self.assertEqual(tracker.get_breadcrumb(), "Architecture > Databases > PGVector")

        # Higher level heading should pop deeper levels
        tracker.process_line("## Networking")
        self.assertEqual(tracker.get_breadcrumb(), "Architecture > Networking")

    def test_extract_keywords_rake_stop_words(self):
        text = "The PGVector database integration provides high performance vector similarity search."
        keywords = extract_keywords_rake(text, top_k=3)
        # Should filter out stop words like 'the', 'provides' etc or return high-value terms
        self.assertIn("pgvector", keywords)
        self.assertIn("database", keywords)

    def test_process_table_block_atomic_small(self):
        table_lines = [
            "| Col1 | Col2 |",
            "| ---- | ---- |",
            "| A    | B    |",
        ]
        chunks = process_table_block(table_lines, breadcrumb="Test Section", max_chars=4000)
        self.assertEqual(len(chunks), 1)
        content, is_table = chunks[0]
        self.assertTrue(is_table)
        self.assertIn("[Heading Context: Test Section]", content)
        self.assertIn("| Col1 | Col2 |", content)

    def test_process_table_block_massive_split_with_headers(self):
        header_lines = [
            "| ID | Description | Amount |",
            "| -- | ----------- | ------ |",
        ]
        data_rows = [f"| {i} | Long ledger entry number {i} with financial description | $100 |" for i in range(50)]
        all_lines = header_lines + data_rows
        # Set max_chars low enough to force split
        chunks = process_table_block(all_lines, breadcrumb="Financials", max_chars=300)
        self.assertGreater(len(chunks), 1)
        for content, is_table in chunks:
            self.assertTrue(is_table)
            self.assertIn("[Heading Context: Financials]", content)
            self.assertIn("| ID | Description | Amount |", content)

    def test_chunk_documents_end_to_end(self):
        markdown_text = """---
title: System Overview
---

# Architecture Overview

This section explains the overall system architecture.

## Database Layer

The database layer utilizes embedded Qdrant and SQLite for local storage.

| Table | Engine | Purpose |
| ----- | ------ | ------- |
| chunks | Qdrant | Dense embeddings |
| log | JSON | Evaluation history |
"""
        doc = Document(page_content=markdown_text, metadata={"source": "data/system.md", "doc_type": "document"})
        chunks = chunk_documents([doc])
        self.assertGreaterEqual(len(chunks), 2)

        # Check metadata enrichment
        for chunk in chunks:
            self.assertIn("heading_breadcrumb", chunk.metadata)
            self.assertIn("is_table", chunk.metadata)
            self.assertIn("summary", chunk.metadata)
            self.assertIn("keywords", chunk.metadata)
            self.assertIn("hypothetical_questions", chunk.metadata)
            self.assertEqual(chunk.metadata["note_name"], "system")
            self.assertEqual(chunk.metadata["filename"], "system.md")

        # Verify table chunk specifically
        table_chunks = [c for c in chunks if c.metadata.get("is_table")]
        self.assertEqual(len(table_chunks), 1)
        self.assertEqual(table_chunks[0].metadata["heading_breadcrumb"], "Architecture Overview > Database Layer")


if __name__ == "__main__":
    unittest.main()
