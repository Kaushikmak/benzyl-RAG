import argparse
import os
import sys
import warnings
import logging

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Universal RAG Application")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Indexing command
    subparsers.add_parser("index", help="Run document indexing")

    # Interactive CLI command
    subparsers.add_parser("cli", help="Ask queries on the CLI")

    # One-shot query command
    parser_query = subparsers.add_parser("query", help="Run a one-shot query")
    parser_query.add_argument("-q", "--query", type=str, required=True, help="Query text")
    parser_query.add_argument("-k", "--top-k", type=int, default=5, help="Number of results to return")

    # Test command
    subparsers.add_parser("test", help="Run unit tests across all modules")

    args = parser.parse_args()

    if args.command == "index":
        from app.progress import AnimatedProgress
        with AnimatedProgress("Indexing documents into Qdrant & building relationship graphs", start_pct=10, target_pct=98) as prog:
            with open(os.devnull, "w") as devnull:
                old_stderr = sys.stderr
                sys.stderr = devnull
                try:
                    from indexing.build_indexes import main as index_main
                    index_main()
                finally:
                    sys.stderr = old_stderr
                    
    elif args.command == "cli":
        from app.progress import AnimatedProgress
        with AnimatedProgress("Loading RAG models & Qdrant vector indices", start_pct=15, target_pct=95) as prog:
            with open(os.devnull, "w") as devnull:
                old_stderr = sys.stderr
                sys.stderr = devnull
                try:
                    from app.rag import main as cli_main
                finally:
                    sys.stderr = old_stderr
        cli_main()
        
    elif args.command == "query":
        from app.progress import AnimatedProgress
        with AnimatedProgress("Executing similarity search", start_pct=15, target_pct=95) as prog:
            with open(os.devnull, "w") as devnull:
                old_stderr = sys.stderr
                sys.stderr = devnull
                try:
                    from app.rag import ImprovedRAG
                    rag = ImprovedRAG()
                    cands, _ = rag.retrieve_candidates(args.query)
                    cands = cands[:args.top_k]
                finally:
                    sys.stderr = old_stderr
                    
        logging.disable(logging.NOTSET)

        print(f"\n==================================================================")
        print(f" RETRIEVAL FOR QUERY: '{args.query}'")
        print(f"==================================================================")
        for i, c in enumerate(cands, 1):
            print(f"  [{i}] ID: {c.doc_id}")
            print(f"      Source: {c.source}")
            print(f"      Snippet: {c.content[:100]}...")

        try:
            result = rag.answer(args.query)
            answer_text = result.get("answer", str(result)) if isinstance(result, dict) else str(result)
            print("=== BEGIN ANSWER ===")
            print(answer_text.strip())
            print("=== END ANSWER ===")
        except Exception as e:
            print("=== BEGIN ANSWER ===")
            print(f"Could not generate answer via LLM ({e}).")
            print("=== END ANSWER ===")
        print(f"==================================================================\n")
        
    elif args.command == "test":
        import unittest
        suite = unittest.defaultTestLoader.discover("tests")
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()