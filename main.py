import argparse
import sys
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Obsidian RAG Application")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Indexing command
    parser_index = subparsers.add_parser("index", help="Run document indexing")

    # CLI command
    parser_cli = subparsers.add_parser("cli", help="Ask queries on the CLI")


    args = parser.parse_args()

    if args.command == "index":
        logger.info("Starting indexing...")
        from indexing.build_indexes import main as index_main
        index_main()
    elif args.command == "cli":
        logger.info("Starting CLI mode...")
        from app.rag import main as cli_main
        cli_main()


if __name__ == "__main__":
    main()
