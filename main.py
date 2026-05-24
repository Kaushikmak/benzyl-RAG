import argparse
import sys
import subprocess
import logging
from pathlib import Path

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

    # Webapp command
    parser_web = subparsers.add_parser("web", help="Start the FastAPI backend and Next.js frontend")

    args = parser.parse_args()

    if args.command == "index":
        logger.info("Starting indexing...")
        from indexing.build_indexes import main as index_main
        index_main()
    elif args.command == "cli":
        logger.info("Starting CLI mode...")
        from app.rag import main as cli_main
        cli_main()
    elif args.command == "web":
        logger.info("Starting Web App (Backend & Frontend)...")
        
        try:
            web_dir = Path("web")
            node_modules_dir = web_dir / "node_modules"
            if not node_modules_dir.exists():
                logger.info("Installing Next.js dependencies...")
                subprocess.run(
                    ["npm", "install"],
                    cwd=str(web_dir),
                    check=True,
                )

            with open("backend.log", "w") as blog, open("frontend.log", "w") as flog:
                logger.info("Starting FastAPI backend...")
                backend = subprocess.Popen([sys.executable, "api.py"], stdout=blog, stderr=subprocess.STDOUT)
                
                logger.info("Starting Next.js frontend...")
                frontend = subprocess.Popen(
                    ["npm", "run", "dev"],
                    cwd="web",
                    stdout=flog,
                    stderr=subprocess.STDOUT
                )
                
                print("\n========================================")
                print("Both services are running!")
                print("========================================\n")
                print("Access the web interface at: http://localhost:3000")
                print("API documentation at: http://localhost:8000/docs\n")
                print("Logs:")
                print("  Backend: backend.log")
                print("  Frontend: frontend.log\n")
                print("Press Ctrl+C to stop both services\n")
                
                backend.wait()
                frontend.wait()
        except KeyboardInterrupt:
            print("\nShutting down services...")
            backend.terminate()
            frontend.terminate()
            sys.exit(0)

if __name__ == "__main__":
    main()
