import argparse
import sys
import logging

def main() -> None:
    """CLI entrypoint for Agentic-RAG."""
    parser = argparse.ArgumentParser(
        prog="agenticrag",
        description="CLI for Agentic Customer Support RAG System"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Command: serve
    serve_parser = subparsers.add_parser("serve", help="Start the FastAPI backend server")
    serve_parser.add_argument("--host", default="0.0.0.0", help="Host address (default: 0.0.0.0)")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port to bind (default: 8000)")
    serve_parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")

    # Command: ingest
    subparsers.add_parser("ingest", help="Run the knowledge base ingestion pipeline into Pinecone")

    args = parser.parse_args()

    # Default to serve if no command provided
    if not args.command or args.command == "serve":
        import uvicorn
        host = getattr(args, "host", "0.0.0.0")
        port = getattr(args, "port", 8000)
        reload = getattr(args, "reload", False)
        print(f"Starting Agentic-RAG server on http://{host}:{port} ...")
        uvicorn.run("agenticrag.api:app", host=host, port=port, reload=reload)
    elif args.command == "ingest":
        logging.basicConfig(level=logging.INFO)
        from agenticrag.ingest import run_ingestion
        run_ingestion()
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
