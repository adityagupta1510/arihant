#!/usr/bin/env python3
"""
ARIHANT SOC - Quick Start Script
================================
Run this script to start the FastAPI backend server

Usage:
    python run.py
    python run.py --port 8080
    python run.py --reload
"""

import argparse
import uvicorn


def main():
    parser = argparse.ArgumentParser(description="ARIHANT SOC API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    
    args = parser.parse_args()
    
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║     █████╗ ██████╗ ██╗██╗  ██╗ █████╗ ███╗   ██╗████████╗ ║
    ║    ██╔══██╗██╔══██╗██║██║  ██║██╔══██╗████╗  ██║╚══██╔══╝ ║
    ║    ███████║██████╔╝██║███████║███████║██╔██╗ ██║   ██║    ║
    ║    ██╔══██║██╔══██╗██║██╔══██║██╔══██║██║╚██╗██║   ██║    ║
    ║    ██║  ██║██║  ██║██║██║  ██║██║  ██║██║ ╚████║   ██║    ║
    ║    ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝   ╚═╝    ║
    ║                                                           ║
    ║           AI-Driven Security Operations Center            ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    print(f"  Starting server at http://{args.host}:{args.port}")
    print(f"  API Docs: http://localhost:{args.port}/docs")
    print(f"  ReDoc: http://localhost:{args.port}/redoc")
    print(f"  WebSocket: ws://localhost:{args.port}/ws/alerts")
    print()
    
    uvicorn.run(
        "main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1,
        log_level="info"
    )


if __name__ == "__main__":
    main()
