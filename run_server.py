#!/usr/bin/env python
"""Script to run the Medical QA Chat API server."""

import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def main():
    """Run the FastAPI server."""
    try:
        import uvicorn
    except ImportError:
        print("Error: uvicorn not installed.")
        print("Please run: pip install fastapi uvicorn[standard] python-multipart aiofiles")
        sys.exit(1)
    
    print("=" * 60)
    print("Starting Medical QA Chat Server")
    print("=" * 60)
    print("\nMake sure you have installed the required dependencies:")
    print("  pip install -r requirements.txt")
    print("\nAPI will be available at:")
    print("  - Chat UI: http://localhost:8000")
    print("  - API Docs: http://localhost:8000/docs")
    print("  - ReDoc: http://localhost:8000/redoc")
    print("=" * 60)
    
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Enable auto-reload during development
        log_level="info"
    )


if __name__ == "__main__":
    main()

