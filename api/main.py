"""FastAPI application for Medical QA Chat."""

import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from api.routes import chat_router
from utils.config import Config

# Create FastAPI app
app = FastAPI(
    title="Medical QA Chat API",
    description="Multi-agent Medical Question Answering System with Chat Interface",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(chat_router)

# Static files directories
static_dir = os.path.join(project_root, "static")
uploads_dir = os.path.join(project_root, "uploads")

# Mount static files
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

if os.path.exists(uploads_dir):
    app.mount("/uploads", StaticFiles(directory=uploads_dir), name="uploads")


@app.get("/")
async def root():
    """Serve the main chat interface."""
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Medical QA Chat API", "docs": "/docs"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": "1.0.0"
    }


@app.get("/config/status")
async def config_status():
    """Check configuration status."""
    try:
        Config.validate()
        return {
            "status": "ok",
            "model": Config.GEMINI_MODEL,
            "medprompt_enabled": Config.ENABLE_FEW_SHOT or Config.ENABLE_COT or Config.ENABLE_ENSEMBLE,
            "reflexion_enabled": Config.ENABLE_REFLEXION
        }
    except ValueError as e:
        return {
            "status": "error",
            "error": str(e)
        }


@app.on_event("startup")
async def startup_event():
    """Startup tasks."""
    print("=" * 60)
    print("Medical QA Chat API Starting...")
    print("=" * 60)
    
    # Validate configuration
    try:
        Config.validate()
        print("✓ Configuration validated")
        print(f"  - Model: {Config.GEMINI_MODEL}")
        print(f"  - Medprompt: Few-shot={Config.ENABLE_FEW_SHOT}, CoT={Config.ENABLE_COT}, Ensemble={Config.ENABLE_ENSEMBLE}")
        print(f"  - Reflexion: {Config.ENABLE_REFLEXION}")
    except ValueError as e:
        print(f"⚠ Configuration warning: {e}")
        print("  Some features may not work without proper API keys.")
    
    print("=" * 60)
    print("API ready at http://localhost:8000")
    print("Docs at http://localhost:8000/docs")
    print("=" * 60)


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    print("Medical QA Chat API shutting down...")
    
    # Cleanup sessions and uploaded images
    from api.session_store import session_store
    session_store.cleanup_all_sessions()
    session_store.cleanup_all_uploads()
    
    print("Cleanup completed.")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

