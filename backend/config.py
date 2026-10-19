"""Shared configuration. Set DEMO_MODE=False when the RAG backend is ready."""

from pathlib import Path

# Test the UI flow without loading documents, embedding or calling Ollama.
DEMO_MODE = False

# Absolute uploads location, independent of the current working directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
UPLOAD_ROOT = PROJECT_ROOT / "uploads"
ALLOWED_EXTENSIONS = {"pdf", "txt", "doc", "docx", "png", "jpg", "jpeg"}
