"""Pre-flight script to download ChromaDB ONNX embedding model.

Forces ChromaDB to download and cache the all-MiniLM-L6-v2 ONNX model
locally so the RAG engine can initialize without network access at runtime.

Usage:
    python -m recovery_agent.scripts.download_models

Exit codes:
    0 — Model already cached or download successful
    1 — Download failed (network error, timeout, etc.)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

MODEL_CACHE_DIR = Path.home() / ".cache" / "chroma" / "onnx_models" / "all-MiniLM-L6-v2"


def is_model_cached() -> bool:
    """Check if the ONNX model is already cached locally."""
    if not MODEL_CACHE_DIR.exists():
        return False
    onnx_files = list(MODEL_CACHE_DIR.glob("*.onnx"))
    onnx_subdir = list((MODEL_CACHE_DIR / "onnx").glob("*.onnx")) if (MODEL_CACHE_DIR / "onnx").exists() else []
    return len(onnx_files) > 0 or len(onnx_subdir) > 0


def download_model() -> bool:
    """Download the ONNX embedding model by invoking ChromaDB's default embedding function.

    Returns True on success, False on failure.
    """
    try:
        from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
    except ImportError:
        print("[download_models] ERROR: chromadb not installed. Install with: pip install chromadb", file=sys.stderr)
        return False

    print("[download_models] Initializing ChromaDB embedding function (will download ONNX model if not cached)...")

    try:
        ef = ONNXMiniLM_L6_V2()
        # Force the model download by embedding a dummy string
        result = ef(["AutoRecover pre-flight check"])
        if result and len(result) > 0:
            print(f"[download_models] Model downloaded and cached at: {MODEL_CACHE_DIR}")
            return True
        else:
            print("[download_models] ERROR: Embedding function returned empty result", file=sys.stderr)
            return False
    except Exception as e:
        print(f"[download_models] ERROR: Failed to download embedding model: {e}", file=sys.stderr)
        return False


def main() -> int:
    """Main entry point. Returns exit code 0 on success, 1 on failure."""
    if is_model_cached():
        print(f"[download_models] Model already cached at: {MODEL_CACHE_DIR}")
        return 0

    print("[download_models] ONNX embedding model not found. Downloading...")
    start = time.time()

    success = download_model()

    elapsed = time.time() - start
    if success:
        print(f"[download_models] Download complete in {elapsed:.1f}s")
        return 0
    else:
        print(f"[download_models] Download failed after {elapsed:.1f}s", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
