"""
main.py  —  Project root launcher
Smart AI Diet & Meal Optimization System

Usage (from project root):
    python main.py          # starts API server on port 8000
    python main.py api      # same
    python main.py train    # re-trains all models
"""

import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).parent.resolve()


def run_api():
    """Launch the FastAPI server via uvicorn."""
    import uvicorn
    # Add project root to path so all imports resolve
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )


def run_train():
    """Re-train all neural network models."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from models.neural_networks import run_model_training
    run_model_training()


if __name__ == "__main__":
    cmd = sys.argv[1].lower() if len(sys.argv) > 1 else "api"

    if cmd in ("api", "server", "run"):
        run_api()
    elif cmd in ("train", "models"):
        run_train()
    else:
        print(f"Unknown command: {cmd}")
        print("Usage: python main.py [api|train]")
        sys.exit(1)
