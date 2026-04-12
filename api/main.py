from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import sys

ROOT = Path(__file__).parent.parent

app = FastAPI()

# Serve static files
frontend_path = ROOT / "frontend"
if frontend_path.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")


@app.get("/")
def serve_home():
    return FileResponse(frontend_path / "index.html")


@app.get("/app")
def serve_app():
    return FileResponse(frontend_path / "index.html")


@app.get("/health")
def health():
    return {"status": "ok"}