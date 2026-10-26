"""
DeepShield FastAPI Application
Main server entry point connecting existing PyTorch checkpoints, preprocessing pipelines,
and evaluation suites to the Next.js frontend.
"""

from contextlib import asynccontextmanager
import logging
import sys
from pathlib import Path

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.api import (
    datasets_router,
    detect_router,
    evaluate_router,
    explain_router,
    health_router,
    models_router,
    reports_router,
    robustness_router,
    samples_router,
)
from backend.config import CORS_ORIGINS, HOST, PORT
from backend.services.model_service import ModelService

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("deepshield.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Eagerly initializes model weights into GPU memory on startup."""
    logger.info("DeepShield Backend starting up...")
    model_service = ModelService()
    logger.info(
        f"Preloaded models on device '{model_service.device}': {list(model_service.models.keys())}"
    )
    yield
    logger.info("DeepShield Backend shutting down...")


app = FastAPI(
    title="DeepShield — Audio Deepfake Detection API",
    description=(
        "Production-grade cybersecurity research API comparing Vision Transformer (ViT-B/16) "
        "and CNN Baseline (EfficientNet-B0) on ASVspoof benchmarks under strict academic integrity protocols."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS + ["*"],  # Allow all localhost development origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Routers under /api
app.include_router(health_router, prefix="/api")
app.include_router(models_router, prefix="/api")
app.include_router(detect_router, prefix="/api")
app.include_router(explain_router, prefix="/api")
app.include_router(robustness_router, prefix="/api")
app.include_router(datasets_router, prefix="/api")
app.include_router(evaluate_router, prefix="/api")
app.include_router(reports_router, prefix="/api")
app.include_router(samples_router, prefix="/api")


@app.get("/")
def root():
    return {
        "service": "DeepShield AI Detection Backend",
        "version": "2.0.0",
        "status": "online",
        "docs_url": "/docs",
        "api_endpoints": {
            "health": "/api/health",
            "models": "/api/models",
            "metrics_summary": "/api/metrics/summary",
            "detect": "/api/detect",
            "detect_compare": "/api/detect/compare",
            "explain": "/api/explain",
            "robustness": "/api/robustness/test",
            "datasets": "/api/datasets",
            "evaluate": "/api/evaluate",
            "reports": "/api/reports",
            "samples": "/api/samples",
        },
    }


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception on {request.method} {request.url}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "InternalServerError", "message": str(exc)},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host=HOST, port=PORT, reload=False)
