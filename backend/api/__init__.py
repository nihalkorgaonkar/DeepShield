"""
DeepShield API Routers
"""

from .routes_health import router as health_router
from .routes_models import router as models_router
from .routes_detect import router as detect_router
from .routes_explain import router as explain_router
from .routes_robustness import router as robustness_router
from .routes_datasets import router as datasets_router
from .routes_evaluate import router as evaluate_router
from .routes_reports import router as reports_router
from .routes_samples import router as samples_router

__all__ = [
    "health_router",
    "models_router",
    "detect_router",
    "explain_router",
    "robustness_router",
    "datasets_router",
    "evaluate_router",
    "reports_router",
    "samples_router",
]
