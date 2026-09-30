"""FastAPI Modular APIRouter package for Engram AI Web & API routes (Issue #26 / Q-01)."""

from engramvault.web.routes.agent import build_agent_router
from engramvault.web.routes.curation import build_curation_router
from engramvault.web.routes.security import build_security_router, require_mutation_auth

__all__ = [
    "build_agent_router",
    "build_curation_router",
    "build_security_router",
    "require_mutation_auth",
]

