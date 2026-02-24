"""SC-Router HTTP Gateway (Starlette ASGI).

Optional dependency: ``pip install sc-router[gateway]``

Usage:
    from sc_router.gateway import create_app
    app = create_app("config.yaml")
    # Run with: uvicorn sc_router.gateway:app
"""

from .app import create_app

__all__ = ['create_app']
