from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.router import api_router
from app.config import get_settings
from app.db.session import Base, SessionLocal, engine
from app.services.retrieval import retrieval_index
from app.services.seeder import seed_sample_knowledge

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s | %(message)s')
logger = logging.getLogger('clarityai')

settings = get_settings()

# Where the built frontend lives (vite build output). Resolved relative to this file
# so it works regardless of where uvicorn is launched from.
FRONTEND_DIST = (Path(__file__).resolve().parents[2] / 'frontend' / 'dist').resolve()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        seeded = seed_sample_knowledge(db)
        if seeded:
            logger.info('Auto-seeded %d sample document(s) on first run.', seeded)
        retrieval_index.rebuild(db)
        logger.info(
            'ClarityAI ready. remote_llm=%s | dense_retrieval=%s | web_research=%s',
            settings.remote_llm_configured,
            settings.embedding_provider_configured,
            settings.web_research_configured,
        )
        if FRONTEND_DIST.is_dir():
            logger.info('Serving frontend from %s', FRONTEND_DIST)
        else:
            logger.warning(
                'Frontend build not found at %s. Run "npm run build" in the frontend folder, '
                'then refresh the browser. The API will still work on its own.',
                FRONTEND_DIST,
            )
        if not settings.remote_llm_configured:
            logger.warning(
                'No LLM configured. Set LLM_BASE_URL, LLM_API_KEY, and CHAT_MODEL in backend/.env '
                'for real model answers.'
            )
    finally:
        db.close()
    yield


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description='Production-ready conversational knowledge assistant with grounded answers, research routing, safety, and a modern UI.',
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


@app.get('/health')
def health() -> dict:
    return {
        'status': 'ok',
        'remote_llm_configured': settings.remote_llm_configured,
        'chat_model': settings.chat_model if settings.remote_llm_configured else None,
        'dense_retrieval_enabled': settings.embedding_provider_configured,
        'web_research_configured': settings.web_research_configured,
    }


# All API endpoints are mounted FIRST so they take priority over the SPA catch-all below.
app.include_router(api_router, prefix=settings.api_prefix)


# ----------------------------------------------------------------------------
# Frontend serving
# ----------------------------------------------------------------------------
# After the API routes are registered, mount the built frontend so that:
#   - /assets/* and other static files come from frontend/dist
#   - any other path returns index.html so the React SPA can handle routing
# If the build folder is missing, '/' returns a friendly JSON pointer instead of
# crashing the app.

if FRONTEND_DIST.is_dir():
    assets_dir = FRONTEND_DIST / 'assets'
    if assets_dir.is_dir():
        app.mount('/assets', StaticFiles(directory=str(assets_dir)), name='assets')

    NO_CACHE_HEADERS = {
        'Cache-Control': 'no-store, no-cache, must-revalidate, max-age=0',
        'Pragma': 'no-cache',
        'Expires': '0',
    }

    def _index_response() -> FileResponse:
        # The SPA shell must never be cached, otherwise a fresh build won't be picked
        # up by browsers that already have it. The hashed /assets/* files are fine to
        # cache and StaticFiles handles them.
        return FileResponse(FRONTEND_DIST / 'index.html', headers=NO_CACHE_HEADERS)

    @app.get('/', include_in_schema=False)
    def serve_root() -> FileResponse:
        return _index_response()

    @app.get('/{full_path:path}', include_in_schema=False)
    def serve_spa(full_path: str, request: Request):
        # Don't intercept api/health/docs/openapi - they're handled above; this is
        # only reached for paths that don't match any registered route.
        if full_path.startswith(('api/', 'health', 'docs', 'redoc', 'openapi.json')):
            return JSONResponse({'detail': 'Not Found'}, status_code=404)

        candidate = (FRONTEND_DIST / full_path).resolve()
        # Prevent path traversal: candidate must stay inside FRONTEND_DIST
        try:
            candidate.relative_to(FRONTEND_DIST)
        except ValueError:
            return JSONResponse({'detail': 'Not Found'}, status_code=404)

        if candidate.is_file():
            return FileResponse(candidate)
        return _index_response()
else:
    @app.get('/')
    def root_no_build() -> dict:
        return {
            'name': settings.app_name,
            'version': settings.app_version,
            'status': 'ok',
            'docs': '/docs',
            'note': 'Frontend build not found. Run "npm run build" in the frontend folder to enable the UI.',
        }
