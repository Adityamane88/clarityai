from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import inspect, text

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


def _ensure_schema_columns() -> None:
    """
    Lightweight forward-compat migration so existing SQLite databases pick up
    columns we added in the Elite version (e.g. images_json on chat_messages).

    `Base.metadata.create_all` won't add new columns to a pre-existing table,
    so we ALTER TABLE manually for the few additions we know about. This is
    intentionally simple - for serious schema work, plug Alembic in.
    """
    inspector = inspect(engine)
    if not inspector.has_table('chat_messages'):
        return  # First-run: create_all() will handle it.

    existing = {col['name'] for col in inspector.get_columns('chat_messages')}
    statements: list[str] = []
    if 'images_json' not in existing:
        statements.append("ALTER TABLE chat_messages ADD COLUMN images_json JSON")

    if not statements:
        return

    with engine.begin() as conn:
        for stmt in statements:
            try:
                conn.execute(text(stmt))
                logger.info('Schema migration applied: %s', stmt)
            except Exception as exc:  # noqa: BLE001
                logger.warning('Schema migration skipped (%s): %s', stmt, exc)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    Base.metadata.create_all(bind=engine)
    _ensure_schema_columns()
    db = SessionLocal()
    try:
        seeded = seed_sample_knowledge(db)
        if seeded:
            logger.info('Auto-seeded %d sample document(s) on first run.', seeded)
        retrieval_index.rebuild(db)
        logger.info(
            'ClarityAI ready. easy=%s (%s) | heavy=%s (%s) | dense_retrieval=%s | web_research=%s',
            settings.remote_llm_configured,
            settings.chat_model or '-',
            settings.heavy_llm_configured,
            settings.heavy_chat_model or '-',
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
    description=(
        'ClarityAI Elite - production-ready conversational knowledge assistant '
        'with grounded answers, research routing, image search, safety, and a modern UI.'
    ),
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
        'easy_provider_configured': settings.remote_llm_configured,
        'easy_chat_model': settings.chat_model if settings.remote_llm_configured else None,
        'heavy_provider_configured': settings.heavy_llm_configured,
        'heavy_chat_model': settings.heavy_chat_model if settings.heavy_llm_configured else None,
        # Legacy keys kept for any frontend code that already reads them.
        'remote_llm_configured': settings.remote_llm_configured,
        'chat_model': settings.chat_model if settings.remote_llm_configured else None,
        'dense_retrieval_enabled': settings.embedding_provider_configured,
        'web_research_configured': settings.web_research_configured,
        'image_search_enabled': True,
    }


# All API endpoints are mounted FIRST so they take priority over the SPA catch-all below.
app.include_router(api_router, prefix=settings.api_prefix)


# ----------------------------------------------------------------------------
# Frontend serving (unchanged from previous version)
# ----------------------------------------------------------------------------

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
        return FileResponse(FRONTEND_DIST / 'index.html', headers=NO_CACHE_HEADERS)

    @app.get('/', include_in_schema=False)
    def serve_root() -> FileResponse:
        return _index_response()

    @app.get('/{full_path:path}', include_in_schema=False)
    def serve_spa(full_path: str, request: Request):
        if full_path.startswith(('api/', 'health', 'docs', 'redoc', 'openapi.json')):
            return JSONResponse({'detail': 'Not Found'}, status_code=404)

        candidate = (FRONTEND_DIST / full_path).resolve()
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