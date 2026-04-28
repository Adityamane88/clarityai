from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import api_router
from app.config import get_settings
from app.db.session import Base, SessionLocal, engine
from app.services.retrieval import retrieval_index
from app.services.seeder import seed_sample_knowledge

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s | %(message)s')
logger = logging.getLogger('clarityai')

settings = get_settings()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Startup: create tables, seed sample knowledge if empty, build the retrieval index.
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
        if not settings.remote_llm_configured:
            logger.warning(
                'No LLM configured. Set LLM_BASE_URL, LLM_API_KEY, and CHAT_MODEL in backend/.env '
                'for real model answers. Without these, the app uses fallback evidence-only replies.'
            )
    finally:
        db.close()
    yield
    # Shutdown: nothing to clean up explicitly; SQLAlchemy engine handles connections.


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


@app.get('/')
def root() -> dict:
    return {
        'name': settings.app_name,
        'version': settings.app_version,
        'status': 'ok',
        'docs': '/docs',
    }


@app.get('/health')
def health() -> dict:
    return {
        'status': 'ok',
        'remote_llm_configured': settings.remote_llm_configured,
        'chat_model': settings.chat_model if settings.remote_llm_configured else None,
        'dense_retrieval_enabled': settings.embedding_provider_configured,
        'web_research_configured': settings.web_research_configured,
    }


app.include_router(api_router, prefix=settings.api_prefix)
