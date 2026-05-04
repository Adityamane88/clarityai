from fastapi import APIRouter

from app.api.routes.chat import router as chat_router
from app.api.routes.feedback import router as feedback_router
from app.api.routes.images import router as images_router
from app.api.routes.knowledge import router as knowledge_router
from app.api.routes.sessions import router as sessions_router

api_router = APIRouter()
api_router.include_router(sessions_router, prefix='/sessions', tags=['sessions'])
api_router.include_router(chat_router, prefix='/chat', tags=['chat'])
api_router.include_router(knowledge_router, prefix='/knowledge', tags=['knowledge'])
api_router.include_router(feedback_router, prefix='/feedback', tags=['feedback'])
api_router.include_router(images_router, prefix='/images', tags=['images'])
