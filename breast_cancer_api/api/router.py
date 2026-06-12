from fastapi import APIRouter

from .routes.health import router as health_router
from .routes.predict import router as predict_router
from .routes.recommend import router as recommend_router
from .routes.explain import router as explain_router
from .routes.disclaimer import router as disclaimer_router
from .routes.model_info import router as model_info_router

api_router = APIRouter()

api_router.include_router(health_router)
api_router.include_router(predict_router)
api_router.include_router(recommend_router)
api_router.include_router(explain_router)
api_router.include_router(disclaimer_router)
api_router.include_router(model_info_router)
