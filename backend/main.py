from contextlib import asynccontextmanager

from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from backend.core import auth
from backend.routes import views
from backend.apis.llama import ModelManager
from backend.apis.config import MllamaConfig

@asynccontextmanager
async def lifespan(app: FastAPI):
    llama_model = ModelManager(MllamaConfig())
    app.state.llama_model = llama_model
    yield

    # Clean up resources
    del app.state.llama_model


def create_app() -> FastAPI:
    """Create a FastAPI application."""

    app = FastAPI(lifespan=lifespan)

    # Set all CORS enabled origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(auth.router)
    app.include_router(views.router)

    return app


app = create_app()
