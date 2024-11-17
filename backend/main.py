from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from backend.apis.config import MllamaConfig
from backend.apis.llama import ModelManager
from backend.core import auth
from backend.routes import views


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
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
