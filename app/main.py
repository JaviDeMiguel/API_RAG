"""Punto de entrada de la API de Asistente Virtual para Documentos.

Ejecutar en desarrollo con:
    uvicorn app.main:app --reload
Documentación interactiva disponible en /docs (Swagger) y /redoc.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app import __version__
from app.db import get_database
from app.routers import auth, documents, questions, search


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Inicializa la base de datos (crea las tablas si no existen) al arrancar."""
    get_database()
    yield


app = FastAPI(
    title="Asistente Virtual para Documentos",
    description=(
        "API de tipo RAG: regístrate, sube un texto largo o un PDF y hazle "
        "preguntas en lenguaje natural. Los fragmentos relevantes se recuperan "
        "por similitud de embeddings y las respuestas las genera Claude. Cada "
        "usuario solo accede a sus propios documentos."
    ),
    version=__version__,
    lifespan=lifespan,
)

app.include_router(auth.router)
app.include_router(documents.router)
app.include_router(questions.router)
app.include_router(search.router)


@app.get("/health", tags=["salud"], summary="Comprobación de estado")
def health() -> dict[str, str]:
    """Endpoint sencillo para verificar que el servicio está en marcha."""
    return {"status": "ok", "version": __version__}
