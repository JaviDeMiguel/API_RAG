"""Esquemas Pydantic para validar las entradas y salidas de la API."""

from datetime import datetime

from pydantic import BaseModel, Field

# --- Autenticación -----------------------------------------------------------

class UserCreate(BaseModel):
    """Datos para registrar un nuevo usuario."""

    username: str = Field(..., min_length=3, max_length=64)
    password: str = Field(..., min_length=8, max_length=128)


class UserPublic(BaseModel):
    """Representación pública de un usuario (sin credenciales)."""

    id: str
    username: str
    created_at: datetime


class Token(BaseModel):
    """Token de acceso devuelto al iniciar sesión."""

    access_token: str
    token_type: str = "bearer"


# --- Documentos --------------------------------------------------------------

class DocumentIngestRequest(BaseModel):
    """Cuerpo para dar de alta un documento a partir de texto plano."""

    title: str = Field(
        ...,
        min_length=1,
        max_length=256,
        description="Título identificativo del documento.",
    )
    content: str = Field(
        ...,
        min_length=1,
        description="Texto largo sobre el que se harán las preguntas.",
    )


class DocumentSummary(BaseModel):
    """Metadatos de un documento almacenado (sin devolver todo el contenido)."""

    id: str = Field(..., description="Identificador único del documento.")
    title: str
    chunk_count: int = Field(..., description="Número de fragmentos generados.")
    char_count: int = Field(..., description="Longitud del contenido en caracteres.")
    created_at: datetime


# --- Preguntas / Respuestas --------------------------------------------------

class QuestionRequest(BaseModel):
    """Pregunta que el usuario formula sobre un documento."""

    question: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="Pregunta en lenguaje natural sobre el documento.",
    )


class SourceChunk(BaseModel):
    """Fragmento del documento utilizado como fuente para la respuesta."""

    index: int = Field(..., description="Posición del fragmento en el documento.")
    text: str
    score: float = Field(
        ..., description="Similitud (coseno) del fragmento frente a la pregunta."
    )


class AnswerResponse(BaseModel):
    """Respuesta generada por el asistente junto con sus fuentes."""

    document_id: str
    question: str
    answer: str
    sources: list[SourceChunk] = Field(
        default_factory=list,
        description="Fragmentos recuperados que fundamentan la respuesta.",
    )


# --- Búsqueda entre documentos ----------------------------------------------

class SearchRequest(BaseModel):
    """Consulta para buscar fragmentos en todos los documentos del usuario."""

    query: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="Texto de búsqueda en lenguaje natural.",
    )
    top_k: int | None = Field(
        default=None,
        ge=1,
        le=50,
        description="Nº de fragmentos a devolver (por defecto, el de la config).",
    )


class SearchResult(BaseModel):
    """Fragmento relevante encontrado, con el documento al que pertenece."""

    document_id: str
    title: str = Field(..., description="Título del documento de origen.")
    index: int = Field(..., description="Posición del fragmento en el documento.")
    text: str
    score: float = Field(
        ..., description="Similitud (coseno) del fragmento frente a la consulta."
    )


class SearchResponse(BaseModel):
    """Resultados de una búsqueda entre los documentos del usuario."""

    query: str
    results: list[SearchResult] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    """Forma estándar de los errores devueltos por la API."""

    detail: str
