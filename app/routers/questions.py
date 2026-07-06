"""Ruta para formular preguntas sobre un documento (endpoint RAG)."""

import json
from typing import Iterator

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from app.models.schemas import (
    AnswerResponse,
    ErrorResponse,
    QuestionRequest,
    SourceChunk,
)
from app.repositories.user_repository import UserRecord
from app.security import get_current_user
from app.services.document_service import (
    DocumentNotFoundError,
    DocumentService,
    get_document_service,
)
from app.services.llm_service import LLMConfigurationError

router = APIRouter(prefix="/documents", tags=["preguntas"])


def _sse_event(payload: dict) -> str:
    """Serializa un objeto como un evento Server-Sent Events (SSE)."""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _stream_answer(
    sources: list[SourceChunk], token_stream: Iterator[str]
) -> Iterator[str]:
    """Genera el flujo SSE: primero las fuentes, luego el texto por trozos."""
    yield _sse_event(
        {"type": "sources", "sources": [s.model_dump() for s in sources]}
    )
    for text in token_stream:
        if text:
            yield _sse_event({"type": "token", "text": text})
    yield _sse_event({"type": "done"})


@router.post(
    "/{document_id}/ask",
    response_model=AnswerResponse,
    responses={
        status.HTTP_404_NOT_FOUND: {"model": ErrorResponse},
        status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ErrorResponse},
    },
    summary="Hacer una pregunta sobre un documento",
)
def ask_question(
    document_id: str,
    payload: QuestionRequest,
    current_user: UserRecord = Depends(get_current_user),
    service: DocumentService = Depends(get_document_service),
) -> AnswerResponse:
    try:
        return service.answer_question(
            user_id=current_user.id,
            document_id=document_id,
            question=payload.question,
        )
    except DocumentNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Documento no encontrado.",
        ) from exc
    except LLMConfigurationError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc


@router.post(
    "/{document_id}/ask/stream",
    responses={
        status.HTTP_404_NOT_FOUND: {"model": ErrorResponse},
        status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ErrorResponse},
    },
    summary="Hacer una pregunta con respuesta en streaming (SSE)",
    description=(
        "Igual que `/ask`, pero devuelve la respuesta en streaming como "
        "Server-Sent Events. El primer evento es `{'type': 'sources', ...}` con "
        "los fragmentos recuperados; después llegan eventos "
        "`{'type': 'token', 'text': ...}` con el texto según se genera y, al "
        "final, `{'type': 'done'}`."
    ),
)
def ask_question_stream(
    document_id: str,
    payload: QuestionRequest,
    current_user: UserRecord = Depends(get_current_user),
    service: DocumentService = Depends(get_document_service),
) -> StreamingResponse:
    # La recuperación y las validaciones (404 / 503) ocurren aquí, antes de
    # empezar a emitir: una vez enviadas las cabeceras no se puede cambiar
    # el estado de la respuesta.
    try:
        sources, token_stream = service.answer_question_stream(
            user_id=current_user.id,
            document_id=document_id,
            question=payload.question,
        )
    except DocumentNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Documento no encontrado.",
        ) from exc
    except LLMConfigurationError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    return StreamingResponse(
        _stream_answer(sources, token_stream),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
