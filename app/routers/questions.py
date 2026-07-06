"""Ruta para formular preguntas sobre un documento (endpoint RAG)."""

from fastapi import APIRouter, Depends, HTTPException, status

from app.models.schemas import AnswerResponse, ErrorResponse, QuestionRequest
from app.repositories.user_repository import UserRecord
from app.security import get_current_user
from app.services.document_service import (
    DocumentNotFoundError,
    DocumentService,
    get_document_service,
)
from app.services.llm_service import LLMConfigurationError

router = APIRouter(prefix="/documents", tags=["preguntas"])


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
