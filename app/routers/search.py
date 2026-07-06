"""Ruta para buscar fragmentos en todos los documentos del usuario.

A diferencia de `/documents/{id}/ask`, esta búsqueda no se acota a un único
documento: recupera los fragmentos más relevantes de toda la biblioteca del
usuario (respaldada por la base de datos vectorial) y no genera respuesta con
el LLM, solo devuelve las coincidencias.
"""

from fastapi import APIRouter, Depends

from app.models.schemas import SearchRequest, SearchResponse
from app.repositories.user_repository import UserRecord
from app.security import get_current_user
from app.services.document_service import DocumentService, get_document_service

router = APIRouter(prefix="/search", tags=["búsqueda"])


@router.post(
    "",
    response_model=SearchResponse,
    summary="Buscar en todos los documentos del usuario",
)
def search(
    payload: SearchRequest,
    current_user: UserRecord = Depends(get_current_user),
    service: DocumentService = Depends(get_document_service),
) -> SearchResponse:
    return service.search(
        user_id=current_user.id, query=payload.query, top_k=payload.top_k
    )
