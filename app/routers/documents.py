"""Rutas para dar de alta, listar, consultar y eliminar documentos.

Todas requieren autenticación: cada usuario solo accede a sus documentos.
"""

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from app.models.schemas import (
    DocumentIngestRequest,
    DocumentSummary,
    ErrorResponse,
)
from app.repositories.document_repository import DocumentRecord
from app.repositories.user_repository import UserRecord
from app.security import get_current_user
from app.services.document_service import (
    DocumentNotFoundError,
    DocumentService,
    EmptyDocumentError,
    get_document_service,
)

router = APIRouter(prefix="/documents", tags=["documentos"])


def _to_summary(document: DocumentRecord) -> DocumentSummary:
    """Convierte la entidad almacenada en su representación pública."""
    return DocumentSummary(
        id=document.id,
        title=document.title,
        chunk_count=document.chunk_count,
        char_count=document.char_count,
        created_at=document.created_at,
    )


@router.post(
    "",
    response_model=DocumentSummary,
    status_code=status.HTTP_201_CREATED,
    summary="Dar de alta un documento a partir de texto plano",
)
def create_document(
    payload: DocumentIngestRequest,
    current_user: UserRecord = Depends(get_current_user),
    service: DocumentService = Depends(get_document_service),
) -> DocumentSummary:
    try:
        document = service.ingest_text(
            user_id=current_user.id,
            title=payload.title,
            content=payload.content,
        )
    except EmptyDocumentError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
    return _to_summary(document)


@router.post(
    "/upload",
    response_model=DocumentSummary,
    status_code=status.HTTP_201_CREATED,
    summary="Dar de alta un documento a partir de un archivo PDF",
)
async def upload_document(
    title: str = Form(..., min_length=1, max_length=256),
    file: UploadFile = File(..., description="Archivo PDF a procesar."),
    current_user: UserRecord = Depends(get_current_user),
    service: DocumentService = Depends(get_document_service),
) -> DocumentSummary:
    if file.content_type not in ("application/pdf", "application/octet-stream"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Solo se aceptan archivos PDF.",
        )

    pdf_bytes = await file.read()
    try:
        document = service.ingest_pdf(
            user_id=current_user.id, title=title, pdf_bytes=pdf_bytes
        )
    except EmptyDocumentError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No se pudo extraer texto del PDF.",
        ) from exc
    except Exception as exc:  # p. ej. PDF corrupto
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"No se pudo procesar el PDF: {exc}",
        ) from exc
    return _to_summary(document)


@router.get(
    "",
    response_model=list[DocumentSummary],
    summary="Listar los documentos del usuario",
)
def list_documents(
    current_user: UserRecord = Depends(get_current_user),
    service: DocumentService = Depends(get_document_service),
) -> list[DocumentSummary]:
    return [_to_summary(doc) for doc in service.list_documents(current_user.id)]


@router.get(
    "/{document_id}",
    response_model=DocumentSummary,
    responses={status.HTTP_404_NOT_FOUND: {"model": ErrorResponse}},
    summary="Obtener los metadatos de un documento",
)
def get_document(
    document_id: str,
    current_user: UserRecord = Depends(get_current_user),
    service: DocumentService = Depends(get_document_service),
) -> DocumentSummary:
    try:
        document = service.get_document(current_user.id, document_id)
    except DocumentNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Documento no encontrado.",
        ) from exc
    return _to_summary(document)


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={status.HTTP_404_NOT_FOUND: {"model": ErrorResponse}},
    summary="Eliminar un documento",
)
def delete_document(
    document_id: str,
    current_user: UserRecord = Depends(get_current_user),
    service: DocumentService = Depends(get_document_service),
) -> None:
    try:
        service.delete_document(current_user.id, document_id)
    except DocumentNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Documento no encontrado.",
        ) from exc
