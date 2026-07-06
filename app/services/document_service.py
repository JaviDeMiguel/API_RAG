"""Lógica de negocio del asistente de documentos (orquestación RAG).

Coordina las tres piezas del flujo:
  1. Ingesta: fragmentar el texto, calcular embeddings y persistir.
  2. Recuperación: seleccionar los fragmentos más similares a la pregunta
     mediante similitud de coseno sobre los embeddings.
  3. Generación: pedir al LLM una respuesta basada en esos fragmentos.

Todas las operaciones están acotadas al usuario propietario del documento.
"""

from io import BytesIO

from fastapi import Depends
from pypdf import PdfReader

from app.config import Settings, get_settings
from app.models.schemas import AnswerResponse, SourceChunk
from app.repositories.document_repository import (
    DocumentRecord,
    DocumentRepository,
    StoredChunk,
    get_document_repository,
)
from app.services import text_utils
from app.services.embedding_service import (
    EmbeddingProvider,
    cosine_similarity,
    get_embedding_provider,
)
from app.services.llm_service import LLMService


class DocumentNotFoundError(Exception):
    """El documento solicitado no existe o no pertenece al usuario."""


class EmptyDocumentError(Exception):
    """El contenido proporcionado no contiene texto utilizable."""


class DocumentService:
    """Servicio principal que implementa el RAG (con embeddings)."""

    def __init__(
        self,
        repository: DocumentRepository,
        settings: Settings,
        llm_service: LLMService,
        embedding_provider: EmbeddingProvider,
    ) -> None:
        self._repository = repository
        self._settings = settings
        self._llm = llm_service
        self._embeddings = embedding_provider

    # --- Ingesta -------------------------------------------------------------

    def ingest_text(
        self, user_id: str, title: str, content: str
    ) -> DocumentRecord:
        """Fragmenta, vectoriza y almacena un documento de texto plano."""
        text = content.strip()
        if not text:
            raise EmptyDocumentError("El documento está vacío.")

        fragments = text_utils.chunk_text(
            text,
            chunk_size=self._settings.chunk_size,
            overlap=self._settings.chunk_overlap,
        )
        if not fragments:
            raise EmptyDocumentError("No se pudo fragmentar el documento.")

        chunks = [
            StoredChunk(
                index=index,
                text=fragment,
                embedding=self._embeddings.embed(fragment),
            )
            for index, fragment in enumerate(fragments)
        ]

        return self._repository.add(
            user_id=user_id, title=title, content=text, chunks=chunks
        )

    def ingest_pdf(
        self, user_id: str, title: str, pdf_bytes: bytes
    ) -> DocumentRecord:
        """Extrae el texto de un PDF y lo almacena como documento."""
        text = self._extract_pdf_text(pdf_bytes)
        return self.ingest_text(user_id=user_id, title=title, content=text)

    @staticmethod
    def _extract_pdf_text(pdf_bytes: bytes) -> str:
        """Extrae el texto de todas las páginas de un PDF."""
        reader = PdfReader(BytesIO(pdf_bytes))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages).strip()

    # --- Consulta ------------------------------------------------------------

    def list_documents(self, user_id: str) -> list[DocumentRecord]:
        """Devuelve los documentos del usuario."""
        return self._repository.list_by_user(user_id)

    def get_document(self, user_id: str, document_id: str) -> DocumentRecord:
        """Recupera un documento del usuario o lanza `DocumentNotFoundError`."""
        document = self._repository.get(document_id, user_id)
        if document is None:
            raise DocumentNotFoundError(document_id)
        return document

    def delete_document(self, user_id: str, document_id: str) -> None:
        """Elimina un documento del usuario o lanza `DocumentNotFoundError`."""
        if not self._repository.delete(document_id, user_id):
            raise DocumentNotFoundError(document_id)

    # --- Pregunta / Respuesta (RAG) -----------------------------------------

    def answer_question(
        self, user_id: str, document_id: str, question: str
    ) -> AnswerResponse:
        """Responde una pregunta sobre un documento aplicando RAG.

        Recupera los `top_k` fragmentos más similares (coseno) y se los pasa al
        LLM como contexto para generar la respuesta.
        """
        document = self.get_document(user_id, document_id)

        sources = self._retrieve(document_id, question)
        context_chunks = [source.text for source in sources]

        answer = self._llm.answer(question=question, context_chunks=context_chunks)

        return AnswerResponse(
            document_id=document.id,
            question=question,
            answer=answer,
            sources=sources,
        )

    def _retrieve(self, document_id: str, question: str) -> list[SourceChunk]:
        """Selecciona los fragmentos más similares a la pregunta por coseno."""
        query_vector = self._embeddings.embed(question)

        scored: list[SourceChunk] = []
        for chunk in self._repository.get_chunks(document_id):
            score = cosine_similarity(query_vector, chunk.embedding)
            scored.append(
                SourceChunk(index=chunk.index, text=chunk.text, score=score)
            )

        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[: self._settings.top_k]


def get_document_service(
    repository: DocumentRepository = Depends(get_document_repository),
    settings: Settings = Depends(get_settings),
) -> DocumentService:
    """Dependencia de FastAPI que construye el servicio con sus colaboradores."""
    return DocumentService(
        repository=repository,
        settings=settings,
        llm_service=LLMService(settings),
        embedding_provider=get_embedding_provider(settings),
    )
