"""Lógica de negocio del asistente de documentos (orquestación RAG).

Coordina las tres piezas del flujo:
  1. Ingesta: fragmentar el texto, calcular embeddings y persistir.
  2. Recuperación: seleccionar los fragmentos más similares a la pregunta
     mediante similitud de coseno sobre los embeddings.
  3. Generación: pedir al LLM una respuesta basada en esos fragmentos.

Todas las operaciones están acotadas al usuario propietario del documento.
"""

from collections.abc import Iterator
from io import BytesIO

from fastapi import Depends
from pypdf import PdfReader

from app.config import Settings, get_settings
from app.models.schemas import (
    AnswerResponse,
    SearchResponse,
    SearchResult,
    SourceChunk,
)
from app.repositories.document_repository import (
    DocumentRecord,
    DocumentRepository,
    get_document_repository,
)
from app.repositories.vector_store import (
    ChunkVectorStore,
    StoredChunk,
    get_vector_store,
)
from app.services import text_utils
from app.services.embedding_service import (
    EmbeddingProvider,
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
        vector_store: ChunkVectorStore,
        settings: Settings,
        llm_service: LLMService,
        embedding_provider: EmbeddingProvider,
    ) -> None:
        self._repository = repository
        self._vectors = vector_store
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

        # Vectorizamos todos los fragmentos de una vez: con proveedores remotos
        # (p. ej. Voyage) esto es una única llamada en lote en lugar de N.
        embeddings = self._embeddings.embed_documents(fragments)
        chunks = [
            StoredChunk(index=index, text=fragment, embedding=embedding)
            for index, (fragment, embedding) in enumerate(
                zip(fragments, embeddings, strict=True)
            )
        ]

        # Metadatos en SQLite; vectores en la base de datos vectorial (Chroma).
        record = self._repository.add(
            user_id=user_id, title=title, content=text, chunk_count=len(chunks)
        )
        self._vectors.add(document_id=record.id, user_id=user_id, chunks=chunks)
        return record

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
        # Los metadatos existían y se han borrado: eliminamos también sus vectores.
        self._vectors.delete(document_id)

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

    def answer_question_stream(
        self, user_id: str, document_id: str, question: str
    ) -> tuple[list[SourceChunk], Iterator[str]]:
        """Como `answer_question`, pero devuelve la respuesta en streaming.

        Realiza la comprobación de propiedad, la recuperación y valida que el
        LLM está configurado **antes** de devolver el generador, de modo que los
        errores (404 / 503) se produzcan antes de empezar a emitir la respuesta.

        Returns:
            Una tupla `(fuentes, generador_de_texto)`.
        """
        self.get_document(user_id, document_id)  # valida propiedad (404)

        sources = self._retrieve(document_id, question)
        context_chunks = [source.text for source in sources]

        self._llm.ensure_configured()  # valida configuración (503) antes de emitir
        token_stream = self._llm.answer_stream(
            question=question, context_chunks=context_chunks
        )
        return sources, token_stream

    # --- Búsqueda entre documentos ------------------------------------------

    def search(
        self, user_id: str, query: str, top_k: int | None = None
    ) -> SearchResponse:
        """Busca los fragmentos más relevantes entre TODOS los documentos del
        usuario (no acotado a un único documento)."""
        limit = top_k or self._settings.top_k
        query_vector = self._embeddings.embed_query(query)
        hits = self._vectors.query_user(user_id, query_vector, limit)

        titles = self._repository.titles(
            user_id, list({hit.document_id for hit in hits})
        )
        results = [
            SearchResult(
                document_id=hit.document_id,
                title=titles.get(hit.document_id, ""),
                index=hit.index,
                text=hit.text,
                score=hit.score,
            )
            for hit in hits
        ]
        return SearchResponse(query=query, results=results)

    def _retrieve(self, document_id: str, question: str) -> list[SourceChunk]:
        """Selecciona los fragmentos más similares a la pregunta.

        La búsqueda por vecinos más cercanos (coseno) la realiza la base de
        datos vectorial; aquí solo vectorizamos la pregunta y delegamos.
        """
        query_vector = self._embeddings.embed_query(question)
        return self._vectors.query(document_id, query_vector, self._settings.top_k)


def get_document_service(
    repository: DocumentRepository = Depends(get_document_repository),
    vector_store: ChunkVectorStore = Depends(get_vector_store),
    settings: Settings = Depends(get_settings),
) -> DocumentService:
    """Dependencia de FastAPI que construye el servicio con sus colaboradores."""
    return DocumentService(
        repository=repository,
        vector_store=vector_store,
        settings=settings,
        llm_service=LLMService(settings),
        embedding_provider=get_embedding_provider(settings),
    )
