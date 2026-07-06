"""Acceso a datos de los metadatos de documentos (SQLite).

Guarda los metadatos relacionales de cada documento (propietario, título,
contenido, nº de fragmentos, fecha). Los fragmentos y sus embeddings viven en
la base de datos vectorial (ChromaDB, ver `vector_store.py`). Todas las
consultas se filtran por `user_id` para acotar el acceso al propietario.
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import uuid4

from fastapi import Depends

from app.db import Database, get_database


@dataclass
class DocumentRecord:
    """Documento almacenado. `content` puede omitirse en los listados."""

    id: str
    user_id: str
    title: str
    created_at: str
    char_count: int
    chunk_count: int
    content: str | None = None


class DocumentRepository:
    """Operaciones de persistencia sobre los metadatos de documentos."""

    def __init__(self, db: Database) -> None:
        self._db = db

    def add(
        self,
        user_id: str,
        title: str,
        content: str,
        chunk_count: int,
    ) -> DocumentRecord:
        """Guarda los metadatos de un documento y devuelve el registro creado."""
        document_id = uuid4().hex
        created_at = datetime.now(UTC).isoformat()

        self._db.execute(
            "INSERT INTO documents (id, user_id, title, content, chunk_count, "
            "created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (document_id, user_id, title, content, chunk_count, created_at),
        )

        return DocumentRecord(
            id=document_id,
            user_id=user_id,
            title=title,
            created_at=created_at,
            char_count=len(content),
            chunk_count=chunk_count,
            content=content,
        )

    def list_by_user(self, user_id: str) -> list[DocumentRecord]:
        """Lista los documentos de un usuario (sin el contenido completo)."""
        rows = self._db.query_all(
            "SELECT id, user_id, title, created_at, chunk_count, "
            "       LENGTH(content) AS char_count "
            "FROM documents WHERE user_id = ? ORDER BY created_at DESC",
            (user_id,),
        )
        return [
            DocumentRecord(
                id=row["id"],
                user_id=row["user_id"],
                title=row["title"],
                created_at=row["created_at"],
                char_count=row["char_count"],
                chunk_count=row["chunk_count"],
            )
            for row in rows
        ]

    def get(self, document_id: str, user_id: str) -> DocumentRecord | None:
        """Recupera un documento del usuario (con su contenido)."""
        row = self._db.query_one(
            "SELECT id, user_id, title, content, created_at, chunk_count, "
            "       LENGTH(content) AS char_count "
            "FROM documents WHERE id = ? AND user_id = ?",
            (document_id, user_id),
        )
        if row is None:
            return None
        return DocumentRecord(
            id=row["id"],
            user_id=row["user_id"],
            title=row["title"],
            content=row["content"],
            created_at=row["created_at"],
            char_count=row["char_count"],
            chunk_count=row["chunk_count"],
        )

    def titles(
        self, user_id: str, document_ids: list[str]
    ) -> dict[str, str]:
        """Devuelve `{id: título}` para los documentos del usuario indicados."""
        if not document_ids:
            return {}
        placeholders = ",".join("?" * len(document_ids))
        rows = self._db.query_all(
            f"SELECT id, title FROM documents "
            f"WHERE user_id = ? AND id IN ({placeholders})",
            (user_id, *document_ids),
        )
        return {row["id"]: row["title"] for row in rows}

    def delete(self, document_id: str, user_id: str) -> bool:
        """Elimina un documento del usuario. Devuelve `True` si existía.

        No borra los fragmentos vectoriales: de eso se encarga el servicio a
        través del almacén vectorial.
        """
        row = self._db.query_one(
            "SELECT id FROM documents WHERE id = ? AND user_id = ?",
            (document_id, user_id),
        )
        if row is None:
            return False
        self._db.execute(
            "DELETE FROM documents WHERE id = ? AND user_id = ?",
            (document_id, user_id),
        )
        return True


def get_document_repository(
    db: Database = Depends(get_database),
) -> DocumentRepository:
    """Dependencia de FastAPI que construye el repositorio de documentos."""
    return DocumentRepository(db)
