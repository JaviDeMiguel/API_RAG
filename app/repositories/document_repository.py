"""Acceso a datos de documentos y sus fragmentos (SQLite).

Cada fragmento se guarda junto con su vector de embedding (serializado como
JSON) para permitir la recuperación por similitud de coseno. Todos los
documentos pertenecen a un usuario, de modo que las consultas se filtran
siempre por `user_id`.
"""

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import uuid4

from fastapi import Depends

from app.db import Database, get_database


@dataclass
class StoredChunk:
    """Fragmento de un documento con su vector de embedding."""

    index: int
    text: str
    embedding: list[float]


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
    """Operaciones de persistencia sobre documentos y fragmentos."""

    def __init__(self, db: Database) -> None:
        self._db = db

    def add(
        self,
        user_id: str,
        title: str,
        content: str,
        chunks: list[StoredChunk],
    ) -> DocumentRecord:
        """Guarda un documento y todos sus fragmentos con embeddings."""
        document_id = uuid4().hex
        created_at = datetime.now(timezone.utc).isoformat()

        self._db.execute(
            "INSERT INTO documents (id, user_id, title, content, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (document_id, user_id, title, content, created_at),
        )
        self._db.executemany(
            "INSERT INTO chunks (document_id, idx, text, embedding) "
            "VALUES (?, ?, ?, ?)",
            [
                (document_id, chunk.index, chunk.text, json.dumps(chunk.embedding))
                for chunk in chunks
            ],
        )

        return DocumentRecord(
            id=document_id,
            user_id=user_id,
            title=title,
            created_at=created_at,
            char_count=len(content),
            chunk_count=len(chunks),
            content=content,
        )

    def list_by_user(self, user_id: str) -> list[DocumentRecord]:
        """Lista los documentos de un usuario (sin el contenido completo)."""
        rows = self._db.query_all(
            "SELECT d.id, d.user_id, d.title, d.created_at, "
            "       LENGTH(d.content) AS char_count, "
            "       (SELECT COUNT(*) FROM chunks c WHERE c.document_id = d.id) "
            "           AS chunk_count "
            "FROM documents d WHERE d.user_id = ? "
            "ORDER BY d.created_at DESC",
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
            "SELECT d.id, d.user_id, d.title, d.content, d.created_at, "
            "       LENGTH(d.content) AS char_count, "
            "       (SELECT COUNT(*) FROM chunks c WHERE c.document_id = d.id) "
            "           AS chunk_count "
            "FROM documents d WHERE d.id = ? AND d.user_id = ?",
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

    def get_chunks(self, document_id: str) -> list[StoredChunk]:
        """Devuelve todos los fragmentos (con embeddings) de un documento."""
        rows = self._db.query_all(
            "SELECT idx, text, embedding FROM chunks "
            "WHERE document_id = ? ORDER BY idx",
            (document_id,),
        )
        return [
            StoredChunk(
                index=row["idx"],
                text=row["text"],
                embedding=json.loads(row["embedding"]),
            )
            for row in rows
        ]

    def delete(self, document_id: str, user_id: str) -> bool:
        """Elimina un documento del usuario. Devuelve `True` si existía."""
        # Borramos primero los fragmentos por si las claves foráneas en cascada
        # no estuvieran activas en algún entorno.
        self._db.execute(
            "DELETE FROM chunks WHERE document_id = "
            "(SELECT id FROM documents WHERE id = ? AND user_id = ?)",
            (document_id, user_id),
        )
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
