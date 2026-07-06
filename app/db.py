"""Acceso a la base de datos SQLite.

Envuelve una conexión `sqlite3` con un cerrojo para que sea segura frente a la
concurrencia (FastAPI ejecuta los endpoints síncronos en un pool de hilos).
Toda la aplicación comparte una única instancia de `Database`.
"""

import sqlite3
import threading
from pathlib import Path
from typing import Any, Sequence

from app.config import get_settings

# Esquema de la base de datos. Se crea si no existe al arrancar.
_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id            TEXT PRIMARY KEY,
    username      TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS documents (
    id         TEXT PRIMARY KEY,
    user_id    TEXT NOT NULL,
    title      TEXT NOT NULL,
    content    TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS chunks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id TEXT NOT NULL,
    idx         INTEGER NOT NULL,
    text        TEXT NOT NULL,
    embedding   TEXT NOT NULL,
    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_documents_user ON documents(user_id);
CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);
"""


class Database:
    """Conexión SQLite compartida y protegida por un cerrojo."""

    def __init__(self, path: str) -> None:
        if path != ":memory:":
            parent = Path(path).parent
            if parent and not parent.exists():
                parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._lock = threading.Lock()
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    def execute(self, sql: str, params: Sequence[Any] = ()) -> int:
        """Ejecuta una sentencia de escritura y devuelve el `lastrowid`."""
        with self._lock:
            cursor = self._conn.execute(sql, params)
            self._conn.commit()
            return cursor.lastrowid

    def executemany(self, sql: str, seq_params: Sequence[Sequence[Any]]) -> None:
        """Ejecuta la misma sentencia para múltiples filas."""
        with self._lock:
            self._conn.executemany(sql, seq_params)
            self._conn.commit()

    def query_one(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Row | None:
        """Devuelve la primera fila o `None`."""
        with self._lock:
            return self._conn.execute(sql, params).fetchone()

    def query_all(self, sql: str, params: Sequence[Any] = ()) -> list[sqlite3.Row]:
        """Devuelve todas las filas."""
        with self._lock:
            return self._conn.execute(sql, params).fetchall()


# Instancia única compartida por toda la aplicación.
_database: Database | None = None


def get_database() -> Database:
    """Dependencia de FastAPI que expone la base de datos compartida."""
    global _database
    if _database is None:
        _database = Database(get_settings().db_path)
    return _database
