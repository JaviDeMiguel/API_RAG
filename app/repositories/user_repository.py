"""Acceso a datos de usuarios (tabla `users`)."""

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import uuid4

from fastapi import Depends

from app.db import Database, get_database


@dataclass
class UserRecord:
    """Usuario tal y como se almacena (incluye el hash de la contraseña)."""

    id: str
    username: str
    password_hash: str
    created_at: str


class UsernameAlreadyExistsError(Exception):
    """El nombre de usuario ya está registrado."""


class UserRepository:
    """Operaciones de persistencia sobre usuarios."""

    def __init__(self, db: Database) -> None:
        self._db = db

    def create(self, username: str, password_hash: str) -> UserRecord:
        """Crea un usuario. Lanza `UsernameAlreadyExistsError` si ya existe."""
        record = UserRecord(
            id=uuid4().hex,
            username=username,
            password_hash=password_hash,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        try:
            self._db.execute(
                "INSERT INTO users (id, username, password_hash, created_at) "
                "VALUES (?, ?, ?, ?)",
                (record.id, record.username, record.password_hash, record.created_at),
            )
        except sqlite3.IntegrityError as exc:
            raise UsernameAlreadyExistsError(username) from exc
        return record

    def get_by_username(self, username: str) -> UserRecord | None:
        row = self._db.query_one(
            "SELECT id, username, password_hash, created_at "
            "FROM users WHERE username = ?",
            (username,),
        )
        return self._to_record(row)

    def get_by_id(self, user_id: str) -> UserRecord | None:
        row = self._db.query_one(
            "SELECT id, username, password_hash, created_at FROM users WHERE id = ?",
            (user_id,),
        )
        return self._to_record(row)

    @staticmethod
    def _to_record(row: sqlite3.Row | None) -> UserRecord | None:
        if row is None:
            return None
        return UserRecord(
            id=row["id"],
            username=row["username"],
            password_hash=row["password_hash"],
            created_at=row["created_at"],
        )


def get_user_repository(
    db: Database = Depends(get_database),
) -> UserRepository:
    """Dependencia de FastAPI que construye el repositorio de usuarios."""
    return UserRepository(db)
