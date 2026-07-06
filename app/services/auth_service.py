"""Lógica de autenticación: hashing de contraseñas y tokens JWT.

El hashing de contraseñas usa PBKDF2-HMAC-SHA256 (de la librería estándar), por
lo que no se requieren dependencias binarias adicionales. Los tokens de acceso
son JWT firmados con HMAC (PyJWT).
"""

import hashlib
import hmac
import secrets
from base64 import urlsafe_b64decode, urlsafe_b64encode
from datetime import datetime, timedelta, timezone

import jwt

from app.config import Settings

_PBKDF2_ROUNDS = 240_000
_ALGORITHM_LABEL = "pbkdf2_sha256"


class AuthError(Exception):
    """Error genérico de autenticación (credenciales o token inválidos)."""


# --- Hashing de contraseñas --------------------------------------------------

def hash_password(password: str) -> str:
    """Genera un hash seguro de la contraseña con sal aleatoria.

    Formato: `pbkdf2_sha256$<rondas>$<sal_b64>$<hash_b64>`.
    """
    salt = secrets.token_bytes(16)
    derived = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, _PBKDF2_ROUNDS
    )
    salt_b64 = urlsafe_b64encode(salt).decode("ascii")
    hash_b64 = urlsafe_b64encode(derived).decode("ascii")
    return f"{_ALGORITHM_LABEL}${_PBKDF2_ROUNDS}${salt_b64}${hash_b64}"


def verify_password(password: str, stored: str) -> bool:
    """Verifica una contraseña contra su hash almacenado (tiempo constante)."""
    try:
        label, rounds_str, salt_b64, hash_b64 = stored.split("$")
        if label != _ALGORITHM_LABEL:
            return False
        salt = urlsafe_b64decode(salt_b64)
        expected = urlsafe_b64decode(hash_b64)
        derived = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), salt, int(rounds_str)
        )
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(derived, expected)


# --- Tokens JWT --------------------------------------------------------------

def create_access_token(subject: str, settings: Settings) -> str:
    """Crea un JWT de acceso cuyo `sub` es el id de usuario."""
    now = datetime.now(timezone.utc)
    payload = {
        "sub": subject,
        "iat": now,
        "exp": now + timedelta(minutes=settings.access_token_expire_minutes),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def decode_access_token(token: str, settings: Settings) -> str:
    """Valida un JWT y devuelve el `sub` (id de usuario). Lanza `AuthError`."""
    try:
        payload = jwt.decode(
            token, settings.jwt_secret, algorithms=[settings.jwt_algorithm]
        )
    except jwt.PyJWTError as exc:
        raise AuthError("Token inválido o expirado.") from exc

    subject = payload.get("sub")
    if not subject:
        raise AuthError("Token sin sujeto.")
    return subject
