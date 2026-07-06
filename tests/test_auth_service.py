"""Pruebas del hashing de contraseñas y los tokens JWT."""

import pytest

from app.config import Settings
from app.services import auth_service
from app.services.auth_service import AuthError

_SETTINGS = Settings(
    jwt_secret="secreto-de-prueba-suficientemente-largo-para-hs256",
    access_token_expire_minutes=5,
)


def test_hash_y_verificacion_de_contrasena():
    hashed = auth_service.hash_password("SuperSecreta123")
    assert hashed != "SuperSecreta123"
    assert auth_service.verify_password("SuperSecreta123", hashed)
    assert not auth_service.verify_password("incorrecta", hashed)


def test_hashes_distintos_por_sal_aleatoria():
    a = auth_service.hash_password("misma-clave")
    b = auth_service.hash_password("misma-clave")
    assert a != b  # la sal aleatoria produce hashes diferentes


def test_token_ida_y_vuelta():
    token = auth_service.create_access_token("user-123", _SETTINGS)
    assert auth_service.decode_access_token(token, _SETTINGS) == "user-123"


def test_token_invalido_lanza_error():
    with pytest.raises(AuthError):
        auth_service.decode_access_token("token.falso.invalido", _SETTINGS)
