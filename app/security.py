"""Dependencias de seguridad de FastAPI (autenticación de peticiones)."""

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

from app.config import Settings, get_settings
from app.repositories.user_repository import (
    UserRecord,
    UserRepository,
    get_user_repository,
)
from app.services.auth_service import AuthError, decode_access_token

# El cliente obtiene el token en POST /auth/token (flujo OAuth2 password).
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/token")


def get_current_user(
    token: str = Depends(oauth2_scheme),
    settings: Settings = Depends(get_settings),
    users: UserRepository = Depends(get_user_repository),
) -> UserRecord:
    """Resuelve el usuario autenticado a partir del token JWT del encabezado."""
    credentials_error = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="No se pudieron validar las credenciales.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        user_id = decode_access_token(token, settings)
    except AuthError as exc:
        raise credentials_error from exc

    user = users.get_by_id(user_id)
    if user is None:
        raise credentials_error
    return user
