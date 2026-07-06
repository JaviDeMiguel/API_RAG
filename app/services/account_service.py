"""Servicio de cuentas: registro y autenticación de usuarios."""

from fastapi import Depends

from app.config import Settings, get_settings
from app.repositories.user_repository import (
    UsernameAlreadyExistsError,
    UserRecord,
    UserRepository,
    get_user_repository,
)
from app.services import auth_service


class DuplicateUsernameError(Exception):
    """El nombre de usuario ya existe."""


class InvalidCredentialsError(Exception):
    """Usuario o contraseña incorrectos."""


class AccountService:
    """Casos de uso relacionados con las cuentas de usuario."""

    def __init__(self, users: UserRepository, settings: Settings) -> None:
        self._users = users
        self._settings = settings

    def register(self, username: str, password: str) -> UserRecord:
        """Registra un usuario nuevo con la contraseña ya hasheada."""
        password_hash = auth_service.hash_password(password)
        try:
            return self._users.create(username=username, password_hash=password_hash)
        except UsernameAlreadyExistsError as exc:
            raise DuplicateUsernameError(username) from exc

    def authenticate(self, username: str, password: str) -> str:
        """Valida las credenciales y devuelve un token de acceso (JWT)."""
        user = self._users.get_by_username(username)
        if user is None or not auth_service.verify_password(
            password, user.password_hash
        ):
            raise InvalidCredentialsError()
        return auth_service.create_access_token(user.id, self._settings)


def get_account_service(
    users: UserRepository = Depends(get_user_repository),
    settings: Settings = Depends(get_settings),
) -> AccountService:
    """Dependencia de FastAPI que construye el servicio de cuentas."""
    return AccountService(users=users, settings=settings)
