"""Rutas de autenticación: registro, obtención de token y perfil."""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm

from app.models.schemas import ErrorResponse, Token, UserCreate, UserPublic
from app.repositories.user_repository import UserRecord
from app.security import get_current_user
from app.services.account_service import (
    AccountService,
    DuplicateUsernameError,
    InvalidCredentialsError,
    get_account_service,
)

router = APIRouter(prefix="/auth", tags=["autenticación"])


@router.post(
    "/register",
    response_model=UserPublic,
    status_code=status.HTTP_201_CREATED,
    responses={status.HTTP_409_CONFLICT: {"model": ErrorResponse}},
    summary="Registrar un nuevo usuario",
)
def register(
    payload: UserCreate,
    service: AccountService = Depends(get_account_service),
) -> UserPublic:
    try:
        user = service.register(payload.username, payload.password)
    except DuplicateUsernameError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="El nombre de usuario ya está en uso.",
        ) from exc
    return UserPublic(id=user.id, username=user.username, created_at=user.created_at)


@router.post(
    "/token",
    response_model=Token,
    responses={status.HTTP_401_UNAUTHORIZED: {"model": ErrorResponse}},
    summary="Obtener un token de acceso (login)",
)
def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    service: AccountService = Depends(get_account_service),
) -> Token:
    try:
        access_token = service.authenticate(form_data.username, form_data.password)
    except InvalidCredentialsError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Usuario o contraseña incorrectos.",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc
    return Token(access_token=access_token)


@router.get(
    "/me",
    response_model=UserPublic,
    summary="Obtener el perfil del usuario autenticado",
)
def me(current_user: UserRecord = Depends(get_current_user)) -> UserPublic:
    return UserPublic(
        id=current_user.id,
        username=current_user.username,
        created_at=current_user.created_at,
    )
