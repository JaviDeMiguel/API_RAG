"""Fixtures compartidas de pytest.

Preparan la aplicación para los tests **sin llamar a la API real del LLM**:
  - La base de datos se fuerza a SQLite en memoria (rápida y aislada).
  - `LLMService` se sustituye por un doble (`RecordingLLM`) que registra las
    llamadas y devuelve una respuesta fija, de modo que ningún test gasta
    cuota de la API.
"""

import os

# Debe fijarse ANTES de importar la configuración de la app.
os.environ["DB_PATH"] = ":memory:"
os.environ["CHROMA_PATH"] = ":memory:"  # base vectorial efímera en memoria
os.environ["JWT_SECRET"] = "clave-de-test-suficientemente-larga-para-hs256-123"

import pytest
from fastapi.testclient import TestClient

import app.db as db_module
import app.repositories.vector_store as vector_store_module
from app.config import get_settings
from app.main import app

# La configuración se cachea; la limpiamos para que lea las variables de test.
get_settings.cache_clear()


class RecordingLLM:
    """Doble de `LLMService`: registra las llamadas y no toca la red."""

    calls: list[dict] = []

    def __init__(self, settings=None) -> None:  # firma compatible con LLMService
        pass

    def ensure_configured(self) -> None:
        pass

    def answer(self, question: str, context_chunks: list[str]) -> str:
        RecordingLLM.calls.append(
            {"question": question, "context": list(context_chunks)}
        )
        return "RESPUESTA-SIMULADA-DEL-LLM"

    def answer_stream(self, question: str, context_chunks: list[str]):
        RecordingLLM.calls.append(
            {"question": question, "context": list(context_chunks), "stream": True}
        )
        yield from ["RESPUESTA-", "SIMULADA-", "DEL-", "LLM"]


@pytest.fixture
def llm() -> type[RecordingLLM]:
    """Reinicia el registro de llamadas y expone el doble del LLM."""
    RecordingLLM.calls = []
    return RecordingLLM


@pytest.fixture
def client(monkeypatch, llm) -> TestClient:
    """Cliente de pruebas con BD en memoria nueva y el LLM mockeado."""
    # BD relacional nueva por test (aislamiento entre tests).
    db_module._database = None
    # El cliente efímero de Chroma comparte sistema en el proceso, así que
    # reconstruimos el almacén y vaciamos la colección para aislar cada test.
    vector_store_module._vector_store = None
    vector_store_module.get_vector_store().reset()
    # Sustituimos el LLM que usa la fábrica del servicio de documentos.
    monkeypatch.setattr("app.services.document_service.LLMService", llm)

    test_client = TestClient(app)
    yield test_client

    db_module._database = None
    vector_store_module._vector_store = None


@pytest.fixture
def auth_headers(client) -> dict[str, str]:
    """Registra un usuario, hace login y devuelve la cabecera Authorization."""
    creds = {"username": "tester", "password": "password-123"}
    client.post("/auth/register", json=creds)
    token = client.post("/auth/token", data=creds).json()["access_token"]
    return {"Authorization": f"Bearer {token}"}
