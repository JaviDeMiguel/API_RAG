"""Pruebas de `LLMService` con el cliente de Anthropic mockeado.

Objetivo: verificar que construimos correctamente la petición y extraemos el
texto de la respuesta, SIN realizar ninguna llamada real a la API.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from app.config import Settings
from app.services.llm_service import LLMConfigurationError, LLMService


def _fake_response() -> SimpleNamespace:
    """Simula la respuesta del SDK: bloques de pensamiento + texto."""
    return SimpleNamespace(
        content=[
            SimpleNamespace(type="thinking", text=""),
            SimpleNamespace(type="text", text="Respuesta "),
            SimpleNamespace(type="text", text="final."),
        ]
    )


def test_answer_construye_peticion_y_extrae_texto(monkeypatch):
    settings = Settings(
        anthropic_api_key="clave-falsa",
        anthropic_model="claude-opus-4-8",
        max_answer_tokens=1234,
    )

    fake_client = MagicMock()
    fake_client.messages.create.return_value = _fake_response()
    # Sustituimos el constructor del cliente: nunca se abre una conexión real.
    monkeypatch.setattr(
        "app.services.llm_service.anthropic.Anthropic",
        MagicMock(return_value=fake_client),
    )

    service = LLMService(settings)
    answer = service.answer("¿Qué es X?", ["fragmento uno", "fragmento dos"])

    # Solo se concatenan los bloques de tipo "text".
    assert answer == "Respuesta final."

    # Se llamó exactamente una vez, con los parámetros esperados.
    fake_client.messages.create.assert_called_once()
    kwargs = fake_client.messages.create.call_args.kwargs
    assert kwargs["model"] == "claude-opus-4-8"
    assert kwargs["max_tokens"] == 1234
    assert kwargs["thinking"] == {"type": "adaptive"}

    # El contexto recuperado y la pregunta viajan en el mensaje del usuario.
    user_message = kwargs["messages"][0]["content"]
    assert "fragmento uno" in user_message
    assert "fragmento dos" in user_message
    assert "¿Qué es X?" in user_message


def test_answer_sin_clave_lanza_error_configuracion():
    service = LLMService(Settings(anthropic_api_key=None))
    with pytest.raises(LLMConfigurationError):
        service.answer("pregunta", ["contexto"])


def test_cliente_se_crea_una_sola_vez(monkeypatch):
    settings = Settings(anthropic_api_key="clave-falsa")
    fake_client = MagicMock()
    fake_client.messages.create.return_value = _fake_response()
    constructor = MagicMock(return_value=fake_client)
    monkeypatch.setattr("app.services.llm_service.anthropic.Anthropic", constructor)

    service = LLMService(settings)
    service.answer("p1", ["c"])
    service.answer("p2", ["c"])

    # El cliente se construye de forma perezosa y se reutiliza.
    constructor.assert_called_once()
