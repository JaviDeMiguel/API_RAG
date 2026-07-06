"""Servicio de acceso al LLM (Claude) mediante el SDK oficial de Anthropic.

Encapsula toda la interacción con la API de Anthropic para que el resto de la
aplicación no dependa directamente del SDK. Usamos pensamiento adaptativo
(`thinking: adaptive`), el modo recomendado para los modelos Claude 4.6+.
"""

import anthropic

from app.config import Settings

# Prompt de sistema: fija el comportamiento del asistente para que responda
# ÚNICAMENTE con la información del contexto recuperado (patrón RAG).
_SYSTEM_PROMPT = (
    "Eres un asistente que responde preguntas sobre un documento. "
    "Responde de forma precisa y concisa usando EXCLUSIVAMENTE la información "
    "de los fragmentos de contexto proporcionados. Si la respuesta no está en "
    "el contexto, indícalo claramente diciendo que el documento no contiene esa "
    "información. No inventes datos ni uses conocimiento externo."
)


class LLMConfigurationError(RuntimeError):
    """Se lanza cuando falta la configuración necesaria para llamar al LLM."""


class LLMService:
    """Genera respuestas en lenguaje natural a partir de un contexto."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: anthropic.Anthropic | None = None

    def _get_client(self) -> anthropic.Anthropic:
        """Crea (perezosamente) el cliente de Anthropic."""
        if self._client is None:
            if not self._settings.anthropic_api_key:
                raise LLMConfigurationError(
                    "Falta ANTHROPIC_API_KEY. Configúrala en el archivo .env "
                    "o como variable de entorno para poder responder preguntas."
                )
            self._client = anthropic.Anthropic(
                api_key=self._settings.anthropic_api_key
            )
        return self._client

    def answer(self, question: str, context_chunks: list[str]) -> str:
        """Pide a Claude una respuesta a `question` usando `context_chunks`.

        Args:
            question: Pregunta del usuario.
            context_chunks: Fragmentos relevantes recuperados del documento.

        Returns:
            El texto de la respuesta generada por el modelo.
        """
        client = self._get_client()

        context = "\n\n".join(
            f"[Fragmento {i + 1}]\n{chunk}"
            for i, chunk in enumerate(context_chunks)
        )
        user_message = (
            f"Contexto extraído del documento:\n\n{context}\n\n"
            f"Pregunta: {question}"
        )

        response = client.messages.create(
            model=self._settings.anthropic_model,
            max_tokens=self._settings.max_answer_tokens,
            thinking={"type": "adaptive"},
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )

        # La respuesta puede incluir bloques de pensamiento antes del texto;
        # nos quedamos únicamente con los bloques de tipo "text".
        return "".join(
            block.text for block in response.content if block.type == "text"
        ).strip()
