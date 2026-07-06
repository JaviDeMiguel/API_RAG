"""Configuración de la aplicación cargada desde variables de entorno / .env."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Ajustes globales de la aplicación.

    Los valores se leen (en este orden de prioridad) de las variables de
    entorno del proceso y del archivo `.env` de la raíz del proyecto.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Anthropic / Claude ---
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-opus-4-8"
    max_answer_tokens: int = 2048

    # --- Base de datos ---
    db_path: str = "data/app.db"

    # --- Base de datos vectorial (ChromaDB) ---
    # Directorio de persistencia de Chroma. Usa ":memory:" para una instancia
    # efímera en memoria (útil en tests).
    chroma_path: str = "data/chroma"
    chroma_collection: str = "chunks"

    # --- Autenticación (JWT) ---
    jwt_secret: str = "cambia-esto-por-un-secreto-largo-y-aleatorio"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60

    # --- Fragmentación (chunking) ---
    chunk_size: int = 180          # nº de palabras por fragmento
    chunk_overlap: int = 40        # solapamiento entre fragmentos consecutivos

    # --- Recuperación (retrieval por embeddings) ---
    top_k: int = 4                 # fragmentos relevantes que se envían al LLM

    # Proveedor de embeddings: "local" (por defecto, sin claves ni red) o
    # "voyage" (embeddings semánticos de Voyage AI, el partner recomendado por
    # Anthropic para usar con Claude).
    embedding_provider: str = "local"
    embedding_dim: int = 512       # dimensión del vector del proveedor local

    # --- Voyage AI (embeddings semánticos, opcional) ---
    voyage_api_key: str | None = None
    voyage_model: str = "voyage-3.5"
    # Dimensión de salida del vector (voyage-3.5 admite 256/512/1024/2048).
    # None => se usa la dimensión por defecto del modelo.
    voyage_embedding_dim: int | None = None


@lru_cache
def get_settings() -> Settings:
    """Devuelve una instancia cacheada de la configuración."""
    return Settings()
