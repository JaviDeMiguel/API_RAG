"""Generación de embeddings y búsqueda por similitud.

Se define una interfaz `EmbeddingProvider` para que el proveedor de embeddings
sea intercambiable. La implementación por defecto (`LocalHashingEmbedding`) es
**local y determinista**: no requiere claves de API, descargas de modelos ni
dependencias pesadas, y funciona sin conexión.

Cada texto se proyecta a un vector de dimensión fija usando el truco del
*hashing* (feature hashing) sobre las palabras y sus n-gramas de caracteres —
estos últimos aportan cierta robustez morfológica (plurales, variantes). Los
vectores se normalizan (norma L2), de modo que la similitud de coseno se reduce
a un producto escalar. Para un RAG de producción, este proveedor se sustituiría
por embeddings semánticos (p. ej. Voyage AI o sentence-transformers) sin
cambiar el resto de la aplicación.
"""

import hashlib
import math
from typing import Protocol

from app.config import Settings
from app.services.text_utils import tokenize


class EmbeddingProvider(Protocol):
    """Contrato que debe cumplir cualquier proveedor de embeddings."""

    def embed(self, text: str) -> list[float]:
        """Devuelve el vector de embedding (normalizado) de un texto."""
        ...


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Similitud de coseno entre dos vectores.

    Si ambos están normalizados (norma L2 = 1), equivale al producto escalar.
    """
    return sum(x * y for x, y in zip(a, b))


class LocalHashingEmbedding:
    """Embedding local mediante feature hashing de palabras y n-gramas."""

    def __init__(self, dim: int = 512) -> None:
        if dim <= 0:
            raise ValueError("dim debe ser mayor que 0")
        self._dim = dim

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self._dim
        for token in tokenize(text):
            self._add_feature(vector, token, weight=1.0)
            # N-gramas de caracteres para captar similitud morfológica.
            for ngram in self._char_ngrams(token, 3):
                self._add_feature(vector, f"#{ngram}", weight=0.5)

        norm = math.sqrt(sum(v * v for v in vector))
        if norm > 0:
            vector = [v / norm for v in vector]
        return vector

    def _add_feature(self, vector: list[float], feature: str, weight: float) -> None:
        digest = hashlib.md5(feature.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % self._dim
        # El bit siguiente decide el signo, reduciendo colisiones sistemáticas.
        sign = 1.0 if digest[4] & 1 else -1.0
        vector[index] += sign * weight

    @staticmethod
    def _char_ngrams(token: str, n: int) -> list[str]:
        padded = f"^{token}$"
        if len(padded) < n:
            return [padded]
        return [padded[i : i + n] for i in range(len(padded) - n + 1)]


def get_embedding_provider(settings: Settings) -> EmbeddingProvider:
    """Construye el proveedor de embeddings según la configuración."""
    return LocalHashingEmbedding(dim=settings.embedding_dim)
