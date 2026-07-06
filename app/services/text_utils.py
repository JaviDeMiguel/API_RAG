"""Utilidades de procesamiento de texto: tokenización y fragmentación."""

import re

# Palabras vacías (stopwords) básicas en español e inglés. No aportan a la
# hora de medir la similitud entre la pregunta y los fragmentos.
_STOPWORDS: frozenset[str] = frozenset([
    # Español
    "el", "la", "los", "las", "un", "una", "unos", "unas", "de", "del", "a", "al",
    "y", "o", "u", "en", "con", "por", "para", "que", "como", "se", "su", "sus",
    "lo", "le", "les", "es", "son", "fue", "ser", "este", "esta", "estos", "estas",
    "ese", "esa", "eso",
    # Inglés
    "the", "an", "and", "or", "of", "to", "in", "on", "for", "with", "is", "are",
    "was", "were", "be", "this", "that", "these", "those", "it", "its", "as", "at",
    "by", "from",
])

_TOKEN_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Divide el texto en tokens en minúsculas, ignorando stopwords y números."""
    tokens = _TOKEN_RE.findall(text.lower())
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Divide un texto en fragmentos de `chunk_size` palabras con solapamiento.

    Args:
        text: Texto completo a fragmentar.
        chunk_size: Número máximo de palabras por fragmento.
        overlap: Palabras que se repiten entre fragmentos consecutivos para no
            perder contexto en los límites.

    Returns:
        Lista de fragmentos de texto (sin fragmentos vacíos).
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size debe ser mayor que 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap debe estar en el rango [0, chunk_size)")

    words = text.split()
    if not words:
        return []

    step = chunk_size - overlap
    chunks: list[str] = []
    for start in range(0, len(words), step):
        fragment = " ".join(words[start : start + chunk_size]).strip()
        if fragment:
            chunks.append(fragment)
        if start + chunk_size >= len(words):
            break
    return chunks
