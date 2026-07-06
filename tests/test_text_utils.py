"""Pruebas de la tokenización y la fragmentación de texto."""

from app.services import text_utils


def test_chunk_text_respeta_tamano_y_solapamiento():
    words = [f"p{i}" for i in range(100)]
    text = " ".join(words)

    chunks = text_utils.chunk_text(text, chunk_size=30, overlap=10)

    assert len(chunks) > 1
    # Cada fragmento tiene como mucho 30 palabras.
    assert all(len(c.split()) <= 30 for c in chunks)
    # El solapamiento hace que el segundo fragmento empiece en la palabra 20.
    assert chunks[1].split()[0] == "p20"


def test_chunk_text_vacio_devuelve_lista_vacia():
    assert text_utils.chunk_text("   ", chunk_size=10, overlap=2) == []


def test_tokenize_ignora_stopwords_y_numeros():
    tokens = text_utils.tokenize("El gato y la casa 123 azul")
    assert tokens == ["gato", "casa", "azul"]
