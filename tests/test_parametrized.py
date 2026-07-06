"""Tests parametrizados de las funciones puras del núcleo.

Demuestran `@pytest.mark.parametrize`: un mismo test se ejecuta con múltiples
juegos de datos, cubriendo varios casos límite con poco código.
"""

import pytest

from app.services.auth_service import hash_password, verify_password
from app.services.embedding_service import LocalHashingEmbedding, cosine_similarity
from app.services.text_utils import chunk_text, tokenize


@pytest.mark.parametrize(
    "texto, esperado",
    [
        ("El gato y la casa", ["gato", "casa"]),        # elimina stopwords
        ("PLANTAS Verdes", ["plantas", "verdes"]),      # normaliza a minúsculas
        ("agua 2024 sol", ["agua", "sol"]),             # descarta números
        ("de la a el", []),                             # solo stopwords -> vacío
    ],
)
def test_tokenize_casos(texto, esperado):
    assert tokenize(texto) == esperado


@pytest.mark.parametrize(
    "n_palabras, tam, solapamiento, n_chunks",
    [
        (10, 10, 0, 1),    # cabe en un solo fragmento
        (20, 10, 0, 2),    # dos fragmentos exactos
        (100, 30, 10, 5),  # paso=20 -> inicios en 0,20,40,60,80
    ],
)
def test_chunk_text_numero_de_fragmentos(n_palabras, tam, solapamiento, n_chunks):
    texto = " ".join(f"w{i}" for i in range(n_palabras))
    assert len(chunk_text(texto, chunk_size=tam, overlap=solapamiento)) == n_chunks


@pytest.mark.parametrize(
    "password",
    [
        "abc12345",
        "ñÑüü-Pass-9",
        "una frase larga con espacios 123",
        "P@ssw0rd!#$%",
    ],
)
def test_password_roundtrip(password):
    hashed = hash_password(password)
    assert verify_password(password, hashed)
    assert not verify_password(password + "x", hashed)


@pytest.mark.parametrize(
    "a, b, esperado",
    [
        ([1.0, 0.0], [1.0, 0.0], 1.0),   # vectores idénticos
        ([1.0, 0.0], [0.0, 1.0], 0.0),   # ortogonales
        ([0.6, 0.8], [0.6, 0.8], 1.0),   # unitarios idénticos
    ],
)
def test_cosine_similarity_casos(a, b, esperado):
    assert cosine_similarity(a, b) == pytest.approx(esperado)


@pytest.mark.parametrize("dim", [64, 128, 256, 512])
def test_embedding_normalizado_para_varias_dimensiones(dim):
    vector = LocalHashingEmbedding(dim=dim).embed("las plantas hacen fotosíntesis")
    assert len(vector) == dim
    norma = sum(v * v for v in vector) ** 0.5
    assert norma == pytest.approx(1.0, abs=1e-6)
