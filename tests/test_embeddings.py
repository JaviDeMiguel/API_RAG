"""Pruebas del proveedor de embeddings y la similitud de coseno."""

import math

from app.services.embedding_service import (
    LocalHashingEmbedding,
    cosine_similarity,
)


def test_embedding_esta_normalizado():
    emb = LocalHashingEmbedding(dim=256)
    vector = emb.embed("las plantas realizan la fotosíntesis")
    norm = math.sqrt(sum(v * v for v in vector))
    assert vector != [0.0] * 256
    assert abs(norm - 1.0) < 1e-6


def test_texto_relevante_tiene_mayor_similitud():
    emb = LocalHashingEmbedding(dim=512)
    pregunta = emb.embed("¿qué es la fotosíntesis de las plantas?")
    relevante = emb.embed("La fotosíntesis permite a las plantas producir energía")
    irrelevante = emb.embed("El coche rojo circula por la carretera nacional")

    assert cosine_similarity(pregunta, relevante) > cosine_similarity(
        pregunta, irrelevante
    )


def test_texto_vacio_devuelve_vector_cero():
    emb = LocalHashingEmbedding(dim=64)
    assert emb.embed("   ") == [0.0] * 64
