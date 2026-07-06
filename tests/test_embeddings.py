"""Pruebas del proveedor de embeddings y la similitud de coseno."""

import math

import pytest

from app.config import Settings
from app.services.embedding_service import (
    EmbeddingConfigurationError,
    LocalHashingEmbedding,
    VoyageEmbedding,
    cosine_similarity,
    get_embedding_provider,
)

# --- Proveedor local -------------------------------------------------------


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


def test_embed_query_y_documents_son_coherentes():
    emb = LocalHashingEmbedding(dim=128)
    texto = "energía solar de las plantas"
    # En el proveedor local, consulta y documento son equivalentes.
    assert emb.embed_query(texto) == emb.embed(texto)
    docs = ["primer fragmento", "segundo fragmento"]
    vectores = emb.embed_documents(docs)
    assert len(vectores) == 2
    assert vectores[0] == emb.embed(docs[0])


# --- Similitud de coseno ---------------------------------------------------


def test_cosine_similarity_dimensiones_distintas_lanza_error():
    with pytest.raises(ValueError):
        cosine_similarity([0.1, 0.2, 0.3], [0.1, 0.2])


# --- Selección de proveedor ------------------------------------------------


def test_get_embedding_provider_local_por_defecto():
    provider = get_embedding_provider(Settings(embedding_provider="local"))
    assert isinstance(provider, LocalHashingEmbedding)


def test_get_embedding_provider_voyage():
    provider = get_embedding_provider(
        Settings(embedding_provider="voyage", voyage_api_key="pa-test")
    )
    assert isinstance(provider, VoyageEmbedding)


def test_get_embedding_provider_voyage_sin_clave_lanza_error():
    with pytest.raises(EmbeddingConfigurationError):
        get_embedding_provider(Settings(embedding_provider="voyage"))


def test_get_embedding_provider_desconocido_lanza_error():
    with pytest.raises(EmbeddingConfigurationError):
        get_embedding_provider(Settings(embedding_provider="otro"))


# --- Proveedor Voyage (con cliente falso, sin red) -------------------------


class _FakeVoyageResult:
    def __init__(self, embeddings):
        self.embeddings = embeddings


class _FakeVoyageClient:
    """Registra las llamadas y devuelve vectores sin normalizar."""

    def __init__(self):
        self.calls = []

    def embed(self, texts, model, input_type, output_dimension=None):
        self.calls.append(
            {
                "texts": list(texts),
                "model": model,
                "input_type": input_type,
                "output_dimension": output_dimension,
            }
        )
        # Vectores deliberadamente sin normalizar para comprobar la normalización.
        return _FakeVoyageResult([[3.0, 0.0, 4.0] for _ in texts])


def _voyage_con_cliente_falso():
    emb = VoyageEmbedding(api_key="pa-test", model="voyage-3.5")
    fake = _FakeVoyageClient()
    emb._client = fake  # inyectamos el cliente para no tocar la red
    return emb, fake


def test_voyage_normaliza_los_vectores():
    emb, _ = _voyage_con_cliente_falso()
    vector = emb.embed_query("una pregunta")
    norm = math.sqrt(sum(v * v for v in vector))
    assert abs(norm - 1.0) < 1e-6
    assert vector == pytest.approx([0.6, 0.0, 0.8])


def test_voyage_usa_input_type_document_para_documentos():
    emb, fake = _voyage_con_cliente_falso()
    vectores = emb.embed_documents(["a", "b"])
    assert len(vectores) == 2
    assert fake.calls[0]["input_type"] == "document"


def test_voyage_usa_input_type_query_para_consultas():
    emb, fake = _voyage_con_cliente_falso()
    emb.embed_query("una pregunta")
    assert fake.calls[0]["input_type"] == "query"


def test_voyage_documentos_vacios_no_llama_a_la_api():
    emb, fake = _voyage_con_cliente_falso()
    assert emb.embed_documents([]) == []
    assert fake.calls == []
