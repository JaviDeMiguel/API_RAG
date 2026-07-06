"""Pruebas del almacén vectorial (ChromaDB) sobre una instancia en memoria."""

import pytest

from app.config import Settings
from app.repositories.vector_store import ChunkVectorStore, StoredChunk


@pytest.fixture
def store() -> ChunkVectorStore:
    """Almacén vectorial efímero y aislado por test."""
    s = ChunkVectorStore(Settings(chroma_path=":memory:", chroma_collection="chunks"))
    s.reset()
    return s


def _chunk(index: int, text: str, embedding: list[float]) -> StoredChunk:
    return StoredChunk(index=index, text=text, embedding=embedding)


def test_query_ordena_por_similitud_y_calcula_score(store):
    store.add(
        "doc1",
        "user1",
        [
            _chunk(0, "vector alineado con la consulta", [1.0, 0.0]),
            _chunk(1, "vector ortogonal a la consulta", [0.0, 1.0]),
        ],
    )

    results = store.query("doc1", [1.0, 0.0], top_k=2)

    # El más similar primero; score = 1 - distancia_coseno.
    assert [r.index for r in results] == [0, 1]
    assert results[0].score == pytest.approx(1.0)
    assert results[1].score == pytest.approx(0.0)
    assert results[0].text == "vector alineado con la consulta"


def test_query_respeta_top_k(store):
    store.add(
        "doc1",
        "user1",
        [_chunk(i, f"fragmento {i}", [1.0, float(i)]) for i in range(5)],
    )
    assert len(store.query("doc1", [1.0, 0.0], top_k=3)) == 3


def test_query_esta_acotada_por_documento(store):
    store.add("doc1", "user1", [_chunk(0, "de doc1", [1.0, 0.0])])
    store.add("doc2", "user1", [_chunk(0, "de doc2", [1.0, 0.0])])

    results = store.query("doc2", [1.0, 0.0], top_k=5)
    assert len(results) == 1
    assert results[0].text == "de doc2"


def test_delete_elimina_solo_su_documento(store):
    store.add("doc1", "user1", [_chunk(0, "de doc1", [1.0, 0.0])])
    store.add("doc2", "user1", [_chunk(0, "de doc2", [0.0, 1.0])])

    store.delete("doc1")

    assert store.query("doc1", [1.0, 0.0], top_k=5) == []
    assert len(store.query("doc2", [0.0, 1.0], top_k=5)) == 1


def test_query_documento_inexistente_devuelve_vacio(store):
    assert store.query("no-existe", [1.0, 0.0], top_k=5) == []


def test_add_sin_fragmentos_no_falla(store):
    store.add("doc1", "user1", [])
    assert store.query("doc1", [1.0, 0.0], top_k=5) == []
