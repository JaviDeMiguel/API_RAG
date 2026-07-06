"""Almacén vectorial de fragmentos basado en ChromaDB.

ChromaDB actúa como **base de datos vectorial**: guarda cada fragmento junto a
su embedding y realiza la búsqueda por vecinos más cercanos con un índice HNSW
(distancia de coseno). Sustituye al scan lineal en Python sobre embeddings
serializados en SQLite.

Nosotros seguimos calculando los embeddings con nuestros proveedores
(`EmbeddingProvider`: local o Voyage); Chroma se usa como puro almacén + índice,
por lo que los vectores se pasan ya calculados (`embeddings=` / `query_embeddings=`)
y **no** se delega el cálculo a Chroma.

Cada fragmento se guarda con metadatos `{document_id, user_id, index}`, de modo
que las consultas se acotan por documento con un filtro `where`. Los metadatos
relacionales (usuarios, títulos, propiedad) siguen en SQLite; aquí solo viven
los vectores y el texto de los fragmentos.
"""

from dataclasses import dataclass

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import Settings, get_settings
from app.models.schemas import SourceChunk

# Desactivamos la telemetría anónima de Chroma (evita conexiones en segundo plano).
_CHROMA_SETTINGS = ChromaSettings(anonymized_telemetry=False)


@dataclass
class StoredChunk:
    """Fragmento de un documento con su vector de embedding."""

    index: int
    text: str
    embedding: list[float]


@dataclass
class UserSearchHit:
    """Fragmento recuperado en una búsqueda entre documentos del usuario."""

    document_id: str
    index: int
    text: str
    score: float


class ChunkVectorStore:
    """Operaciones vectoriales sobre los fragmentos (respaldadas por Chroma)."""

    def __init__(self, settings: Settings) -> None:
        self._name = settings.chroma_collection
        if settings.chroma_path == ":memory:":
            self._client = chromadb.EphemeralClient(settings=_CHROMA_SETTINGS)
        else:
            self._client = chromadb.PersistentClient(
                path=settings.chroma_path, settings=_CHROMA_SETTINGS
            )
        self._collection = self._get_or_create()

    def _get_or_create(self):
        # Espacio de coseno: como los vectores están normalizados (norma L2 = 1),
        # es equivalente al producto escalar y coherente con el resto de la app.
        return self._client.get_or_create_collection(
            name=self._name,
            configuration={"hnsw": {"space": "cosine"}},
        )

    def add(
        self, document_id: str, user_id: str, chunks: list[StoredChunk]
    ) -> None:
        """Indexa los fragmentos de un documento con sus embeddings."""
        if not chunks:
            return
        self._collection.add(
            ids=[f"{document_id}:{chunk.index}" for chunk in chunks],
            embeddings=[chunk.embedding for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            metadatas=[
                {
                    "document_id": document_id,
                    "user_id": user_id,
                    "index": chunk.index,
                }
                for chunk in chunks
            ],
        )

    def query(
        self, document_id: str, query_embedding: list[float], top_k: int
    ) -> list[SourceChunk]:
        """Devuelve los `top_k` fragmentos más similares del documento.

        La similitud se expresa como `score = 1 - distancia_coseno`, de modo que
        1.0 es idéntico y valores menores indican menor relevancia (igual que la
        similitud de coseno que devolvía la implementación anterior).
        """
        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where={"document_id": document_id},
        )

        ids = result["ids"][0]
        if not ids:
            return []

        documents = result["documents"][0]
        metadatas = result["metadatas"][0]
        distances = result["distances"][0]
        return [
            SourceChunk(
                index=int(meta["index"]),
                text=text,
                score=1.0 - float(distance),
            )
            for text, meta, distance in zip(documents, metadatas, distances)
        ]

    def query_user(
        self, user_id: str, query_embedding: list[float], top_k: int
    ) -> list[UserSearchHit]:
        """Devuelve los `top_k` fragmentos más similares entre TODOS los
        documentos del usuario (búsqueda global acotada por `user_id`)."""
        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where={"user_id": user_id},
        )

        ids = result["ids"][0]
        if not ids:
            return []

        documents = result["documents"][0]
        metadatas = result["metadatas"][0]
        distances = result["distances"][0]
        return [
            UserSearchHit(
                document_id=str(meta["document_id"]),
                index=int(meta["index"]),
                text=text,
                score=1.0 - float(distance),
            )
            for text, meta, distance in zip(documents, metadatas, distances)
        ]

    def delete(self, document_id: str) -> None:
        """Elimina todos los fragmentos de un documento."""
        self._collection.delete(where={"document_id": document_id})

    def reset(self) -> None:
        """Vacía la colección (recreándola). Útil para aislar tests."""
        self._client.delete_collection(self._name)
        self._collection = self._get_or_create()


# Instancia única compartida por toda la aplicación.
_vector_store: ChunkVectorStore | None = None


def get_vector_store() -> ChunkVectorStore:
    """Dependencia de FastAPI que expone el almacén vectorial compartido."""
    global _vector_store
    if _vector_store is None:
        _vector_store = ChunkVectorStore(get_settings())
    return _vector_store
