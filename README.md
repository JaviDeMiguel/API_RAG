# Asistente Virtual para Documentos (RAG)

[![CI](https://github.com/JaviDeMiguel/API_RAG/actions/workflows/ci.yml/badge.svg)](https://github.com/JaviDeMiguel/API_RAG/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/JaviDeMiguel/API_RAG/branch/main/graph/badge.svg)](https://codecov.io/gh/JaviDeMiguel/API_RAG)

API construida con **FastAPI** que implementa un sistema **RAG** (Retrieval-Augmented
Generation): te registras, subes un texto largo o un PDF y le haces preguntas en
lenguaje natural. Los fragmentos relevantes se recuperan por **similitud de
embeddings** y las respuestas las genera **Claude** (Anthropic) usando únicamente
ese contexto. Cada usuario solo accede a sus propios documentos.

## Características

- **Autenticación** con **JWT** (flujo OAuth2 *password*): registro, login y
  rutas protegidas. Cada documento pertenece a su usuario.
- **Persistencia híbrida**: **SQLite** (biblioteca estándar `sqlite3`, sin ORM
  pesado) para usuarios y metadatos de documentos, y **ChromaDB** como **base de
  datos vectorial** para los fragmentos y su búsqueda por similitud.
- Ingesta de documentos desde **texto plano** o **PDF**.
- **Fragmentación** (chunking) con solapamiento configurable.
- **Recuperación por embeddings**: cada fragmento se vectoriza y la relevancia se
  mide por **similitud de coseno**, resuelta por el índice **HNSW** de ChromaDB
  (búsqueda por vecinos más cercanos, sin scan lineal). El proveedor de
  embeddings está detrás de una
  interfaz (`EmbeddingProvider`) y se elige por configuración
  (`EMBEDDING_PROVIDER`):
  - **`local`** (por defecto): **determinista y sin conexión** (sin claves ni
    descargas). Usa *feature hashing*; es rápido y gratis, pero **no semántico**.
  - **`voyage`**: embeddings **semánticos** de **Voyage AI** (el partner de
    embeddings recomendado por Anthropic para usar con Claude). Entienden el
    significado, por lo que recuperan mejor aunque la pregunta use otras
    palabras. Requiere `VOYAGE_API_KEY` y el paquete `voyageai`.
  Cambiar de proveedor no toca el resto de la app.
- **Generación** de respuestas con Claude (`claude-opus-4-8`), restringida al
  contexto recuperado para reducir alucinaciones. Disponible también en
  **streaming** (Server-Sent Events) para mostrar la respuesta según se genera.
- **Búsqueda global**: recupera los fragmentos más relevantes entre **todos** los
  documentos del usuario (no solo uno), aprovechando el filtro por usuario de la
  base vectorial.
- Validación de entradas y salidas con **Pydantic**.
- Documentación interactiva automática en `/docs`.

## Arquitectura

El proyecto separa responsabilidades en capas:

```
app/
├── main.py                     # Punto de entrada FastAPI (+ lifespan)
├── config.py                   # Configuración (variables de entorno / .env)
├── db.py                       # Conexión SQLite compartida y esquema
├── security.py                 # Dependencia de autenticación (get_current_user)
├── models/
│   └── schemas.py              # Esquemas Pydantic (validación)
├── routers/                    # Rutas / endpoints HTTP
│   ├── auth.py                 # Registro, login (token) y perfil
│   ├── documents.py            # Alta, listado, consulta y borrado
│   ├── questions.py            # Preguntas sobre un documento (RAG + streaming)
│   └── search.py               # Búsqueda global entre documentos del usuario
├── services/                   # Lógica de negocio
│   ├── account_service.py      # Registro y autenticación
│   ├── auth_service.py         # Hashing (PBKDF2) y tokens JWT
│   ├── document_service.py     # Orquesta ingesta + recuperación + generación
│   ├── embedding_service.py    # Embeddings + similitud de coseno
│   ├── llm_service.py          # Acceso a la API de Claude (Anthropic)
│   └── text_utils.py           # Tokenización y fragmentación
└── repositories/               # Acceso a datos
    ├── document_repository.py  # Metadatos de documentos (SQLite)
    ├── vector_store.py         # Fragmentos + embeddings + búsqueda (ChromaDB)
    └── user_repository.py      # Usuarios (SQLite)
```

El flujo RAG es: **ingesta** (fragmentar → vectorizar → guardar en la base
vectorial) → **recuperación** (embeber la pregunta y pedir a ChromaDB los
`top_k` fragmentos más similares por coseno) → **generación** (Claude responde
con ese contexto).

## Instalación

Requiere Python 3.10+.

```powershell
# Crear y activar el entorno virtual
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Instalar dependencias
pip install -r requirements.txt

# Configurar variables de entorno
copy .env.example .env
# Edita .env: añade ANTHROPIC_API_KEY y cambia JWT_SECRET por un valor aleatorio
```

## Ejecución

```powershell
uvicorn app.main:app --reload
```

Abre la documentación interactiva en http://localhost:8000/docs
(desde ahí puedes autenticarte con el botón **Authorize**).

## Uso rápido

**1. Registrarse:**

```bash
curl -X POST http://localhost:8000/auth/register \
  -H "Content-Type: application/json" \
  -d '{"username": "alice", "password": "clave-segura-1"}'
```

**2. Obtener un token (login):**

```bash
curl -X POST http://localhost:8000/auth/token \
  -d "username=alice&password=clave-segura-1"
# -> {"access_token": "eyJ...", "token_type": "bearer"}
```

Guarda el token y envíalo en la cabecera `Authorization: Bearer <token>`.

**3. Dar de alta un documento (texto):**

```bash
curl -X POST http://localhost:8000/documents \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"title": "Notas", "content": "La fotosíntesis es el proceso por el que las plantas convierten la luz en energía química..."}'
```

**4. Subir un PDF:**

```bash
curl -X POST http://localhost:8000/documents/upload \
  -H "Authorization: Bearer $TOKEN" \
  -F "title=Informe" \
  -F "file=@informe.pdf"
```

**5. Hacer una pregunta:**

```bash
curl -X POST http://localhost:8000/documents/{id}/ask \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"question": "¿Qué es la fotosíntesis?"}'
```

Respuesta:

```json
{
  "document_id": "a1b2c3...",
  "question": "¿Qué es la fotosíntesis?",
  "answer": "Según el documento, la fotosíntesis es el proceso por el que...",
  "sources": [{ "index": 0, "text": "...", "score": 0.34 }]
}
```

**6. Preguntar con respuesta en streaming (SSE):**

```bash
curl -N -X POST http://localhost:8000/documents/{id}/ask/stream \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"question": "¿Qué es la fotosíntesis?"}'
```

Devuelve **Server-Sent Events**: primero `{"type":"sources", ...}` con los
fragmentos recuperados, luego varios `{"type":"token","text":"..."}` con el
texto según se genera y, al final, `{"type":"done"}`.

**7. Buscar en toda la biblioteca del usuario (sin generar respuesta):**

```bash
curl -X POST http://localhost:8000/search \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query": "energía de las plantas", "top_k": 5}'
```

```json
{
  "query": "energía de las plantas",
  "results": [
    { "document_id": "a1b2c3...", "title": "Botánica", "index": 0, "text": "...", "score": 0.41 }
  ]
}
```

## Endpoints

| Método | Ruta                    | Auth | Descripción                          |
| ------ | ----------------------- | :--: | ------------------------------------ |
| POST   | `/auth/register`        |  —   | Registrar un usuario                 |
| POST   | `/auth/token`           |  —   | Obtener token de acceso (login)      |
| GET    | `/auth/me`              |  ✔   | Perfil del usuario autenticado       |
| POST   | `/documents`            |  ✔   | Alta de documento desde texto        |
| POST   | `/documents/upload`     |  ✔   | Alta de documento desde PDF          |
| GET    | `/documents`            |  ✔   | Listar documentos del usuario        |
| GET    | `/documents/{id}`       |  ✔   | Metadatos de un documento            |
| DELETE | `/documents/{id}`       |  ✔   | Eliminar un documento                |
| POST   | `/documents/{id}/ask`   |  ✔   | Preguntar sobre un documento (RAG)   |
| POST   | `/documents/{id}/ask/stream` | ✔ | Preguntar con respuesta en streaming (SSE) |
| POST   | `/search`               |  ✔   | Buscar en todos los documentos del usuario |
| GET    | `/health`               |  —   | Comprobación de estado               |

## Configuración

Todas las opciones se ajustan por variables de entorno o `.env` (ver
`.env.example`): `ANTHROPIC_API_KEY`, `ANTHROPIC_MODEL`, `DB_PATH`, `JWT_SECRET`,
`ACCESS_TOKEN_EXPIRE_MINUTES`, `CHUNK_SIZE`, `CHUNK_OVERLAP`, `TOP_K`,
`EMBEDDING_PROVIDER`, `EMBEDDING_DIM`, `CHROMA_PATH`, `CHROMA_COLLECTION` y, para
el proveedor semántico, `VOYAGE_API_KEY`, `VOYAGE_MODEL`, `VOYAGE_EMBEDDING_DIM`.

### Usar embeddings semánticos (Voyage AI)

```powershell
pip install voyageai   # ya está en requirements.txt
```

En `.env`:

```
EMBEDDING_PROVIDER=voyage
VOYAGE_API_KEY=pa-...
# Opcional: VOYAGE_MODEL=voyage-3.5  ·  VOYAGE_EMBEDDING_DIM=1024
```

> **Importante:** los embeddings son específicos del proveedor y del modelo. Si
> cambias `EMBEDDING_PROVIDER` (o el modelo/dimensión) con documentos ya
> almacenados, **vuelve a subirlos**: los vectores antiguos son incompatibles y
> la consulta fallará con un error claro de dimensiones.

## Tests

Los tests usan **mocks del LLM**: nunca llaman a la API real de Anthropic, así
que no consumen cuota ni requieren clave. La base de datos se ejecuta en memoria.

```powershell
pip install -r requirements-dev.txt
pytest
```

Cobertura: utilidades de texto, embeddings (local y Voyage con cliente
mockeado), el almacén vectorial (ChromaDB en memoria), hashing/JWT, `LLMService`
(con el cliente de Anthropic mockeado) y los endpoints de extremo a extremo
(autenticación, flujo RAG, aislamiento entre usuarios, validación). ChromaDB se
ejecuta en memoria (`CHROMA_PATH=:memory:`), sin tocar disco ni red.

## Docker

```powershell
# Construir la imagen
docker build -t asistente-documentos .

# Ejecutar (pasando la clave y un secreto JWT; se persiste la BD en un volumen)
docker run -p 8000:8000 `
  -e ANTHROPIC_API_KEY=sk-ant-... `
  -e JWT_SECRET=un-secreto-largo-y-aleatorio `
  -v ${PWD}/data:/app/data `
  asistente-documentos
```

La API queda en http://localhost:8000/docs. El contenedor corre como usuario sin
privilegios, expone el puerto 8000 e incluye un `HEALTHCHECK` sobre `/health`.
Tanto la base SQLite (`/app/data/app.db`) como la base vectorial de Chroma
(`/app/data/chroma`) se guardan en el volumen montado en `/app/data`.

## Notas de diseño

- **Contraseñas**: se almacenan hasheadas con PBKDF2-HMAC-SHA256 (biblioteca
  estándar), con sal aleatoria por usuario. Nunca se guardan en claro.
- **Persistencia**: los metadatos (usuarios, documentos) van en SQLite
  (`DB_PATH`) y los fragmentos con sus embeddings en la base vectorial ChromaDB
  (`CHROMA_PATH`). El acceso a datos está aislado en `repositories/`, por lo que
  migrar a otro motor (PostgreSQL/pgvector, Qdrant…) solo afecta a esa capa. Usa
  `DB_PATH=:memory:` y `CHROMA_PATH=:memory:` para bases efímeras (útil en tests).
- **Base de datos vectorial**: ChromaDB indexa los fragmentos con un índice HNSW
  (espacio de coseno) y resuelve la búsqueda por vecinos más cercanos, en lugar
  del scan lineal en Python de la versión anterior. Los embeddings se calculan
  con nuestros proveedores (`EmbeddingProvider`); Chroma se usa como puro almacén
  vectorial e índice.
- **Embeddings**: el proveedor local usa *feature hashing* de palabras y
  n-gramas de caracteres (determinista, sin conexión). Para producción, activa
  el proveedor semántico **Voyage AI** con `EMBEDDING_PROVIDER=voyage` (o añade
  otro implementando la interfaz `EmbeddingProvider` en `embedding_service.py`);
  el resto de la app no cambia. La ingesta vectoriza los fragmentos **en lote**
  (una sola llamada) y la recuperación distingue consulta (`input_type=query`)
  de documento (`input_type=document`) para mejorar la relevancia.
- En producción, define `JWT_SECRET` con un valor aleatorio de ≥32 bytes y sirve
  la API siempre sobre HTTPS.
