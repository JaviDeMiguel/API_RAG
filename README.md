# Asistente Virtual para Documentos (RAG)

API construida con **FastAPI** que implementa un sistema **RAG** (Retrieval-Augmented
Generation): te registras, subes un texto largo o un PDF y le haces preguntas en
lenguaje natural. Los fragmentos relevantes se recuperan por **similitud de
embeddings** y las respuestas las genera **Claude** (Anthropic) usando únicamente
ese contexto. Cada usuario solo accede a sus propios documentos.

## Características

- **Autenticación** con **JWT** (flujo OAuth2 *password*): registro, login y
  rutas protegidas. Cada documento pertenece a su usuario.
- **Persistencia en SQLite** (biblioteca estándar `sqlite3`, sin ORM pesado).
- Ingesta de documentos desde **texto plano** o **PDF**.
- **Fragmentación** (chunking) con solapamiento configurable.
- **Recuperación por embeddings**: cada fragmento se vectoriza y la relevancia se
  mide por **similitud de coseno**. El proveedor de embeddings está detrás de una
  interfaz (`EmbeddingProvider`); el de por defecto es **local y determinista**
  (sin claves ni descargas), y es intercambiable por uno semántico (Voyage AI,
  sentence-transformers) sin tocar el resto de la app.
- **Generación** de respuestas con Claude (`claude-opus-4-8`), restringida al
  contexto recuperado para reducir alucinaciones.
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
│   └── questions.py            # Preguntas sobre un documento (RAG)
├── services/                   # Lógica de negocio
│   ├── account_service.py      # Registro y autenticación
│   ├── auth_service.py         # Hashing (PBKDF2) y tokens JWT
│   ├── document_service.py     # Orquesta ingesta + recuperación + generación
│   ├── embedding_service.py    # Embeddings + similitud de coseno
│   ├── llm_service.py          # Acceso a la API de Claude (Anthropic)
│   └── text_utils.py           # Tokenización y fragmentación
└── repositories/               # Acceso a datos (SQLite)
    ├── document_repository.py  # Documentos y fragmentos (con embeddings)
    └── user_repository.py      # Usuarios
```

El flujo RAG es: **ingesta** (fragmentar → vectorizar → guardar) →
**recuperación** (embeber la pregunta y elegir los `top_k` fragmentos más
similares por coseno) → **generación** (Claude responde con ese contexto).

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
| GET    | `/health`               |  —   | Comprobación de estado               |

## Configuración

Todas las opciones se ajustan por variables de entorno o `.env` (ver
`.env.example`): `ANTHROPIC_API_KEY`, `ANTHROPIC_MODEL`, `DB_PATH`, `JWT_SECRET`,
`ACCESS_TOKEN_EXPIRE_MINUTES`, `CHUNK_SIZE`, `CHUNK_OVERLAP`, `TOP_K`,
`EMBEDDING_DIM`.

## Tests

Los tests usan **mocks del LLM**: nunca llaman a la API real de Anthropic, así
que no consumen cuota ni requieren clave. La base de datos se ejecuta en memoria.

```powershell
pip install -r requirements-dev.txt
pytest
```

Cobertura: utilidades de texto, embeddings, hashing/JWT, `LLMService` (con el
cliente de Anthropic mockeado) y los endpoints de extremo a extremo
(autenticación, flujo RAG, aislamiento entre usuarios, validación).

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
privilegios, expone el puerto 8000 e incluye un `HEALTHCHECK` sobre `/health`. La
base de datos SQLite se guarda en el volumen montado en `/app/data`.

## Notas de diseño

- **Contraseñas**: se almacenan hasheadas con PBKDF2-HMAC-SHA256 (biblioteca
  estándar), con sal aleatoria por usuario. Nunca se guardan en claro.
- **Persistencia**: SQLite (`DB_PATH`). El acceso a datos está aislado en
  `repositories/`, por lo que migrar a PostgreSQL u otro motor solo afecta a esa
  capa. Usa `DB_PATH=:memory:` para una base efímera (útil en tests).
- **Embeddings**: el proveedor local usa *feature hashing* de palabras y
  n-gramas de caracteres (determinista, sin conexión). Para producción se
  sustituiría por embeddings semánticos implementando la interfaz
  `EmbeddingProvider` en `embedding_service.py`; el resto de la app no cambia.
- En producción, define `JWT_SECRET` con un valor aleatorio de ≥32 bytes y sirve
  la API siempre sobre HTTPS.
