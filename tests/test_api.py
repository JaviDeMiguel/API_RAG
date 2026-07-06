"""Pruebas de extremo a extremo de la API (con el LLM mockeado).

Ejercitan los endpoints reales —autenticación, ingesta, recuperación por
embeddings, búsqueda global, streaming y aislamiento entre usuarios— sin llamar
a la API del LLM.
"""

import json


def _crear_documento(client, headers, title="Apuntes", content=None) -> str:
    texto = content or (
        "La fotosintesis permite a las plantas producir energia a partir de la luz. "
        "El interes compuesto hace crecer el capital con el tiempo. "
        "Los volcanes expulsan lava y ceniza durante una erupcion. "
    ) * 4
    resp = client.post(
        "/documents",
        json={"title": title, "content": texto},
        headers=headers,
    )
    assert resp.status_code == 201
    return resp.json()["id"]


def _sse_eventos(texto: str) -> list[dict]:
    """Parsea el cuerpo de una respuesta SSE en una lista de objetos JSON."""
    return [
        json.loads(linea[len("data: ") :])
        for linea in texto.splitlines()
        if linea.startswith("data: ")
    ]


def test_documents_requiere_autenticacion(client):
    assert client.get("/documents").status_code == 401


def test_registro_login_y_perfil(client):
    creds = {"username": "alice", "password": "clave-segura-1"}
    assert client.post("/auth/register", json=creds).status_code == 201

    token = client.post("/auth/token", data=creds).json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    me = client.get("/auth/me", headers=headers)
    assert me.status_code == 200
    assert me.json()["username"] == "alice"


def test_login_con_credenciales_incorrectas(client):
    client.post(
        "/auth/register", json={"username": "bob", "password": "clave-segura-2"}
    )
    resp = client.post(
        "/auth/token", data={"username": "bob", "password": "incorrecta"}
    )
    assert resp.status_code == 401


def test_registro_duplicado_devuelve_conflicto(client):
    creds = {"username": "carol", "password": "clave-segura-3"}
    assert client.post("/auth/register", json=creds).status_code == 201
    assert client.post("/auth/register", json=creds).status_code == 409


def test_flujo_rag_completo(client, auth_headers, llm):
    doc_id = _crear_documento(client, auth_headers)

    resp = client.post(
        f"/documents/{doc_id}/ask",
        json={"question": "¿Como obtienen energia las plantas?"},
        headers=auth_headers,
    )
    assert resp.status_code == 200
    data = resp.json()

    # La respuesta proviene del LLM mockeado (no de la API real).
    assert data["answer"] == "RESPUESTA-SIMULADA-DEL-LLM"
    assert data["document_id"] == doc_id
    assert data["sources"]  # la recuperación devolvió fragmentos

    # El LLM recibió como contexto los fragmentos recuperados.
    assert len(llm.calls) == 1
    assert llm.calls[0]["question"] == "¿Como obtienen energia las plantas?"
    assert llm.calls[0]["context"]


def test_aislamiento_entre_usuarios(client, auth_headers):
    doc_id = _crear_documento(client, auth_headers)

    # Un segundo usuario no ve ni accede al documento del primero.
    client.post("/auth/register", json={"username": "eve", "password": "clave-segura-4"})
    token = client.post(
        "/auth/token", data={"username": "eve", "password": "clave-segura-4"}
    ).json()["access_token"]
    otros = {"Authorization": f"Bearer {token}"}

    assert client.get("/documents", headers=otros).json() == []
    assert client.get(f"/documents/{doc_id}", headers=otros).status_code == 404
    assert (
        client.post(
            f"/documents/{doc_id}/ask",
            json={"question": "cualquier cosa"},
            headers=otros,
        ).status_code
        == 404
    )


def test_validacion_pydantic(client, auth_headers):
    # Contraseña demasiado corta al registrar.
    corta = client.post(
        "/auth/register", json={"username": "dan", "password": "corta"}
    )
    assert corta.status_code == 422

    # Pregunta demasiado corta.
    doc_id = _crear_documento(client, auth_headers)
    resp = client.post(
        f"/documents/{doc_id}/ask", json={"question": "ab"}, headers=auth_headers
    )
    assert resp.status_code == 422


def test_ask_stream_devuelve_sse_con_fuentes_y_texto(client, auth_headers, llm):
    doc_id = _crear_documento(client, auth_headers)

    resp = client.post(
        f"/documents/{doc_id}/ask/stream",
        json={"question": "¿Como obtienen energia las plantas?"},
        headers=auth_headers,
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")

    eventos = _sse_eventos(resp.text)
    # Primero las fuentes, al final 'done'.
    assert eventos[0]["type"] == "sources"
    assert eventos[0]["sources"]  # la recuperación devolvió fragmentos
    assert eventos[-1]["type"] == "done"

    # El texto reconstruido a partir de los tokens coincide con el del LLM.
    texto = "".join(e["text"] for e in eventos if e["type"] == "token")
    assert texto == "RESPUESTA-SIMULADA-DEL-LLM"

    # Se usó la vía de streaming del LLM.
    assert llm.calls[0].get("stream") is True


def test_ask_stream_documento_inexistente(client, auth_headers):
    resp = client.post(
        "/documents/inexistente/ask/stream",
        json={"question": "cualquier cosa"},
        headers=auth_headers,
    )
    assert resp.status_code == 404


def test_busqueda_global_entre_documentos(client, auth_headers):
    plantas = _crear_documento(
        client,
        auth_headers,
        title="Botanica",
        content=(
            "La fotosintesis permite a las plantas producir energia con la luz. "
            "Las hojas captan la luz solar para crecer. "
        ) * 4,
    )
    _crear_documento(
        client,
        auth_headers,
        title="Finanzas",
        content=("El interes compuesto hace crecer el capital ahorrado. ") * 4,
    )

    resp = client.post(
        "/search",
        json={"query": "energia de las plantas y fotosintesis", "top_k": 3},
        headers=auth_headers,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["query"] == "energia de las plantas y fotosintesis"
    assert data["results"], "la búsqueda debería devolver resultados"

    top = data["results"][0]
    # El fragmento más relevante procede del documento de botánica.
    assert top["document_id"] == plantas
    assert top["title"] == "Botanica"
    assert "score" in top and "text" in top


def test_busqueda_requiere_autenticacion(client):
    assert client.post("/search", json={"query": "algo relevante"}).status_code == 401


def test_busqueda_aislada_por_usuario(client, auth_headers):
    _crear_documento(client, auth_headers)

    client.post("/auth/register", json={"username": "frank", "password": "clave-segura-5"})
    token = client.post(
        "/auth/token", data={"username": "frank", "password": "clave-segura-5"}
    ).json()["access_token"]
    otros = {"Authorization": f"Bearer {token}"}

    # El segundo usuario no encuentra fragmentos de los documentos del primero.
    resp = client.post(
        "/search", json={"query": "fotosintesis plantas energia"}, headers=otros
    )
    assert resp.status_code == 200
    assert resp.json()["results"] == []


def test_ciclo_de_vida_del_documento(client, auth_headers):
    doc_id = _crear_documento(client, auth_headers)

    assert len(client.get("/documents", headers=auth_headers).json()) == 1
    assert client.get(f"/documents/{doc_id}", headers=auth_headers).status_code == 200
    assert client.delete(f"/documents/{doc_id}", headers=auth_headers).status_code == 204
    assert len(client.get("/documents", headers=auth_headers).json()) == 0
    # Borrar de nuevo -> 404.
    assert client.delete(f"/documents/{doc_id}", headers=auth_headers).status_code == 404
