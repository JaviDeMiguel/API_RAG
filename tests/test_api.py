"""Pruebas de extremo a extremo de la API (con el LLM mockeado).

Ejercitan los endpoints reales —autenticación, ingesta, recuperación por
embeddings y aislamiento entre usuarios— sin llamar a la API del LLM.
"""


def _crear_documento(client, headers) -> str:
    texto = (
        "La fotosintesis permite a las plantas producir energia a partir de la luz. "
        "El interes compuesto hace crecer el capital con el tiempo. "
        "Los volcanes expulsan lava y ceniza durante una erupcion. "
    ) * 4
    resp = client.post(
        "/documents",
        json={"title": "Apuntes", "content": texto},
        headers=headers,
    )
    assert resp.status_code == 201
    return resp.json()["id"]


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


def test_ciclo_de_vida_del_documento(client, auth_headers):
    doc_id = _crear_documento(client, auth_headers)

    assert len(client.get("/documents", headers=auth_headers).json()) == 1
    assert client.get(f"/documents/{doc_id}", headers=auth_headers).status_code == 200
    assert client.delete(f"/documents/{doc_id}", headers=auth_headers).status_code == 204
    assert len(client.get("/documents", headers=auth_headers).json()) == 0
    # Borrar de nuevo -> 404.
    assert client.delete(f"/documents/{doc_id}", headers=auth_headers).status_code == 404
