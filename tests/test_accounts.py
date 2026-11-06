"""Conversas permanentes, histórico por conta e dono da conversa."""

from __future__ import annotations

import sys
import time
import uuid
from pathlib import Path
from typing import Any

import httpx
import jwt
import pytest

from tests.conftest import FakeBackend

SECRET = "segredo-de-teste-com-mais-de-trinta-e-dois-bytes"
ROOT = Path(__file__).resolve().parents[1]


def session_cookie(user_id: str, secret: str = SECRET, **claims: Any) -> dict[str, str]:
    token = jwt.encode(
        {"sub": user_id, "iss": "cirro-backend", "aud": "cirro-web",
         "exp": int(time.time()) + 3600, **claims},
        secret, algorithm="HS256",
    )
    return {"Cookie": f"cirro_session={token}"}


def say(api: Any, message: str, conversation_id: str | None = None,
        headers: dict[str, str] | None = None) -> Any:
    return api.post("/api/chat", headers=headers or {}, json={
        "message": message, "request_id": str(uuid.uuid4()),
        "conversation_id": conversation_id,
    })


ANA = session_cookie("conta-ana")
BIA = session_cookie("conta-bia")


def test_conversa_logada_aparece_na_lista_da_conta(api: Any, fake_backend: FakeBackend) -> None:
    first = say(api, "quero o dom casmurro", headers=ANA).json()
    say(api, "sim", first["conversation_id"], headers=ANA)
    second = say(api, "algo do tolkien", headers=ANA).json()

    listed = api.get("/api/conversations", headers=ANA).json()["conversations"]
    assert [c["conversation_id"] for c in listed] == [second["conversation_id"],
                                                      first["conversation_id"]]
    assert listed[1]["title"] == "quero o dom casmurro"
    assert listed[1]["last_message"]
    assert listed[1]["stage"] == "quoted"


def test_lista_exige_login(api: Any, fake_backend: FakeBackend) -> None:
    assert api.get("/api/conversations").status_code == 401


def test_cada_conta_ve_so_as_proprias_conversas(api: Any, fake_backend: FakeBackend) -> None:
    say(api, "quero o dom casmurro", headers=ANA)
    assert api.get("/api/conversations", headers=BIA).json()["conversations"] == []


def test_conversa_de_conta_nao_abre_para_mais_ninguem(api: Any, fake_backend: FakeBackend) -> None:
    cid = say(api, "quero o dom casmurro", headers=ANA).json()["conversation_id"]

    assert api.get(f"/api/conversations/{cid}", headers=ANA).status_code == 200
    assert api.get(f"/api/conversations/{cid}", headers=BIA).status_code == 404
    assert api.get(f"/api/conversations/{cid}").status_code == 404
    assert api.post(f"/api/conversations/{cid}/start").status_code == 404
    assert say(api, "sim", cid, headers=BIA).status_code == 404
    assert say(api, "sim", cid).status_code == 404


def test_visitante_continua_conversando(api: Any, fake_backend: FakeBackend) -> None:
    cid = say(api, "quero o dom casmurro").json()["conversation_id"]
    assert say(api, "sim", cid).status_code == 200
    assert api.get(f"/api/conversations/{cid}").status_code == 200


def test_conversa_de_visitante_passa_para_a_conta_ao_logar(
    api: Any, fake_backend: FakeBackend
) -> None:
    cid = say(api, "quero o dom casmurro").json()["conversation_id"]
    say(api, "sim", cid, headers=ANA)

    listed = api.get("/api/conversations", headers=ANA).json()["conversations"]
    assert listed[0]["conversation_id"] == cid
    assert listed[0]["title"] == "quero o dom casmurro"
    assert api.get(f"/api/conversations/{cid}").status_code == 404


def test_mensagens_e_estado_nao_expiram(api: Any, dynamo: Any, fake_backend: FakeBackend) -> None:
    cid = say(api, "quero o dom casmurro", headers=ANA).json()["conversation_id"]
    items = dynamo.query(
        KeyConditionExpression="conversation_id = :c",
        ExpressionAttributeValues={":c": cid},
    )["Items"]
    by_kind = {i["sk"].split("#")[0]: i for i in items}
    assert {"MSG", "STATE", "META", "REQ", "RATE"} <= set(by_kind)
    for kind in ("MSG", "STATE", "META"):
        assert "expires_at" not in by_kind[kind]
    for kind in ("REQ", "RATE"):
        assert "expires_at" in by_kind[kind]


@pytest.mark.parametrize("cookie", [
    session_cookie("conta-ana", secret="outro-segredo-tambem-com-mais-de-trinta-e-dois"),
    session_cookie("conta-ana", aud="outro-sistema"),
    session_cookie("conta-ana", iss="outro-emissor"),
    session_cookie("conta-ana", exp=1),
    {"Cookie": "cirro_session=lixo"},
])
def test_cookie_invalido_nao_vira_a_conta(
    api: Any, fake_backend: FakeBackend, cookie: dict[str, str]
) -> None:
    assert api.get("/api/conversations", headers=cookie).status_code == 401
    cid = say(api, "quero o dom casmurro", headers=ANA).json()["conversation_id"]
    assert api.get(f"/api/conversations/{cid}", headers=cookie).status_code == 404


@pytest.mark.parametrize("bad_id", ["USER#conta-ana", "CONV#x", "a b", "x" * 65])
def test_identificador_nao_pode_imitar_as_chaves_internas(
    api: Any, fake_backend: FakeBackend, bad_id: str
) -> None:
    assert say(api, "oi", bad_id).status_code == 422


def test_identificador_interno_tambem_falha_na_url(api: Any, fake_backend: FakeBackend) -> None:
    assert api.get("/api/conversations/USER%23conta-ana").status_code == 422


def test_sessao_segue_para_o_backend_e_nao_vai_para_os_logs() -> None:
    from app.adapters import backend
    from app.shared import context

    assert "Authorization" not in backend._headers()
    reset = context.bind_auth_token("token-do-cliente")
    try:
        assert backend._headers()["Authorization"] == "Bearer token-do-cliente"
        assert "token-do-cliente" not in context.current().values()
    finally:
        context.unbind_auth_token(reset)
    assert "Authorization" not in backend._headers()


def test_pedido_pelo_chat_envia_a_sessao_do_cliente(
    api: Any, monkeypatch: pytest.MonkeyPatch, fake_backend: FakeBackend
) -> None:
    from app.adapters import backend
    from app.shared import context

    seen: list[str | None] = []
    original = fake_backend.get_product

    def spy(product_id: str) -> dict[str, Any]:
        seen.append(context.auth_token())
        return original(product_id)

    monkeypatch.setattr(backend, "get_product", spy)
    say(api, "quero o dom casmurro", headers=ANA)
    assert seen and all(token and token in ANA["Cookie"] for token in seen)
    seen.clear()
    say(api, "quero o dom casmurro")
    assert seen == [None] * len(seen)


def test_cliente_http_envia_authorization_real() -> None:
    from app.adapters import backend
    from app.shared import context

    captured: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(request.headers)
        return httpx.Response(200, json={"categories": []})

    backend._client = httpx.Client(base_url="http://backend.test",
                                   transport=httpx.MockTransport(handler))
    reset = context.bind_auth_token("token-do-cliente")
    try:
        backend.categories()
    finally:
        context.unbind_auth_token(reset)
    assert captured["authorization"] == "Bearer token-do-cliente"


def test_script_tira_o_ttl_so_de_mensagens_e_estado(dynamo: Any) -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    from remove_message_ttl import permanent_items

    for sk in ("MSG#1", "STATE", "REQ#1", "RATE#1"):
        dynamo.put_item(Item={"conversation_id": "antiga", "sk": sk, "expires_at": 1})
    dynamo.put_item(Item={"conversation_id": "nova", "sk": "MSG#1"})

    keys = permanent_items(dynamo)
    assert sorted(k["sk"] for k in keys) == ["MSG#1", "STATE"]
    assert all(k["conversation_id"] == "antiga" for k in keys)
