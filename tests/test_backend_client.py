"""Cliente do backend: retry, erros de negócio, circuit breaker e contexto."""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from app.adapters import backend
from app.shared import context


@pytest.fixture
def transport(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Substitui a rede por uma fila de respostas. Devolve (fila, requisições)."""
    queue: list[Any] = []
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        outcome = queue.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        status, body = outcome
        return httpx.Response(status, json=body)

    client = httpx.Client(base_url="http://backend.test",
                          transport=httpx.MockTransport(handler))
    monkeypatch.setattr(backend, "_client", client)
    monkeypatch.setattr(backend.time, "sleep", lambda _s: None)
    return [queue, seen]


def test_retry_em_503_e_depois_sucesso(transport: list[Any]) -> None:
    queue, seen = transport
    queue += [(503, {}), (503, {}), (200, {"count": 0, "products": []})]
    assert backend.search_products("x")["count"] == 0
    assert len(seen) == 3


def test_erro_de_negocio_nao_e_retentado(transport: list[Any]) -> None:
    queue, seen = transport
    queue += [(409, {"detail": "cotação expirada, crie uma nova"})]
    with pytest.raises(backend.BackendError) as err:
        backend.place_order("q", "c", "Ana")
    assert err.value.status == 409
    assert len(seen) == 1


def test_erro_de_validacao_vira_texto(transport: list[Any]) -> None:
    queue, _ = transport
    queue += [(422, {"detail": [{"msg": "quantity deve ser >= 1"}]})]
    with pytest.raises(backend.BackendError) as err:
        backend.create_quote("c", [{"product_id": "BK001", "quantity": 0}])
    assert "quantity deve ser >= 1" in err.value.detail


def test_falha_de_conexao_esgota_e_levanta_indisponivel(transport: list[Any]) -> None:
    queue, seen = transport
    queue += [httpx.ConnectError("recusada")] * 3
    with pytest.raises(backend.BackendUnavailable):
        backend.get_product("BK001")
    assert len(seen) == 3


def test_circuito_abre_e_para_de_chamar(transport: list[Any]) -> None:
    queue, seen = transport
    queue += [httpx.ConnectError("recusada")] * (3 * backend.BREAKER_THRESHOLD)
    for _ in range(backend.BREAKER_THRESHOLD):
        with pytest.raises(backend.BackendUnavailable):
            backend.get_product("BK001")
    calls = len(seen)
    with pytest.raises(backend.BackendUnavailable, match="circuito aberto"):
        backend.get_product("BK001")
    assert len(seen) == calls  # nem tentou: falha rápida


def test_propaga_correlation_id(transport: list[Any]) -> None:
    queue, seen = transport
    queue += [(200, {"product_id": "BK001"})]
    tokens = context.bind(correlation_id="corr-123", conversation_id="conv-9")
    try:
        backend.get_product("BK001")
    finally:
        context.unbind(tokens)
    assert seen[0].headers["X-Request-Id"] == "corr-123"
    assert seen[0].headers["X-Conversation-Id"] == "conv-9"
