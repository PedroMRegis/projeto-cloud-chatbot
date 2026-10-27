"""Modo LLM: estado da sessão no prompt, sincronização e degradação."""

from __future__ import annotations

import json
from typing import Any

import pytest

from app.adapters import llm
from app.application import agent
from tests.conftest import Chat, FakeBackend


class Call:
    def __init__(self, name: str, arguments: dict[str, Any]) -> None:
        self.id = f"call-{name}"
        self.function = type("F", (), {"name": name, "arguments": json.dumps(arguments)})()


def response(content: str | None = None, calls: list[Call] | None = None) -> Any:
    message = type("M", (), {"content": content, "tool_calls": calls})()
    return type("R", (), {"choices": [type("C", (), {"message": message})()]})()


class ScriptedLLM:
    """Modelo roteirizado: devolve as respostas em ordem e guarda o que recebeu."""

    def __init__(self) -> None:
        self.script: list[Any] = []
        self.seen: list[list[dict[str, Any]]] = []

    def complete(self, messages: list[dict[str, Any]], tools: Any = None) -> Any:
        self.seen.append(list(messages))
        return self.script.pop(0)


@pytest.fixture
def llm_mode(monkeypatch: pytest.MonkeyPatch) -> ScriptedLLM:
    from app.shared.config import reset_settings_cache

    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "fake")
    reset_settings_cache()
    model = ScriptedLLM()
    monkeypatch.setattr(agent.llm, "complete", model.complete)
    return model


def test_llm_recebe_o_estado_da_sessao(chat: Chat, llm_mode: ScriptedLLM,
                                       fake_backend: FakeBackend) -> None:
    llm_mode.script += [
        response(calls=[Call("search_products", {"query": "dom casmurro"})]),
        response(calls=[Call("get_product", {"product_id": "BK001"})]),
        response("Temos Dom Casmurro por R$ 39,90. Quer levar?"),
        response("Claro! Quantos exemplares?"),
    ]
    r = chat.say("vocês têm dom casmurro?")
    assert r["stage"] == "product"
    assert r["products"][0]["product_id"] == "BK001"
    assert "Quero 1 exemplar" in [s["label"] for s in r["suggestions"]]

    chat.say("sim")
    # No 2º turno o modelo recebe o estado: sabe que "sim" é sobre o BK001.
    context = llm_mode.seen[-1][1]["content"]
    assert "livro em foco: Dom Casmurro (product_id=BK001" in context


def test_cotacao_feita_pelo_llm_vira_botao_de_confirmar(
    chat: Chat, llm_mode: ScriptedLLM, fake_backend: FakeBackend
) -> None:
    llm_mode.script += [
        response(calls=[Call("create_quote", {"items": [{"product_id": "BK016",
                                                         "quantity": 3}]})]),
        response("Cotação pronta. Confirma?"),
    ]
    r = chat.say("quero 3 do 1984")
    assert r["stage"] == "quoted"
    # O botão é tratado pelo vendedor guiado: funciona mesmo que o LLM caia depois.
    r = chat.click("Confirmar pedido")
    assert r["stage"] == "awaiting_name"


def test_llm_fora_degrada_mas_mantem_os_botoes(
    chat: Chat, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.shared.config import reset_settings_cache

    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    reset_settings_cache()

    def down(*_a: Any, **_k: Any) -> Any:
        raise llm.LLMUnavailable("tudo fora")

    monkeypatch.setattr(agent.llm, "complete", down)
    r = chat.say("quero um livro")
    assert r["degraded"] is True
    assert r["reply"] == agent.DEGRADED_REPLY
    assert "Clássicos" in [s["label"] for s in r["suggestions"]]
    # E a compra segue pelos botões, sem LLM.
    r = chat.click("Clássicos")
    assert r["stage"] == "browsing"
