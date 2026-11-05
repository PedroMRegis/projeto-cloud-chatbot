"""Provedor Bedrock: conversão de formato, thinking, recusa e fallback.

A rede é substituída no nível do transporte HTTP em `test_requisicao_real...`,
para exercitar o SDK de verdade; os demais testes usam um cliente roteirizado.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import httpx2
import pytest
from anthropic import AnthropicBedrockMantle

from app.adapters import llm
from app.application.tools import TOOL_SCHEMAS
from app.shared import config
from tests.conftest import Chat, FakeBackend


def text(value: str) -> Any:
    return SimpleNamespace(type="text", text=value)


def thinking() -> Any:
    return SimpleNamespace(type="thinking", thinking="", signature="assinatura")


def tool_use(name: str, arguments: dict[str, Any], call_id: str = "toolu_1") -> Any:
    return SimpleNamespace(type="tool_use", id=call_id, name=name, input=arguments)


def reply(*blocks: Any, stop_reason: str = "end_turn") -> Any:
    return SimpleNamespace(content=list(blocks), stop_reason=stop_reason)


class ScriptedBedrock:
    """Cliente roteirizado: devolve as respostas em ordem e guarda as requisições."""

    def __init__(self, *script: Any) -> None:
        self.script = list(script)
        self.requests: list[dict[str, Any]] = []
        self.messages = self

    def create(self, **kwargs: Any) -> Any:
        # Cópia: a lista de mensagens cresce entre as chamadas do mesmo turno.
        self.requests.append(json.loads(json.dumps(kwargs, default=lambda o: o.__dict__)))
        outcome = self.script.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@pytest.fixture
def bedrock_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "bedrock")
    config.reset_settings_cache()


def use(monkeypatch: pytest.MonkeyPatch, client: ScriptedBedrock) -> ScriptedBedrock:
    monkeypatch.setattr(llm, "_bedrock_client", lambda: client)
    monkeypatch.setattr(llm.time, "sleep", lambda _s: None)
    return client


class TestConversao:
    def test_system_vai_para_o_topo_e_tools_mudam_de_formato(self) -> None:
        system, messages = llm._to_anthropic([
            {"role": "system", "content": "regras"},
            {"role": "system", "content": "estado"},
            {"role": "user", "content": "oi"},
        ])
        assert system == "regras\n\nestado"
        assert messages == [{"role": "user", "content": "oi"}]

        tool = llm._anthropic_tools(TOOL_SCHEMAS)[0]
        assert tool["name"] == "search_products"
        assert tool["input_schema"]["required"] == ["query"]
        assert set(tool) == {"name", "description", "input_schema"}

    def test_resultados_paralelos_ficam_numa_mensagem_so(self) -> None:
        _, messages = llm._to_anthropic([
            {"role": "user", "content": "compare"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "a", "type": "function",
                 "function": {"name": "get_product", "arguments": '{"product_id": "BK001"}'}},
                {"id": "b", "type": "function",
                 "function": {"name": "get_product", "arguments": '{"product_id": "BK004"}'}},
            ]},
            {"role": "tool", "tool_call_id": "a", "content": "{}"},
            {"role": "tool", "tool_call_id": "b", "content": "{}"},
        ])
        assert [m["role"] for m in messages] == ["user", "assistant", "user"]
        assert [b["id"] for b in messages[1]["content"]] == ["a", "b"]
        assert messages[1]["content"][0]["input"] == {"product_id": "BK001"}
        assert [b["tool_use_id"] for b in messages[2]["content"]] == ["a", "b"]
        assert "tool_results" not in messages[2]

    def test_historico_nao_comeca_pelo_vendedor(self) -> None:
        _, messages = llm._to_anthropic([
            {"role": "assistant", "content": "resposta antiga"},
            {"role": "user", "content": "sim"},
        ])
        assert messages == [{"role": "user", "content": "sim"}]

    def test_resposta_vira_chat_completion_ignorando_thinking(self) -> None:
        response = llm._to_chat_completion(reply(
            thinking(), text("Vou "), text("buscar."), tool_use("search_products", {"query": "é"}),
        ))
        message = response.choices[0].message
        assert message.content == "Vou buscar."
        assert message.tool_calls[0].function.name == "search_products"
        assert json.loads(message.tool_calls[0].function.arguments) == {"query": "é"}


class TestProvedores:
    def test_bedrock_vem_antes_dos_fallbacks_com_chave(
        self, bedrock_mode: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GROQ_API_KEY", "chave")
        config.reset_settings_cache()
        assert [p.name for p in llm._providers()] == ["bedrock", "groq"]
        assert llm._providers()[0].model == "anthropic.claude-haiku-5-5"

    def test_bedrock_nao_entra_se_nao_foi_escolhido(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LLM_PROVIDER", "gemini")
        monkeypatch.setenv("GEMINI_API_KEY", "chave")
        config.reset_settings_cache()
        assert [p.name for p in llm._providers()] == ["gemini"]

    def test_nao_envia_temperature(
        self, bedrock_mode: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client = use(monkeypatch, ScriptedBedrock(reply(text("oi"))))
        llm.complete([{"role": "user", "content": "oi"}], TOOL_SCHEMAS)
        sent = client.requests[0]
        assert sent["model"] == "anthropic.claude-haiku-5-5"
        assert sent["output_config"] == {"effort": "medium"}
        assert "temperature" not in sent

    def test_erro_transitorio_tem_retry(
        self, bedrock_mode: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        failure = Exception("throttled")
        failure.status_code = 429  # type: ignore[attr-defined]
        client = use(monkeypatch, ScriptedBedrock(failure, reply(text("oi"))))
        response = llm.complete([{"role": "user", "content": "oi"}])
        assert response.choices[0].message.content == "oi"
        assert len(client.requests) == 2

    def test_recusa_nao_tem_retry_nem_abre_o_circuito(
        self, bedrock_mode: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client = use(monkeypatch, ScriptedBedrock(
            *[reply(stop_reason="refusal") for _ in range(llm.BREAKER_FAILURE_THRESHOLD)]
        ))
        for _ in range(llm.BREAKER_FAILURE_THRESHOLD):
            with pytest.raises(llm.LLMUnavailable):
                llm.complete([{"role": "user", "content": "oi"}])
        assert len(client.requests) == llm.BREAKER_FAILURE_THRESHOLD
        assert llm.breaker("bedrock").is_open() is False

    def test_fallback_nao_recebe_blocos_do_bedrock(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sent: dict[str, Any] = {}
        fake = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=lambda **kwargs: sent.update(kwargs))))
        monkeypatch.setattr(llm, "_client", lambda _provider: fake)
        provider = llm.ProviderConfig("groq", "", "chave", "modelo")
        llm._call_openai(provider, [{"role": "assistant", "content": "",
                                     "native_content": [thinking()]}], [])
        assert sent["messages"] == [{"role": "assistant", "content": ""}]


class TestTurnoCompleto:
    def test_thinking_volta_intacto_com_o_resultado_da_tool(
        self, chat: Chat, bedrock_mode: None, monkeypatch: pytest.MonkeyPatch,
        fake_backend: FakeBackend,
    ) -> None:
        client = use(monkeypatch, ScriptedBedrock(
            reply(thinking(), tool_use("search_products", {"query": "dom casmurro"}),
                  stop_reason="tool_use"),
            reply(text("Temos Dom Casmurro por R$ 39,90. Quer levar?")),
        ))
        r = chat.say("vocês têm dom casmurro?")

        assert r["reply"] == "Temos Dom Casmurro por R$ 39,90. Quer levar?"
        assert r["products"][0]["product_id"] == "BK001"
        assert r["degraded"] is False

        second = client.requests[1]
        assert "Estado da sessão" in second["system"]
        assistant, tool_result = second["messages"][-2:]
        assert [b["type"] for b in assistant["content"]] == ["thinking", "tool_use"]
        assert assistant["content"][0]["signature"] == "assinatura"
        assert tool_result["role"] == "user"
        assert tool_result["content"][0]["tool_use_id"] == "toolu_1"

    def test_bedrock_fora_e_sem_fallback_degrada_sem_5xx(
        self, chat: Chat, bedrock_mode: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        use(monkeypatch, ScriptedBedrock(*[Exception("fora")] * llm.MAX_ATTEMPTS))
        r = chat.say("quero um livro")
        assert r["degraded"] is True
        assert "Clássicos" in [s["label"] for s in r["suggestions"]]


def test_requisicao_real_passa_pelo_sdk_e_aceita_o_retorno(
    bedrock_mode: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SDK de verdade, com só o transporte HTTP falso: valida URL, corpo e parsing."""
    seen: list[httpx2.Request] = []
    bodies = [
        {"id": "msg_1", "type": "message", "role": "assistant",
         "model": "anthropic.claude-haiku-5-5", "stop_reason": "tool_use",
         "stop_sequence": None, "usage": {"input_tokens": 1, "output_tokens": 1},
         "content": [
             {"type": "thinking", "thinking": "", "signature": "assinatura"},
             {"type": "tool_use", "id": "toolu_1", "name": "search_products",
              "input": {"query": "orwell"}},
         ]},
        {"id": "msg_2", "type": "message", "role": "assistant",
         "model": "anthropic.claude-haiku-5-5", "stop_reason": "end_turn",
         "stop_sequence": None, "usage": {"input_tokens": 1, "output_tokens": 1},
         "content": [{"type": "text", "text": "Achei dois."}]},
    ]

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        return httpx2.Response(200, json=bodies.pop(0))

    client = AnthropicBedrockMantle(
        api_key="token", aws_region="us-east-2", max_retries=0,
        http_client=httpx2.Client(transport=httpx2.MockTransport(handler)),
    )
    monkeypatch.setattr(llm, "_bedrock_client", lambda: client)

    conversation: list[dict[str, Any]] = [
        {"role": "system", "content": "regras"}, {"role": "user", "content": "orwell"},
    ]
    first = llm.complete(conversation, TOOL_SCHEMAS).choices[0].message
    assert first.tool_calls[0].id == "toolu_1"

    conversation += [
        {"role": "assistant", "content": first.content, "native_content": first.native_content,
         "tool_calls": [{"id": "toolu_1", "type": "function",
                         "function": {"name": "search_products", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "toolu_1", "content": '{"count": 2}'},
    ]
    second = llm.complete(conversation, TOOL_SCHEMAS).choices[0].message
    assert second.content == "Achei dois."

    assert str(seen[0].url) == "https://bedrock-mantle.us-east-2.api.aws/anthropic/v1/messages"
    body = json.loads(seen[1].content)
    assert body["system"] == "regras"
    assert body["messages"][1]["content"][0] == {
        "type": "thinking", "thinking": "", "signature": "assinatura",
    }
    assert body["messages"][2]["content"][0]["type"] == "tool_result"
