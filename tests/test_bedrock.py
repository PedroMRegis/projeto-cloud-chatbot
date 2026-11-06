"""Provedor Bedrock: conversão de formato, raciocínio, recusa e fallback.

O último teste usa o client real do boto3 com o Stubber, que valida a requisição e
a resposta contra o modelo do serviço. Os demais usam um cliente roteirizado.
"""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace
from typing import Any

import boto3
import pytest
from botocore.exceptions import ClientError
from botocore.stub import Stubber

from app.adapters import llm
from app.application.tools import TOOL_SCHEMAS
from app.shared import config
from tests.conftest import Chat, FakeBackend

MODEL = "global.anthropic.claude-haiku-5-5"
REASONING = {"reasoningContent": {"reasoningText": {"text": "", "signature": "assinatura"}}}


def text(value: str) -> dict[str, Any]:
    return {"text": value}


def tool_use(name: str, arguments: dict[str, Any], call_id: str = "toolu_1") -> dict[str, Any]:
    return {"toolUse": {"toolUseId": call_id, "name": name, "input": arguments}}


def reply(*blocks: dict[str, Any], stop_reason: str = "end_turn") -> dict[str, Any]:
    return {"output": {"message": {"role": "assistant", "content": list(blocks)}},
            "stopReason": stop_reason}


def client_error(status: int, code: str) -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": code},
                        "ResponseMetadata": {"HTTPStatusCode": status}}, "Converse")


class ScriptedBedrock:
    """Cliente roteirizado: devolve as respostas em ordem e guarda as requisições."""

    def __init__(self, *script: Any) -> None:
        self.script = list(script)
        self.requests: list[dict[str, Any]] = []

    def converse(self, **kwargs: Any) -> Any:
        # Cópia: a lista de mensagens cresce entre as chamadas do mesmo turno.
        self.requests.append(copy.deepcopy(kwargs))
        outcome = self.script.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@pytest.fixture
def bedrock_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "bedrock")
    config.reset_settings_cache()


def use(monkeypatch: pytest.MonkeyPatch, client: Any) -> Any:
    monkeypatch.setattr(llm, "bedrock_client", lambda: client)
    monkeypatch.setattr(llm.time, "sleep", lambda _s: None)
    return client


class TestConversao:
    def test_system_vai_para_o_topo_e_tools_mudam_de_formato(self) -> None:
        system, messages = llm._to_converse([
            {"role": "system", "content": "regras"},
            {"role": "system", "content": "estado"},
            {"role": "user", "content": "oi"},
        ])
        assert system == [{"text": "regras"}, {"text": "estado"}]
        assert messages == [{"role": "user", "content": [{"text": "oi"}]}]

        spec = llm._converse_tools(TOOL_SCHEMAS)["tools"][0]["toolSpec"]
        assert spec["name"] == "search_products"
        assert spec["inputSchema"]["json"]["required"] == ["query"]

    def test_resultados_paralelos_ficam_numa_mensagem_so(self) -> None:
        _, messages = llm._to_converse([
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
        assert [b["toolUse"]["toolUseId"] for b in messages[1]["content"]] == ["a", "b"]
        assert messages[1]["content"][0]["toolUse"]["input"] == {"product_id": "BK001"}
        assert [b["toolResult"]["toolUseId"] for b in messages[2]["content"]] == ["a", "b"]

    def test_historico_nao_comeca_pelo_vendedor(self) -> None:
        _, messages = llm._to_converse([
            {"role": "assistant", "content": "resposta antiga"},
            {"role": "user", "content": "sim"},
        ])
        assert messages == [{"role": "user", "content": [{"text": "sim"}]}]

    def test_resposta_vira_chat_completion_ignorando_raciocinio(self) -> None:
        response = llm._to_chat_completion(reply(
            REASONING, text("Vou "), text("buscar."), tool_use("search_products", {"query": "é"}),
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
        assert llm._providers()[0].model == MODEL

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
        assert sent["modelId"] == MODEL
        assert sent["additionalModelRequestFields"] == {"output_config": {"effort": "medium"}}
        assert sent["inferenceConfig"] == {"maxTokens": 16000}

    def test_throttling_tem_retry(
        self, bedrock_mode: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client = use(monkeypatch, ScriptedBedrock(
            client_error(429, "ThrottlingException"), reply(text("oi"))))
        response = llm.complete([{"role": "user", "content": "oi"}])
        assert response.choices[0].message.content == "oi"
        assert len(client.requests) == 2

    def test_acesso_negado_nao_tem_retry(
        self, bedrock_mode: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client = use(monkeypatch, ScriptedBedrock(client_error(403, "AccessDeniedException")))
        with pytest.raises(llm.LLMUnavailable):
            llm.complete([{"role": "user", "content": "oi"}])
        assert len(client.requests) == 1

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
                                     "native_content": [REASONING]}], [])
        assert sent["messages"] == [{"role": "assistant", "content": ""}]


class TestTurnoCompleto:
    def test_raciocinio_volta_intacto_com_o_resultado_da_tool(
        self, chat: Chat, bedrock_mode: None, monkeypatch: pytest.MonkeyPatch,
        fake_backend: FakeBackend,
    ) -> None:
        client = use(monkeypatch, ScriptedBedrock(
            reply(REASONING, tool_use("search_products", {"query": "dom casmurro"}),
                  stop_reason="tool_use"),
            reply(text("Temos Dom Casmurro por R$ 39,90. Quer levar?")),
        ))
        r = chat.say("vocês têm dom casmurro?")

        assert r["reply"] == "Temos Dom Casmurro por R$ 39,90. Quer levar?"
        assert r["products"][0]["product_id"] == "BK001"
        assert r["degraded"] is False

        second = client.requests[1]
        assert "Estado da sessão" in second["system"][1]["text"]
        assistant, tool_result = second["messages"][-2:]
        assert assistant["content"][0] == REASONING
        assert "toolUse" in assistant["content"][1]
        assert tool_result["role"] == "user"
        assert tool_result["content"][0]["toolResult"]["toolUseId"] == "toolu_1"

    def test_bedrock_fora_e_sem_fallback_degrada_sem_5xx(
        self, chat: Chat, bedrock_mode: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        use(monkeypatch, ScriptedBedrock(
            *[client_error(503, "ServiceUnavailableException")] * llm.MAX_ATTEMPTS))
        r = chat.say("quero um livro")
        assert r["degraded"] is True
        assert "Clássicos" in [s["label"] for s in r["suggestions"]]


def test_requisicao_valida_contra_o_modelo_do_servico(
    bedrock_mode: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Client real do boto3: o Stubber recusa parâmetro ou resposta fora do schema."""
    client = boto3.client("bedrock-runtime", region_name="us-east-2")
    usage = {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2}
    metrics = {"latencyMs": 1}
    first = {**reply(REASONING, tool_use("search_products", {"query": "orwell"}),
                     stop_reason="tool_use"), "usage": usage, "metrics": metrics}
    second = {**reply(text("Achei dois.")), "usage": usage, "metrics": metrics}

    conversation: list[dict[str, Any]] = [
        {"role": "system", "content": "regras"}, {"role": "user", "content": "orwell"},
    ]
    with Stubber(client) as stub:
        stub.add_response("converse", first)
        stub.add_response("converse", second)
        monkeypatch.setattr(llm, "bedrock_client", lambda: client)

        message = llm.complete(conversation, TOOL_SCHEMAS).choices[0].message
        assert message.tool_calls[0].id == "toolu_1"

        conversation += [
            {"role": "assistant", "content": message.content,
             "native_content": message.native_content,
             "tool_calls": [{"id": "toolu_1", "type": "function",
                             "function": {"name": "search_products", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "toolu_1", "content": '{"count": 2}'},
        ]
        assert llm.complete(conversation, TOOL_SCHEMAS).choices[0].message.content == "Achei dois."
