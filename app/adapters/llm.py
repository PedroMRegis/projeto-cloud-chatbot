"""Adapter de LLM: Bedrock (Claude) → Gemini → Groq → resposta degradada.

A IA é tratada como dependência sujeita a latência, falhas e limites
(enunciado §4). As três defesas, em ordem:

1. **Timeout + retry** com backoff exponencial e jitter para 429/5xx.
2. **Circuit breaker** por provedor: depois de 5 falhas em 60 s o provedor fica
   30 s aberto, e as requisições vão direto ao fallback em vez de gastar o
   timeout de 20 s em uma dependência que já se sabe quebrada.
3. **Fallback entre provedores** e, se ambos falharem, `LLMUnavailable` — que a
   API converte em resposta degradada com HTTP 200, nunca em 5xx.

Gemini e Groq são usados pelo endpoint compatível com OpenAI, então um único SDK
(`openai`) atende aos dois (PLANO D-26). O Bedrock usa a API Converse do boto3, e
a autenticação é a credencial AWS da task (task role), sem chave no ambiente.

O formato interno das mensagens continua sendo o da OpenAI: é ele que permite
trocar de provedor no meio de um turno. `_to_converse` e `_to_chat_completion`
convertem na entrada e na saída do Bedrock, então o agente quase não muda.
"""

from __future__ import annotations

import json
import random
import threading
import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

from app.adapters.aws import bedrock_client
from app.shared import faults, metrics
from app.shared.config import settings
from app.shared.logging import get_logger

log = get_logger(__name__)

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
BEDROCK = "bedrock"
REFUSAL_STOP_REASONS = {"refusal", "content_filtered", "guardrail_intervened"}

MAX_ATTEMPTS = 3
BREAKER_FAILURE_THRESHOLD = 5
BREAKER_WINDOW_S = 60
BREAKER_OPEN_S = 30


class LLMUnavailable(Exception):
    """Nenhum provedor respondeu. O chamador deve degradar, não falhar."""


class ModelRefused(Exception):
    """O modelo recusou a mensagem. Não é falha do provedor: sem retry nem breaker."""


@dataclass
class CircuitBreaker:
    name: str
    failures: list[float] = field(default_factory=list)
    opened_at: float | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def is_open(self) -> bool:
        with self._lock:
            if self.opened_at is None:
                return False
            if time.monotonic() - self.opened_at >= BREAKER_OPEN_S:
                # Half-open: deixa uma tentativa passar para sondar o provedor.
                self.opened_at = None
                self.failures.clear()
                log.info("circuito fechado após janela", extra={"provider": self.name})
                return False
            return True

    def record_success(self) -> None:
        with self._lock:
            self.failures.clear()
            self.opened_at = None

    def record_failure(self) -> None:
        with self._lock:
            now = time.monotonic()
            self.failures = [t for t in self.failures if now - t < BREAKER_WINDOW_S]
            self.failures.append(now)
            if len(self.failures) >= BREAKER_FAILURE_THRESHOLD and self.opened_at is None:
                self.opened_at = now
                log.error(
                    "circuito aberto",
                    extra={"provider": self.name, "failures": len(self.failures)},
                )
                metrics.emit("LLMCircuitOpen", 1, dim_provider=self.name)


_breakers: dict[str, CircuitBreaker] = {}
_breakers_lock = threading.Lock()


def breaker(provider: str) -> CircuitBreaker:
    with _breakers_lock:
        if provider not in _breakers:
            _breakers[provider] = CircuitBreaker(provider)
        return _breakers[provider]


def reset_breakers() -> None:
    with _breakers_lock:
        _breakers.clear()


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    base_url: str
    api_key: str
    model: str


def _providers() -> list[ProviderConfig]:
    """Ordem de tentativa, conforme `LLM_PROVIDER`.

    Gemini e Groq sem chave ficam fora da lista. O Bedrock não tem chave (usa a
    credencial AWS), então só entra quando é o provedor escolhido.
    """
    cfg = settings()
    gemini = ProviderConfig("gemini", GEMINI_BASE_URL, cfg.gemini_api_key, cfg.gemini_model)
    groq = ProviderConfig("groq", GROQ_BASE_URL, cfg.groq_api_key, cfg.groq_model)
    order = [groq, gemini] if cfg.llm_provider == "groq" else [gemini, groq]
    keyed = [p for p in order if p.api_key]
    if cfg.llm_provider == BEDROCK:
        return [ProviderConfig(BEDROCK, "", "", cfg.bedrock_model), *keyed]
    return keyed


_clients: dict[str, Any] = {}
_clients_lock = threading.Lock()


def _client(provider: ProviderConfig) -> Any:
    from openai import OpenAI

    with _clients_lock:
        if provider.name not in _clients:
            _clients[provider.name] = OpenAI(
                api_key=provider.api_key,
                base_url=provider.base_url,
                timeout=settings().llm_timeout_s,
                max_retries=0,  # o retry é nosso, para contar métrica por tentativa
            )
        return _clients[provider.name]


def _converse_tools(tools: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "tools": [
            {
                "toolSpec": {
                    "name": tool["function"]["name"],
                    "description": tool["function"]["description"],
                    "inputSchema": {"json": tool["function"]["parameters"]},
                }
            }
            for tool in tools
        ]
    }


def _assistant_blocks(message: dict[str, Any]) -> list[dict[str, Any]]:
    # Blocos originais do Bedrock: os de raciocínio precisam voltar intactos junto
    # com o resultado das tools. Só o que veio de outro provedor é reconstruído.
    if message.get("native_content"):
        return message["native_content"]
    blocks: list[dict[str, Any]] = []
    if message.get("content"):
        blocks.append({"text": message["content"]})
    for call in message.get("tool_calls") or []:
        blocks.append({
            "toolUse": {
                "toolUseId": call["id"],
                "name": call["function"]["name"],
                "input": json.loads(call["function"]["arguments"] or "{}"),
            }
        })
    return blocks


def _to_converse(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    """Mensagens no formato OpenAI → (`system`, `messages`) da API Converse."""
    system: list[dict[str, str]] = []
    converted: list[dict[str, Any]] = []
    results: list[dict[str, Any]] | None = None  # bloco de toolResult em aberto
    for message in messages:
        role = message["role"]
        if role == "tool":
            result = {"toolResult": {"toolUseId": message["tool_call_id"],
                                     "content": [{"text": message["content"]}]}}
            # Todos os resultados de um turno vão numa única mensagem de usuário.
            if results is None:
                results = []
                converted.append({"role": "user", "content": results})
            results.append(result)
            continue
        results = None
        if role == "system":
            system.append({"text": message["content"]})
        elif role == "user":
            converted.append({"role": "user", "content": [{"text": message["content"]}]})
        elif converted:
            # A conversa precisa começar pelo usuário: o histórico é um recorte
            # das últimas mensagens e pode começar numa resposta do vendedor.
            converted.append({"role": "assistant", "content": _assistant_blocks(message)})
    return system, converted


def _to_chat_completion(response: dict[str, Any]) -> Any:
    """Resposta da Converse → o formato de chat completion que o agente lê."""
    blocks = response["output"]["message"]["content"]
    calls = [
        SimpleNamespace(
            id=block["toolUse"]["toolUseId"],
            function=SimpleNamespace(
                name=block["toolUse"]["name"],
                arguments=json.dumps(block["toolUse"]["input"], ensure_ascii=False),
            ),
        )
        for block in blocks
        if "toolUse" in block
    ]
    message = SimpleNamespace(
        content="".join(b["text"] for b in blocks if "text" in b),
        tool_calls=calls or None,
        native_content=blocks,
    )
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _call_bedrock(
    provider: ProviderConfig, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
) -> Any:
    cfg = settings()
    system, converted = _to_converse(messages)
    kwargs: dict[str, Any] = {"toolConfig": _converse_tools(tools)} if tools else {}
    # Sem `temperature`: o Claude Haiku 5.5 recusa valores diferentes do padrão.
    response = bedrock_client().converse(
        modelId=provider.model,
        system=system,
        messages=converted,
        inferenceConfig={"maxTokens": cfg.bedrock_max_tokens},
        additionalModelRequestFields={"output_config": {"effort": cfg.bedrock_effort}},
        **kwargs,
    )
    if response["stopReason"] in REFUSAL_STOP_REASONS:
        raise ModelRefused(provider.name)
    return _to_chat_completion(response)


def _call_openai(
    provider: ProviderConfig, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
) -> Any:
    return _client(provider).chat.completions.create(
        model=provider.model,
        # `native_content` só existe para o Bedrock, e a API da OpenAI o recusaria.
        messages=[{k: v for k, v in m.items() if k != "native_content"} for m in messages],
        tools=tools or None,
        tool_choice="auto" if tools else None,
        temperature=0.3,
    )


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, ModelRefused):
        return False
    status = getattr(exc, "status_code", None)
    if status is None:
        response = getattr(exc, "response", None)
        if isinstance(response, dict):  # ClientError do boto3
            status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        else:
            status = getattr(response, "status_code", None)
    if status is not None:
        return status == 429 or status >= 500
    # Timeout e erro de conexão não têm status e valem retry.
    return True


def _call_provider(
    provider: ProviderConfig, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
) -> Any:
    last_exc: Exception | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            with metrics.timed("LLMLatency", dim_provider=provider.name):
                faults.maybe_fail("llm_error_rate", "llm_delay_ms")
                call = _call_bedrock if provider.name == BEDROCK else _call_openai
                return call(provider, messages, tools)
        except Exception as exc:  # noqa: BLE001 — qualquer falha vira fallback
            last_exc = exc
            metrics.emit(
                "LLMErrors", 1, dim_provider=provider.name, error=type(exc).__name__
            )
            log.warning(
                "chamada ao LLM falhou",
                extra={
                    "provider": provider.name,
                    "attempt": attempt,
                    "error": type(exc).__name__,
                },
            )
            if attempt == MAX_ATTEMPTS or not _is_retryable(exc):
                break
            # Backoff exponencial com jitter: evita sincronizar os retries de
            # várias tasks sobre um provedor já sobrecarregado.
            time.sleep(min(2 ** (attempt - 1), 4) * (0.5 + random.random()))

    assert last_exc is not None
    raise last_exc


def complete(
    messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
) -> Any:
    """Uma rodada de chat completion, com fallback entre provedores."""
    providers = _providers()
    if not providers:
        raise LLMUnavailable("nenhum provedor de LLM configurado")

    primary = providers[0].name
    for provider in providers:
        if breaker(provider.name).is_open():
            log.warning(
                "provedor em circuito aberto, pulando",
                extra={"provider": provider.name},
            )
            continue
        try:
            response = _call_provider(provider, messages, tools or [])
        except ModelRefused:
            # Tenta o próximo provedor, mas sem contar como falha: senão cinco
            # mensagens recusadas abririam o circuito para todos os clientes.
            metrics.emit("LLMRefusals", 1, dim_provider=provider.name)
            continue
        except Exception:  # noqa: BLE001
            breaker(provider.name).record_failure()
            continue

        breaker(provider.name).record_success()
        if provider.name != primary:
            metrics.emit("LLMFallbacks", 1, dim_provider=provider.name)
            log.warning("respondido pelo fallback", extra={"provider": provider.name})
        return response

    raise LLMUnavailable("todos os provedores de LLM falharam")
