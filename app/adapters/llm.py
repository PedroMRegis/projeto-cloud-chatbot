"""Adapter de LLM: Gemini (primário) → Groq (fallback) → resposta degradada.

A IA é tratada como dependência sujeita a latência, falhas e limites
(enunciado §4). As três defesas, em ordem:

1. **Timeout + retry** com backoff exponencial e jitter para 429/5xx.
2. **Circuit breaker** por provedor: depois de 5 falhas em 60 s o provedor fica
   30 s aberto, e as requisições vão direto ao fallback em vez de gastar o
   timeout de 20 s em uma dependência que já se sabe quebrada.
3. **Fallback entre provedores** e, se ambos falharem, `LLMUnavailable` — que a
   API converte em resposta degradada com HTTP 200, nunca em 5xx.

Os dois provedores são usados pelo endpoint compatível com OpenAI, então um
único SDK (`openai`) atende aos dois (PLANO D-26).
"""

from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from app.shared import faults, metrics
from app.shared.config import settings
from app.shared.logging import get_logger

log = get_logger(__name__)

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

MAX_ATTEMPTS = 3
BREAKER_FAILURE_THRESHOLD = 5
BREAKER_WINDOW_S = 60
BREAKER_OPEN_S = 30


class LLMUnavailable(Exception):
    """Nenhum provedor respondeu. O chamador deve degradar, não falhar."""


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
    """Ordem de tentativa, conforme `LLM_PROVIDER`. Sem chave = fora da lista."""
    cfg = settings()
    gemini = ProviderConfig("gemini", GEMINI_BASE_URL, cfg.gemini_api_key, cfg.gemini_model)
    groq = ProviderConfig("groq", GROQ_BASE_URL, cfg.groq_api_key, cfg.groq_model)
    order = [groq, gemini] if cfg.llm_provider == "groq" else [gemini, groq]
    return [p for p in order if p.api_key]


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


def _is_retryable(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(getattr(exc, "response", None), "status_code", None)
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
                return _client(provider).chat.completions.create(
                    model=provider.model,
                    messages=messages,
                    tools=tools or None,
                    tool_choice="auto" if tools else None,
                    temperature=0.3,
                )
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
        except Exception:  # noqa: BLE001
            breaker(provider.name).record_failure()
            continue

        breaker(provider.name).record_success()
        if provider.name != primary:
            metrics.emit("LLMFallbacks", 1, dim_provider=provider.name)
            log.warning("respondido pelo fallback", extra={"provider": provider.name})
        return response

    raise LLMUnavailable("todos os provedores de LLM falharam")
