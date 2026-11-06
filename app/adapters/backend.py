"""Cliente HTTP do backend-api: a única forma de o chatbot tocar no catálogo.

O backend é uma dependência remota como o LLM, então recebe o mesmo
tratamento:

* **timeout curto** (5 s) — o chat não pode ficar preso esperando o banco;
* **retry com jitter** em erro de conexão e 502/503/504. Retentar é seguro em
  todas as rotas: leituras não têm efeito, uma cotação repetida só gera uma
  cotação a mais (sem efeito comercial) e o pedido é idempotente por
  `quote_id` no backend;
* **circuit breaker** — 5 falhas em 30 s abrem o circuito por 15 s, e o chat
  responde de forma degradada na hora, em vez de gastar timeout;
* **propagação de contexto** — `X-Request-Id`, `traceparent` e
  `X-Conversation-Id` seguem para o backend, então os logs dos dois serviços
  se correlacionam pelo mesmo ID. A sessão do cliente vai como `Authorization`,
  para o pedido ficar na conta dele.
"""

from __future__ import annotations

import random
import threading
import time
from typing import Any

import httpx

from app.shared import context, faults, metrics
from app.shared.config import settings
from app.shared.logging import get_logger

log = get_logger(__name__)

MAX_ATTEMPTS = 3
RETRY_STATUS = {502, 503, 504}
BREAKER_THRESHOLD = 5
BREAKER_WINDOW_S = 30
BREAKER_OPEN_S = 15


class BackendError(Exception):
    """Resposta 4xx: erro de negócio que o vendedor deve explicar ao cliente."""

    def __init__(self, status: int, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.detail = detail


class BackendUnavailable(Exception):
    """Backend fora do ar, lento demais ou com o circuito aberto."""


class _Breaker:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._failures: list[float] = []
        self._opened_at: float | None = None

    def is_open(self) -> bool:
        with self._lock:
            if self._opened_at is None:
                return False
            if time.monotonic() - self._opened_at >= BREAKER_OPEN_S:
                self._opened_at = None
                self._failures.clear()
                return False
            return True

    def success(self) -> None:
        with self._lock:
            self._failures.clear()
            self._opened_at = None

    def failure(self) -> None:
        with self._lock:
            now = time.monotonic()
            self._failures = [t for t in self._failures if now - t < BREAKER_WINDOW_S]
            self._failures.append(now)
            if len(self._failures) >= BREAKER_THRESHOLD and self._opened_at is None:
                self._opened_at = now
                log.error("circuito do backend aberto")
                metrics.emit("BackendCircuitOpen", 1)


_breaker = _Breaker()
_client: httpx.Client | None = None
_client_lock = threading.Lock()


def _http() -> httpx.Client:
    global _client
    with _client_lock:
        if _client is None:
            cfg = settings()
            _client = httpx.Client(
                base_url=cfg.backend_url,
                timeout=httpx.Timeout(cfg.backend_timeout_s, connect=2.0),
                limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
            )
        return _client


def reset() -> None:
    """Usado pelos testes: novo client e circuito fechado."""
    global _client, _breaker
    with _client_lock:
        if _client is not None:
            _client.close()
        _client = None
    _breaker = _Breaker()


def _headers() -> dict[str, str]:
    ctx = context.current()
    headers = {"X-Request-Id": ctx.get("correlation_id") or context.new_id()}
    if ctx.get("traceparent"):
        headers["traceparent"] = ctx["traceparent"]
    if ctx.get("conversation_id"):
        headers["X-Conversation-Id"] = ctx["conversation_id"]
    token = context.auth_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def request(method: str, path: str, **kwargs: Any) -> Any:
    if _breaker.is_open():
        metrics.emit("BackendShortCircuited", 1)
        raise BackendUnavailable("circuito aberto")

    last_error = "desconhecido"
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            faults.maybe_fail("backend_error_rate")
            with metrics.timed("BackendLatency", dim_route=path.split("/")[2]):
                response = _http().request(method, path, headers=_headers(), **kwargs)
        except (httpx.TransportError, faults.InjectedFault) as exc:
            last_error = type(exc).__name__
        else:
            if response.status_code not in RETRY_STATUS and response.status_code < 500:
                _breaker.success()
                if response.status_code >= 400:
                    detail = response.json().get("detail", response.text)
                    if isinstance(detail, list):  # erro de validação do FastAPI
                        detail = "; ".join(str(d.get("msg", d)) for d in detail)
                    raise BackendError(response.status_code, str(detail))
                return response.json()
            last_error = f"HTTP {response.status_code}"
            if response.status_code not in RETRY_STATUS:
                break

        metrics.emit("BackendErrors", 1, error=last_error)
        log.warning("chamada ao backend falhou",
                    extra={"path": path, "attempt": attempt, "error": last_error})
        if attempt < MAX_ATTEMPTS:
            time.sleep(0.2 * attempt * (0.5 + random.random()))

    _breaker.failure()
    raise BackendUnavailable(last_error)


# ------------------------------------------------------------- rotas usadas


def search_products(query: str, category: str | None = None, limit: int = 5) -> dict[str, Any]:
    params: dict[str, Any] = {"q": query, "limit": limit}
    if category:
        params["category"] = category
    return request("GET", "/api/products", params=params)


def get_product(product_id: str) -> dict[str, Any]:
    return request("GET", f"/api/products/{product_id}")


def check_availability(product_id: str, quantity: int) -> dict[str, Any]:
    return request("GET", f"/api/products/{product_id}/availability",
                   params={"quantity": quantity})


def categories() -> list[dict[str, Any]]:
    return request("GET", "/api/categories")["categories"]


def create_quote(conversation_id: str, items: list[dict[str, Any]],
                 requested_discount_pct: int = 0) -> dict[str, Any]:
    return request("POST", "/api/quotes", json={
        "conversation_id": conversation_id, "items": items,
        "requested_discount_pct": requested_discount_pct,
    })


def place_order(quote_id: str, conversation_id: str) -> dict[str, Any]:
    """O nome do pedido o backend tira da conta da sessão."""
    return request("POST", "/api/orders", json={
        "quote_id": quote_id, "conversation_id": conversation_id,
    })


def get_order(order_id: str, conversation_id: str) -> dict[str, Any]:
    return request("GET", f"/api/orders/{order_id}",
                   params={"conversation_id": conversation_id})


def current_user() -> dict[str, Any]:
    """A conta da sessão repassada no `Authorization`."""
    return request("GET", "/api/auth/me")


def health() -> bool:
    try:
        return _http().get("/health", timeout=2.0).status_code == 200
    except httpx.HTTPError:
        return False
