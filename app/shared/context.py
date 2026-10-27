"""Contexto de correlação que atravessa frontend → chatbot → backend → SQS.

O `correlation_id` nasce no header `X-Request-Id` (gerado pelo frontend ou
aqui), é repassado em toda chamada ao backend, gravado no pedido e reaparece
nos logs do order-worker (PLANO §5.4, demo 12).
"""

from __future__ import annotations

import uuid
from contextvars import ContextVar

_correlation_id: ContextVar[str | None] = ContextVar("correlation_id", default=None)
_conversation_id: ContextVar[str | None] = ContextVar("conversation_id", default=None)
_order_id: ContextVar[str | None] = ContextVar("order_id", default=None)
_traceparent: ContextVar[str | None] = ContextVar("traceparent", default=None)

_FIELDS = {
    "correlation_id": _correlation_id,
    "conversation_id": _conversation_id,
    "order_id": _order_id,
    "traceparent": _traceparent,
}


def new_id() -> str:
    return str(uuid.uuid4())


def bind(**values: str | None) -> dict[str, object]:
    """Liga campos ao contexto e devolve os tokens para `unbind`."""
    tokens: dict[str, object] = {}
    for key, value in values.items():
        var = _FIELDS.get(key)
        if var is not None and value:
            tokens[key] = var.set(value)
    return tokens


def unbind(tokens: dict[str, object]) -> None:
    for key, token in tokens.items():
        _FIELDS[key].reset(token)  # type: ignore[arg-type]


def current() -> dict[str, str]:
    return {key: var.get() for key, var in _FIELDS.items() if var.get()}


def correlation_id() -> str | None:
    return _correlation_id.get()


def traceparent() -> str | None:
    return _traceparent.get()
