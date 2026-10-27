"""Injeção de falhas controlada, para acionar incidentes sem redeploy.

O JSON fica no parâmetro SSM `/convcommerce/<env>/faults` e é lido com cache de
15 s. Nas demos, basta um `aws ssm put-parameter` para provocar o incidente
(PLANO §5.4, demos 10 e 11).
"""

from __future__ import annotations

import json
import random
import threading
import time

from app.shared.config import settings
from app.shared.logging import get_logger

log = get_logger(__name__)

CACHE_TTL_S = 15

DEFAULTS: dict[str, float] = {
    "llm_error_rate": 0.0,
    "llm_delay_ms": 0.0,
    "worker_error_rate": 0.0,
    "worker_delay_ms": 0.0,
    "dynamo_error_rate": 0.0,
    "backend_error_rate": 0.0,
}

_lock = threading.Lock()
_cached: dict[str, float] = dict(DEFAULTS)
_fetched_at: float = 0.0


class InjectedFault(Exception):
    """Falha deliberada. Tratada como erro transitório pelos chamadores."""


def _fetch() -> dict[str, float]:
    cfg = settings()
    if not cfg.faults_parameter:
        return dict(DEFAULTS)
    from app.adapters.ssm import get_parameter

    raw = get_parameter(cfg.faults_parameter)
    values = dict(DEFAULTS)
    if raw:
        parsed = json.loads(raw)
        values.update({k: float(v) for k, v in parsed.items() if k in DEFAULTS})
    return values


def current() -> dict[str, float]:
    global _fetched_at, _cached
    now = time.monotonic()
    with _lock:
        if now - _fetched_at < CACHE_TTL_S and _fetched_at:
            return dict(_cached)
    try:
        values = _fetch()
    except Exception:
        # Falha ao ler o parâmetro nunca pode derrubar o caminho de produção.
        log.warning("falha ao ler parâmetro de faults, mantendo valor anterior")
        values = dict(_cached)
    with _lock:
        _cached = values
        _fetched_at = now
        return dict(values)


def maybe_fail(rate_key: str, delay_key: str | None = None) -> None:
    """Aplica atraso e/ou erro conforme o parâmetro, se configurado."""
    values = current()

    if delay_key:
        delay_ms = values.get(delay_key, 0.0)
        if delay_ms > 0:
            time.sleep(delay_ms / 1000)

    rate = values.get(rate_key, 0.0)
    if rate > 0 and random.random() < rate:
        log.warning("falha injetada", extra={"fault": rate_key, "rate": rate})
        raise InjectedFault(rate_key)


def reset_cache() -> None:
    global _fetched_at, _cached
    with _lock:
        _cached = dict(DEFAULTS)
        _fetched_at = 0.0
