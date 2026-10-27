"""Métricas de negócio no formato EMF (Embedded Metric Format).

O EMF é emitido como uma linha de log no stdout: o CloudWatch Logs extrai as
métricas do namespace `ConvCommerce` sem nenhuma chamada de API, o que evita
latência e throttling de `PutMetricData` no caminho da requisição
(PLANO §8.4).
"""

from __future__ import annotations

import json
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Literal

from app.shared import context
from app.shared.config import settings

NAMESPACE = "ConvCommerce"

Unit = Literal["Count", "Milliseconds", "None"]

_lock = threading.Lock()


def emit(
    name: str,
    value: float = 1,
    unit: Unit = "Count",
    **dimensions_and_properties: object,
) -> None:
    """Publica uma métrica. `service` e `env` são sempre dimensões."""
    cfg = settings()
    dimensions = {"service": cfg.service_name, "env": cfg.env}
    extra_dims = {
        k: str(v)
        for k, v in dimensions_and_properties.items()
        if k.startswith("dim_") and v is not None
    }
    dimensions.update({k.removeprefix("dim_"): v for k, v in extra_dims.items()})

    document: dict[str, object] = {
        "_aws": {
            "Timestamp": int(time.time() * 1000),
            "CloudWatchMetrics": [
                {
                    "Namespace": NAMESPACE,
                    "Dimensions": [list(dimensions)],
                    "Metrics": [{"Name": name, "Unit": unit}],
                }
            ],
        },
        name: value,
        **dimensions,
        **context.current(),
        **{
            k: v
            for k, v in dimensions_and_properties.items()
            if not k.startswith("dim_") and v is not None
        },
    }
    line = json.dumps(document, ensure_ascii=False, default=str)
    with _lock:
        sys.stdout.write(line + "\n")
        sys.stdout.flush()


@contextmanager
def timed(name: str, **properties: object) -> Iterator[None]:
    """Mede a duração do bloco e emite a métrica em milissegundos."""
    started = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - started) * 1000
        emit(name, round(elapsed_ms, 2), "Milliseconds", **properties)
