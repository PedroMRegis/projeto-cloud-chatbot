"""Logs JSON em uma linha, correlacionáveis no CloudWatch Logs Insights.

Sem PII: o conteúdo das mensagens do usuário nunca é logado (só o tamanho e as
tools chamadas) e nomes são mascarados por `mask_name` (PLANO §5.4).
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any

from app.shared import context
from app.shared.config import settings

_RESERVED = frozenset(
    logging.LogRecord("", 0, "", 0, "", None, None).__dict__
) | {"message", "asctime", "taskName"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(
                timespec="milliseconds"
            ),
            "level": record.levelname,
            "service": settings().service_name,
            "msg": record.getMessage(),
        }
        payload.update(context.current())
        # Campos passados via logger.info(..., extra={...}).
        payload.update(
            {k: v for k, v in record.__dict__.items() if k not in _RESERVED}
        )
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging() -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(settings().log_level.upper())
    # O access log do uvicorn duplicaria o que já registramos no middleware.
    logging.getLogger("uvicorn.access").disabled = True
    for noisy in ("botocore", "boto3", "urllib3", "httpx", "httpx2", "openai", "anthropic"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def mask_name(name: str | None) -> str:
    """"Maria Silva" -> "M*** S***". Mantém a forma sem revelar a PII."""
    if not name:
        return ""
    parts = [p for p in str(name).split() if p]
    return " ".join(f"{p[0]}***" for p in parts)
