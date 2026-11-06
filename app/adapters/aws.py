"""Clientes boto3 compartilhados (um por processo)."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import boto3
from botocore.config import Config

from app.shared.config import settings

# Retry adaptativo: respeita o throttling do DynamoDB sem reinventar backoff.
_BOTO_CONFIG = Config(
    retries={"max_attempts": 5, "mode": "adaptive"},
    connect_timeout=3,
    read_timeout=10,
    user_agent_extra="convcommerce",
)


@lru_cache(maxsize=1)
def dynamodb_resource() -> Any:
    cfg = settings()
    kwargs: dict[str, Any] = {"region_name": cfg.region, "config": _BOTO_CONFIG}
    if cfg.dynamodb_endpoint:
        kwargs["endpoint_url"] = cfg.dynamodb_endpoint
    return boto3.resource("dynamodb", **kwargs)


@lru_cache(maxsize=1)
def ssm_client() -> Any:
    return boto3.client("ssm", region_name=settings().region, config=_BOTO_CONFIG)


@lru_cache(maxsize=1)
def bedrock_client() -> Any:
    cfg = settings()
    # O retry do LLM é o nosso (app/adapters/llm.py), para contar métrica por tentativa.
    config = Config(
        retries={"total_max_attempts": 1, "mode": "standard"},
        connect_timeout=3,
        read_timeout=cfg.llm_timeout_s,
        user_agent_extra="convcommerce",
    )
    return boto3.client(
        "bedrock-runtime", region_name=cfg.bedrock_region or cfg.region, config=config
    )


def reset_clients() -> None:
    for factory in (dynamodb_resource, ssm_client, bedrock_client):
        factory.cache_clear()
