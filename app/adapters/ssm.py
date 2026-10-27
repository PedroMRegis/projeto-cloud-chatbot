"""Leitura de parâmetros do SSM Parameter Store."""

from __future__ import annotations

from app.adapters.aws import ssm_client


def get_parameter(name: str, *, decrypt: bool = False) -> str | None:
    response = ssm_client().get_parameter(Name=name, WithDecryption=decrypt)
    return response["Parameter"]["Value"]
