"""Configuração via ambiente. Nenhum segredo tem default."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw else default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return float(raw) if raw else default


@dataclass(frozen=True)
class Settings:
    env: str = field(default_factory=lambda: _env("APP_ENV", "local"))
    region: str = field(default_factory=lambda: _env("AWS_REGION", "us-east-2"))
    service_name: str = field(default_factory=lambda: _env("SERVICE_NAME", "chatbot"))
    log_level: str = field(default_factory=lambda: _env("LOG_LEVEL", "INFO"))

    # DynamoDB (conversas). Vazio = endpoint real da AWS.
    dynamodb_endpoint: str = field(default_factory=lambda: _env("DYNAMODB_ENDPOINT", ""))
    table_conversations: str = field(
        default_factory=lambda: _env("TABLE_CONVERSATIONS", "conversations")
    )

    # Backend (catálogo, cotação, pedido). Na AWS é o endereço interno do
    # serviço (ECS Service Connect); em local, o container do compose.
    backend_url: str = field(default_factory=lambda: _env("BACKEND_URL", "http://localhost:8001"))
    backend_timeout_s: float = field(
        default_factory=lambda: _env_float("BACKEND_TIMEOUT_S", 5.0)
    )

    # LLM: "bedrock" (primário, com fallback gemini/groq se houver chave),
    # "gemini" (primário, com fallback groq), "groq" ou "mock".
    llm_provider: str = field(default_factory=lambda: _env("LLM_PROVIDER", "mock"))
    gemini_model: str = field(default_factory=lambda: _env("GEMINI_MODEL", "gemini-2.0-flash"))
    groq_model: str = field(
        default_factory=lambda: _env("GROQ_MODEL", "llama-3.3-70b-versatile")
    )
    # Perfil de inferência global: o Haiku 5.5 não aceita o ID do modelo direto, e
    # o perfil `us.` não serviria em São Paulo.
    bedrock_model: str = field(
        default_factory=lambda: _env("BEDROCK_MODEL", "global.anthropic.claude-haiku-5-5")
    )
    # Vazio = a mesma região do resto do serviço (AWS_REGION).
    bedrock_region: str = field(default_factory=lambda: _env("BEDROCK_REGION", ""))
    # Inclui o thinking do modelo, por isso bem acima do tamanho de uma resposta.
    bedrock_max_tokens: int = field(
        default_factory=lambda: _env_int("BEDROCK_MAX_TOKENS", 16000)
    )
    bedrock_effort: str = field(default_factory=lambda: _env("BEDROCK_EFFORT", "medium"))
    llm_timeout_s: int = field(default_factory=lambda: _env_int("LLM_TIMEOUT_S", 20))
    llm_max_tool_iterations: int = field(
        default_factory=lambda: _env_int("LLM_MAX_TOOL_ITERATIONS", 6)
    )
    mock_latency_ms: int = field(default_factory=lambda: _env_int("MOCK_LATENCY_MS", 300))

    # As chaves vêm do Parameter Store via secrets.valueFrom (PLANO §7).
    gemini_api_key: str = field(default_factory=lambda: _env("GEMINI_API_KEY", ""))
    groq_api_key: str = field(default_factory=lambda: _env("GROQ_API_KEY", ""))

    faults_parameter: str = field(default_factory=lambda: _env("FAULTS_PARAMETER", ""))

    # Rate limit por conversa (PLANO §7): protege a cota do LLM.
    rate_limit_per_minute: int = field(
        default_factory=lambda: _env_int("RATE_LIMIT_PER_MINUTE", 20)
    )
    conversation_ttl_days: int = field(
        default_factory=lambda: _env_int("CONVERSATION_TTL_DAYS", 7)
    )

    @property
    def is_local(self) -> bool:
        return bool(self.dynamodb_endpoint)


@lru_cache(maxsize=1)
def settings() -> Settings:
    return Settings()


def reset_settings_cache() -> None:
    """Usado pelos testes depois de mexer no ambiente."""
    settings.cache_clear()
