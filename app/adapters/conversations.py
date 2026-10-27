"""Conversas no DynamoDB (ADR 0002: conversa é dado do chatbot, não do backend).

Uma partição por conversa, com itens distinguidos pela sort key:

* `MSG#<ts>#<n>` — uma mensagem por item. Guardar a conversa inteira num item
  só bateria no limite de 400 KB e criaria conflito entre turnos concorrentes.
  A resposta do vendedor guarda também o `ui` (cards, cotação, sugestões), para
  o frontend reconstruir a conversa depois de um reload.
* `STATE` — o estado da sessão de venda (livro em foco, cotação, nome...). É
  ele que dá contexto ao "sim": o vendedor sabe a que pergunta o "sim" responde.
* `REQ#<request_id>` — resposta armazenada, para idempotência da requisição.
* `RATE#<minuto>` — contador do rate limit.

Tudo com TTL: a conversa some sozinha depois de 7 dias.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from botocore.exceptions import ClientError

from app.adapters.aws import dynamodb_resource
from app.shared import faults
from app.shared.config import settings


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def _epoch(offset_seconds: int = 0) -> int:
    return int(time.time()) + offset_seconds


def to_plain(value: Any) -> Any:
    """Converte Decimal do DynamoDB em int/float para serializar em JSON."""
    if isinstance(value, Decimal):
        return int(value) if value % 1 == 0 else float(value)
    if isinstance(value, list):
        return [to_plain(v) for v in value]
    if isinstance(value, dict):
        return {k: to_plain(v) for k, v in value.items()}
    return value


def _clean(value: Any) -> Any:
    """O DynamoDB não aceita float nem string vazia em alguns contextos."""
    if isinstance(value, float):
        return Decimal(str(value))
    if isinstance(value, list):
        return [_clean(v) for v in value]
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items() if v is not None}
    return value


def _table() -> Any:
    faults.maybe_fail("dynamo_error_rate")
    return dynamodb_resource().Table(settings().table_conversations)


class ConversationRepository:
    def _ttl(self) -> int:
        return _epoch(settings().conversation_ttl_days * 86400)

    # --- mensagens ---

    def append_message(
        self, conversation_id: str, index: int, role: str, content: str, **extra: Any
    ) -> None:
        item = {
            "conversation_id": conversation_id,
            "sk": f"MSG#{now_iso()}#{index:04d}",
            "role": role,
            "content": content,
            "expires_at": self._ttl(),
            **{k: _clean(v) for k, v in extra.items() if v is not None},
        }
        _table().put_item(Item=item)

    def history(self, conversation_id: str, limit: int = 40) -> list[dict[str, Any]]:
        response = _table().query(
            KeyConditionExpression="conversation_id = :cid AND begins_with(sk, :prefix)",
            ExpressionAttributeValues={":cid": conversation_id, ":prefix": "MSG#"},
            # Lê as mais recentes e reordena: o histórico relevante é o final.
            ScanIndexForward=False,
            Limit=limit,
        )
        return list(reversed([to_plain(i) for i in response.get("Items", [])]))

    # --- estado da sessão de venda ---

    def get_state(self, conversation_id: str) -> dict[str, Any]:
        response = _table().get_item(
            Key={"conversation_id": conversation_id, "sk": "STATE"}, ConsistentRead=True
        )
        item = response.get("Item")
        return to_plain(item.get("state", {})) if item else {}

    def save_state(self, conversation_id: str, state: dict[str, Any]) -> None:
        _table().put_item(
            Item={
                "conversation_id": conversation_id,
                "sk": "STATE",
                "state": _clean(state),
                "updated_at": now_iso(),
                "expires_at": self._ttl(),
            }
        )

    # --- idempotência da requisição (PLANO §5.2) ---

    def get_cached_response(self, conversation_id: str, request_id: str) -> dict[str, Any] | None:
        response = _table().get_item(
            Key={"conversation_id": conversation_id, "sk": f"REQ#{request_id}"},
            ConsistentRead=True,
        )
        item = response.get("Item")
        return to_plain(item).get("response") if item else None

    def save_response(
        self, conversation_id: str, request_id: str, response: dict[str, Any]
    ) -> None:
        _table().put_item(
            Item={
                "conversation_id": conversation_id,
                "sk": f"REQ#{request_id}",
                "response": _clean(response),
                "created_at": now_iso(),
                "expires_at": self._ttl(),
            }
        )

    def claim_request(self, conversation_id: str, request_id: str) -> bool:
        """Reserva o `request_id`. False = já processado ou em processamento."""
        try:
            _table().put_item(
                Item={
                    "conversation_id": conversation_id,
                    "sk": f"REQ#{request_id}",
                    "created_at": now_iso(),
                    "expires_at": self._ttl(),
                },
                ConditionExpression="attribute_not_exists(sk)",
            )
            return True
        except ClientError as exc:
            if exc.response["Error"]["Code"] == "ConditionalCheckFailedException":
                return False
            raise

    def release_request(self, conversation_id: str, request_id: str) -> None:
        """Libera o `request_id` quando o turno falha antes de responder.

        Sem isso, um erro transitório (backend fora, por exemplo) deixaria o
        `request_id` reservado para sempre e o retry do cliente receberia 409.
        """
        _table().delete_item(Key={"conversation_id": conversation_id, "sk": f"REQ#{request_id}"})

    # --- rate limit por conversa (PLANO §7) ---

    def hit_rate_limit(self, conversation_id: str, limit_per_minute: int) -> bool:
        """Contador atômico por minuto. True = excedeu a cota."""
        minute = int(time.time()) // 60
        response = _table().update_item(
            Key={"conversation_id": conversation_id, "sk": f"RATE#{minute}"},
            UpdateExpression="ADD hits :one SET expires_at = :ttl",
            ExpressionAttributeValues={":one": 1, ":ttl": _epoch(180)},
            ReturnValues="UPDATED_NEW",
        )
        return int(response["Attributes"]["hits"]) > limit_per_minute
