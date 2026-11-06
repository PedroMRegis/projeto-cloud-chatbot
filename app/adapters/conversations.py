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
* `META` — a conta dona da conversa, quando há login.

Mensagens, estado e dono são permanentes. Só `REQ#` e `RATE#` têm TTL.

Para listar as conversas de uma conta sem criar índice, cada conta tem a própria
partição, `USER#<user_id>`, com um item `CONV#<conversation_id>` por conversa.
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

TITLE_MAX_CHARS = 80
PREVIEW_MAX_CHARS = 160


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
    def _request_ttl(self) -> int:
        return _epoch(settings().request_ttl_days * 86400)

    # --- mensagens ---

    def append_message(
        self, conversation_id: str, index: int, role: str, content: str, **extra: Any
    ) -> None:
        item = {
            "conversation_id": conversation_id,
            "sk": f"MSG#{now_iso()}#{index:04d}",
            "role": role,
            "content": content,
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
            }
        )

    # --- dono da conversa e lista por conta ---

    def owner(self, conversation_id: str) -> str | None:
        response = _table().get_item(
            Key={"conversation_id": conversation_id, "sk": "META"}, ConsistentRead=True
        )
        item = response.get("Item")
        return item["user_id"] if item else None

    def claim(self, conversation_id: str, user_id: str) -> str:
        """Dá a conversa à conta, se ainda não tiver dono. Devolve o dono atual."""
        try:
            _table().put_item(
                Item={"conversation_id": conversation_id, "sk": "META", "user_id": user_id,
                      "created_at": now_iso()},
                ConditionExpression="attribute_not_exists(sk)",
            )
            return user_id
        except ClientError as exc:
            if exc.response["Error"]["Code"] != "ConditionalCheckFailedException":
                raise
        return self.owner(conversation_id) or user_id

    def index_for_user(
        self, user_id: str, conversation_id: str, *, title: str, last_message: str, stage: str
    ) -> None:
        now = now_iso()
        _table().update_item(
            Key={"conversation_id": f"USER#{user_id}", "sk": f"CONV#{conversation_id}"},
            UpdateExpression=(
                "SET title = if_not_exists(title, :title), "
                "created_at = if_not_exists(created_at, :now), "
                "updated_at = :now, last_message = :last, stage = :stage"
            ),
            ExpressionAttributeValues={
                ":title": title[:TITLE_MAX_CHARS], ":now": now,
                ":last": last_message[:PREVIEW_MAX_CHARS], ":stage": stage,
            },
        )

    def list_for_user(self, user_id: str) -> list[dict[str, Any]]:
        """Conversas da conta, da mais recente para a mais antiga."""
        items: list[dict[str, Any]] = []
        kwargs: dict[str, Any] = {
            "KeyConditionExpression": "conversation_id = :pk AND begins_with(sk, :prefix)",
            "ExpressionAttributeValues": {":pk": f"USER#{user_id}", ":prefix": "CONV#"},
        }
        while True:
            page = _table().query(**kwargs)
            items.extend(to_plain(i) for i in page.get("Items", []))
            if "LastEvaluatedKey" not in page:
                break
            kwargs["ExclusiveStartKey"] = page["LastEvaluatedKey"]
        items.sort(key=lambda i: i["updated_at"], reverse=True)
        return [
            {"conversation_id": i["sk"].removeprefix("CONV#"), "title": i["title"],
             "last_message": i["last_message"], "stage": i["stage"],
             "created_at": i["created_at"], "updated_at": i["updated_at"]}
            for i in items
        ]

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
                "expires_at": self._request_ttl(),
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
                    "expires_at": self._request_ttl(),
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
