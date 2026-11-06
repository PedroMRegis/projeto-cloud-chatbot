"""Tira o prazo de validade das conversas que já existem.

Mensagens e estado gravados antes de as conversas passarem a ser permanentes
ainda têm `expires_at` e seriam apagados pelo TTL do DynamoDB. Itens `REQ#` e
`RATE#` continuam expirando.

    python scripts/remove_message_ttl.py --table <tabela>            só conta
    python scripts/remove_message_ttl.py --table <tabela> --apply    aplica
"""

from __future__ import annotations

import argparse
from typing import Any

import boto3


def permanent_items(table: Any) -> list[dict[str, str]]:
    keys: list[dict[str, str]] = []
    kwargs: dict[str, Any] = {
        "ProjectionExpression": "conversation_id, sk",
        "FilterExpression": "attribute_exists(expires_at) AND "
                            "(begins_with(sk, :msg) OR sk = :state)",
        "ExpressionAttributeValues": {":msg": "MSG#", ":state": "STATE"},
    }
    while True:
        page = table.scan(**kwargs)
        keys.extend(page.get("Items", []))
        if "LastEvaluatedKey" not in page:
            return keys
        kwargs["ExclusiveStartKey"] = page["LastEvaluatedKey"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", required=True)
    parser.add_argument("--region", default=None)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    table = boto3.resource("dynamodb", region_name=args.region).Table(args.table)
    keys = permanent_items(table)
    print(f"{len(keys)} itens com prazo de validade")
    if not args.apply:
        print("nada alterado: use --apply para aplicar")
        return
    for key in keys:
        table.update_item(Key=key, UpdateExpression="REMOVE expires_at")
    print(f"{len(keys)} itens atualizados")


if __name__ == "__main__":
    main()
