"""Cria a tabela `conversations` (PK conversation_id, SK sk, TTL expires_at).

Usado em local (DynamoDB Local) e nos testes (moto). Na AWS a tabela é criada
pelo console/CLI em Ohio e pelo Terraform em São Paulo, com o mesmo schema.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any

import boto3

TABLE = os.environ.get("TABLE_CONVERSATIONS", "conversations")


def create_table(client: Any, name: str = TABLE) -> None:
    existing = client.list_tables().get("TableNames", [])
    if name not in existing:
        client.create_table(
            TableName=name,
            BillingMode="PAY_PER_REQUEST",
            KeySchema=[
                {"AttributeName": "conversation_id", "KeyType": "HASH"},
                {"AttributeName": "sk", "KeyType": "RANGE"},
            ],
            AttributeDefinitions=[
                {"AttributeName": "conversation_id", "AttributeType": "S"},
                {"AttributeName": "sk", "AttributeType": "S"},
            ],
        )
        client.get_waiter("table_exists").wait(TableName=name)
    try:
        client.update_time_to_live(
            TableName=name,
            TimeToLiveSpecification={"Enabled": True, "AttributeName": "expires_at"},
        )
    except client.exceptions.ClientError as exc:
        # Rodar de novo com TTL já ligado não é erro.
        if "TimeToLive is already enabled" not in str(exc):
            raise


def main() -> int:
    parser = argparse.ArgumentParser()
    # Sem região padrão: vem de --region ou de AWS_REGION (o compose define).
    parser.add_argument("--region", default=os.environ.get("AWS_REGION"))
    args = parser.parse_args()
    if not args.region:
        parser.error("informe --region ou defina AWS_REGION")
    kwargs: dict[str, Any] = {"region_name": args.region}
    if os.environ.get("DYNAMODB_ENDPOINT"):
        kwargs["endpoint_url"] = os.environ["DYNAMODB_ENDPOINT"]
    client = boto3.client("dynamodb", **kwargs)
    for attempt in range(30):
        try:
            create_table(client)
            print(f"tabela {TABLE} pronta")
            return 0
        except Exception as exc:  # noqa: BLE001
            print(f"DynamoDB indisponível ({type(exc).__name__}), tentativa {attempt + 1}",
                  file=sys.stderr)
            time.sleep(2)
    return 1


if __name__ == "__main__":
    sys.exit(main())
