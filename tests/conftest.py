"""Fixtures: DynamoDB em memória (moto) e um backend falso em memória.

O backend falso substitui só as funções de `app.adapters.backend` que fazem
HTTP. Tools, vendedor guiado, estado da sessão e persistência das conversas
são o código real sob teste.
"""

from __future__ import annotations

import itertools
import sys
import time
import unicodedata
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import boto3
import jwt
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def app_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Credenciais falsas antes de tudo: nenhum teste alcança a AWS real."""
    for key, value in {
        "AWS_ACCESS_KEY_ID": "testing",
        "AWS_SECRET_ACCESS_KEY": "testing",
        "AWS_SESSION_TOKEN": "testing",
        "AWS_DEFAULT_REGION": "us-east-2",
        "AWS_REGION": "us-east-2",
        "APP_ENV": "test",
        "LLM_PROVIDER": "mock",
        "MOCK_LATENCY_MS": "0",
        "DYNAMODB_ENDPOINT": "",
        "FAULTS_PARAMETER": "",
        "BACKEND_URL": "http://backend.test",
        "GEMINI_API_KEY": "",
        "GROQ_API_KEY": "",
        "AUTH_JWT_SECRET": "segredo-de-teste-com-mais-de-trinta-e-dois-bytes",
    }.items():
        monkeypatch.setenv(key, value)
    _reset()
    yield
    _reset()


def _reset() -> None:
    from app.adapters import aws, backend, llm
    from app.shared import config, faults

    config.reset_settings_cache()
    aws.reset_clients()
    backend.reset()
    llm.reset_breakers()
    faults.reset_cache()


@pytest.fixture
def reset_caches() -> None:
    """Compatibilidade com os testes de LLM herdados; o reset já é autouse."""


@pytest.fixture
def dynamo() -> Iterator[Any]:
    from moto import mock_aws

    with mock_aws():
        sys.path.insert(0, str(ROOT / "scripts"))
        from provision_tables import create_table

        create_table(boto3.client("dynamodb", region_name="us-east-2"))
        yield boto3.resource("dynamodb", region_name="us-east-2").Table("conversations")


def _fold(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text.lower())
                   if unicodedata.category(c) != "Mn")


def _brl(cents: int) -> str:
    return f"R$ {cents / 100:,.2f}".replace(",", "_").replace(".", ",").replace("_", ".")


class FakeBackend:
    """Catálogo pequeno com as mesmas regras do backend real."""

    def __init__(self) -> None:
        self.products: dict[str, dict[str, Any]] = {}
        for pid, title, author, category, price, stock in [
            ("BK001", "Dom Casmurro", "Machado de Assis", "clássico", 3990, 15),
            ("BK004", "Vidas Secas", "Graciliano Ramos", "clássico", 3490, 3),
            ("BK016", "1984", "George Orwell", "ficção", 4790, 5),
            ("BK017", "A Revolução dos Bichos", "George Orwell", "ficção", 2990, 20),
            ("BK025", "O Hobbit", "J.R.R. Tolkien", "fantasia", 5290, 3),
            ("BK026", "O Senhor dos Anéis: A Sociedade do Anel", "J.R.R. Tolkien",
             "fantasia", 7990, 8),
            ("BK099", "Esgotado", "Ninguém", "ficção", 1000, 0),
        ]:
            self.products[pid] = {
                "product_id": pid, "title": title, "author": author, "category": category,
                "tagline": f"Tagline de {title}.", "description": f"Sinopse longa de {title}.",
                "original_year": 1900, "price_cents": price, "price": _brl(price),
                "stock": stock, "in_stock": stock > 0,
            }
        self.quotes: dict[str, dict[str, Any]] = {}
        self.orders: dict[str, dict[str, Any]] = {}
        self.calls: list[str] = []
        self.down = False
        self.account_name: str | None = "Ana Lima"
        self._ids = itertools.count(1)

    def _check(self, name: str) -> None:
        from app.adapters.backend import BackendUnavailable

        self.calls.append(name)
        if self.down:
            raise BackendUnavailable("fake fora do ar")

    def current_user(self) -> dict[str, Any]:
        self._check("me")
        return {"name": self.account_name}

    def search_products(self, query: str, category: str | None = None,
                        limit: int = 5) -> dict[str, Any]:
        self._check("search")
        q = _fold(query)
        found = [
            p for p in self.products.values()
            if (not category or p["category"] == category)
            and (not q or any(t in _fold(f"{p['title']} {p['author']}") for t in q.split()))
        ]
        return {"count": len(found[:limit]), "products": found[:limit]}

    def get_product(self, product_id: str) -> dict[str, Any]:
        from app.adapters.backend import BackendError

        self._check("get_product")
        if product_id not in self.products:
            raise BackendError(404, "produto não encontrado")
        return dict(self.products[product_id])

    def check_availability(self, product_id: str, quantity: int) -> dict[str, Any]:
        self._check("availability")
        stock = self.products[product_id]["stock"]
        return {"product_id": product_id, "requested": quantity, "stock": stock,
                "available": stock >= quantity}

    def create_quote(self, conversation_id: str, items: list[dict[str, Any]],
                     requested_discount_pct: int = 0) -> dict[str, Any]:
        self._check("quote")
        qty = sum(i["quantity"] for i in items)
        subtotal = sum(self.products[i["product_id"]]["price_cents"] * i["quantity"]
                       for i in items)
        ceiling = 10 if qty >= 5 or subtotal > 30000 else 5 if qty >= 3 else 0
        applied = min(requested_discount_pct, ceiling)
        total = subtotal - subtotal * applied // 100
        quote_id = str(uuid.uuid4())
        quote = {
            "quote_id": quote_id, "conversation_id": conversation_id,
            "items": [{"product_id": i["product_id"],
                       "title": self.products[i["product_id"]]["title"],
                       "quantity": i["quantity"],
                       "unit_price": self.products[i["product_id"]]["price"]} for i in items],
            "subtotal": _brl(subtotal), "discount_pct": applied, "max_discount_pct": ceiling,
            "total": _brl(total), "total_cents": total, "expires_at": "2099-01-01T00:00:00",
        }
        if requested_discount_pct > ceiling:
            quote["note"] = f"Aplicado o máximo de {ceiling}%."
        self.quotes[quote_id] = quote
        return quote

    def place_order(self, quote_id: str, conversation_id: str) -> dict[str, Any]:
        from app.adapters.backend import BackendError

        self._check("order")
        quote = self.quotes.get(quote_id)
        if not quote or quote["conversation_id"] != conversation_id:
            raise BackendError(404, "cotação não existe")
        if quote_id in self.orders:
            return {**self.orders[quote_id], "created": False}
        order = {"order_id": quote_id, "status": "PENDING", "total": quote["total"],
                 "items": quote["items"], "customer_name": self.account_name,
                 "conversation_id": conversation_id}
        self.orders[quote_id] = order
        return {**order, "created": True}

    def get_order(self, order_id: str, conversation_id: str) -> dict[str, Any]:
        self._check("get_order")
        return dict(self.orders[order_id])


@pytest.fixture
def fake_backend(monkeypatch: pytest.MonkeyPatch) -> FakeBackend:
    from app.adapters import backend

    fake = FakeBackend()
    for name in ("search_products", "get_product", "check_availability", "create_quote",
                 "place_order", "get_order", "current_user"):
        monkeypatch.setattr(backend, name, getattr(fake, name))
    monkeypatch.setattr(backend, "health", lambda: not fake.down)
    return fake


@pytest.fixture
def api(dynamo: Any, fake_backend: FakeBackend) -> Iterator[Any]:
    from fastapi.testclient import TestClient

    from app.api.main import app

    with TestClient(app) as client:
        yield client


def session_headers(user_id: str) -> dict[str, str]:
    token = jwt.encode(
        {"sub": user_id, "iss": "cirro-backend", "aud": "cirro-web",
         "exp": int(time.time()) + 3600},
        "segredo-de-teste-com-mais-de-trinta-e-dois-bytes", algorithm="HS256",
    )
    return {"Cookie": f"cirro_session={token}"}


class Chat:
    """Conversa de teste: mantém conversation_id e gera request_id a cada envio."""

    def __init__(self, client: Any) -> None:
        self.client = client
        self.headers = session_headers("conta-de-teste")
        self.conversation_id: str | None = None
        self.last: dict[str, Any] = {}

    def say(self, message: str, action: dict[str, Any] | None = None,
            request_id: str | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {"message": message,
                                   "request_id": request_id or uuid.uuid4().hex}
        if self.conversation_id:
            payload["conversation_id"] = self.conversation_id
        if action:
            payload["action"] = action
        response = self.client.post("/api/chat", json=payload, headers=self.headers)
        assert response.status_code == 200, response.text
        self.last = response.json()
        self.conversation_id = self.last["conversation_id"]
        return self.last

    def click(self, label_prefix: str) -> dict[str, Any]:
        """Clica numa sugestão da última resposta, como o frontend faria."""
        for item in self.last["suggestions"]:
            if item["label"].startswith(label_prefix):
                return self.say(item["message"], item.get("action"))
        labels = [s["label"] for s in self.last["suggestions"]]
        raise AssertionError(f"sugestão '{label_prefix}' não encontrada em {labels}")


@pytest.fixture
def chat(api: Any) -> Chat:
    """Cliente logado."""
    return Chat(api)
