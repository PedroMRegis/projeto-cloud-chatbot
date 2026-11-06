"""Estado da sessão de venda e o que a interface mostra em cada etapa.

O bug que motivou este módulo: o vendedor sugeria um livro, perguntava "quer
levar?", o cliente respondia "sim" e o "sim" virava uma *busca* por "sim". O
vendedor não sabia a que pergunta o "sim" respondia.

Agora cada conversa tem um estado persistido (DynamoDB, item `STATE`):

    idle ──busca──► browsing ──escolhe──► product ──quantidade/sim──► quoted
      ▲                                     │                          │ sim
      │                                     ◄────── não/outras ────────┤
      │                                                                ▼
      └────────────── comprar outro ◄────── ordered ◄──── sim, com login ──┘

O estado guarda o livro em foco, a cotação e o pedido, e cada etapa oferece
**sugestões clicáveis** (com uma `action` estruturada) para o cliente não
precisar adivinhar o que digitar. Texto livre continua funcionando.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

IDLE = "idle"
BROWSING = "browsing"
PRODUCT = "product"
QUOTED = "quoted"
ORDERED = "ordered"

STAGES = (IDLE, BROWSING, PRODUCT, QUOTED, ORDERED)

# Atalhos da tela inicial. `category` casa com o catálogo do backend.
START_SHORTCUTS: list[tuple[str, str]] = [
    ("Clássicos", "clássico"),
    ("Ficção", "ficção"),
    ("Fantasia", "fantasia"),
    ("Não-ficção", "não-ficção"),
    ("Tecnologia", "tecnologia"),
]


def product_card(product: dict[str, Any]) -> dict[str, Any]:
    """Versão enxuta do produto para a UI e para o estado (sem a sinopse)."""
    return {
        "product_id": product["product_id"],
        "title": product.get("title"),
        "author": product.get("author"),
        "category": product.get("category"),
        "tagline": product.get("tagline", ""),
        "original_year": product.get("original_year"),
        "price": product.get("price"),
        "price_cents": product.get("price_cents"),
        "stock": product.get("stock"),
        "in_stock": bool(product.get("in_stock", (product.get("stock") or 0) > 0)),
    }


def quote_card(quote: dict[str, Any]) -> dict[str, Any]:
    return {
        "quote_id": quote["quote_id"],
        "items": [
            {k: i.get(k) for k in ("product_id", "title", "quantity", "unit_price")}
            for i in quote.get("items", [])
        ],
        "subtotal": quote.get("subtotal"),
        "discount_pct": quote.get("discount_pct", 0),
        "max_discount_pct": quote.get("max_discount_pct", 0),
        "total": quote.get("total"),
        "total_cents": quote.get("total_cents"),
        "note": quote.get("note"),
        "expires_at": quote.get("expires_at"),
    }


def order_card(order: dict[str, Any]) -> dict[str, Any]:
    return {
        "order_id": order["order_id"],
        "status": order.get("status"),
        "total": order.get("total"),
        "items": [
            {k: i.get(k) for k in ("product_id", "title", "quantity")}
            for i in order.get("items", [])
        ],
        "reason": order.get("reason"),
        "explanation": order.get("explanation"),
    }


def suggestion(label: str, action: dict[str, Any] | None = None,
               message: str | None = None) -> dict[str, Any]:
    """Um botão de resposta rápida. `message` é o que aparece no balão do cliente."""
    item: dict[str, Any] = {"label": label, "message": message or label}
    if action:
        item["action"] = action
    return item


@dataclass
class Session:
    stage: str = IDLE
    candidates: list[dict[str, Any]] = field(default_factory=list)
    product: dict[str, Any] | None = None
    quantity: int = 1
    quote: dict[str, Any] | None = None
    customer_name: str | None = None
    order: dict[str, Any] | None = None
    last_query: str | None = None

    @classmethod
    def load(cls, raw: dict[str, Any]) -> Session:
        session = cls()
        for key in cls.__dataclass_fields__:
            if key in raw and raw[key] is not None:
                setattr(session, key, raw[key])
        if session.stage not in STAGES:
            # Conversa antiga que esperava o nome: a cotação continua valendo.
            session.stage = QUOTED if session.quote else IDLE
        return session

    def dump(self) -> dict[str, Any]:
        return {key: getattr(self, key) for key in self.__dataclass_fields__}

    def reset(self) -> None:
        """Volta ao início, mas lembra o nome para o próximo pedido."""
        name = self.customer_name
        self.__init__()  # type: ignore[misc]
        self.customer_name = name

    # ------------------------------------------------------------ o que mostrar

    def input_hint(self) -> str:
        return {
            QUOTED: "Confirme, peça desconto ou mude a quantidade",
            PRODUCT: "Quantos exemplares?",
        }.get(self.stage, "Pergunte por título, autor ou assunto")

    def suggestions(self) -> list[dict[str, Any]]:
        if self.stage == BROWSING and self.candidates:
            options = [
                suggestion(f"{i}. {c['title']}", {"type": "select_product",
                                                  "product_id": c["product_id"]},
                           message=f"Quero ver {c['title']}")
                for i, c in enumerate(self.candidates[:3], start=1)
            ]
            return [*options, suggestion("Buscar outro livro", {"type": "reset"})]

        if self.stage == PRODUCT and self.product:
            stock = int(self.product.get("stock") or 0)
            options = [suggestion("Quero 1 exemplar", {"type": "set_quantity", "quantity": 1})]
            if stock >= 2:
                options.append(suggestion("Quero 2", {"type": "set_quantity", "quantity": 2}))
            if stock >= 3:
                options.append(suggestion("Quero 3 (até 5% off)",
                                          {"type": "set_quantity", "quantity": 3},
                                          message="Quero 3 exemplares"))
            options.append(suggestion("Sobre o que é?", {"type": "details"}))
            if len(self.candidates) > 1:
                options.append(suggestion("Ver outras opções", {"type": "back"}))
            return options

        if self.stage == QUOTED and self.quote:
            options = [suggestion("Confirmar pedido", {"type": "confirm"},
                                  message="Pode fechar o pedido")]
            pct = int(self.quote.get("max_discount_pct") or 0)
            if pct > int(self.quote.get("discount_pct") or 0):
                options.append(suggestion(f"Pedir {pct}% de desconto",
                                          {"type": "request_discount", "discount_pct": pct},
                                          message=f"Consegue {pct}% de desconto?"))
            elif self.quantity < 3 and int((self.product or {}).get("stock") or 0) >= 3:
                options.append(suggestion("Levar 3 com desconto",
                                          {"type": "set_quantity", "quantity": 3},
                                          message="E se eu levar 3?"))
            options.append(suggestion("Mudar quantidade", {"type": "change_quantity"}))
            options.append(suggestion("Cancelar", {"type": "cancel"}))
            return options

        if self.stage == ORDERED and self.order:
            return [
                suggestion("Ver status do pedido", {"type": "order_status"},
                           message="Como está meu pedido?"),
                suggestion("Comprar outro livro", {"type": "reset"}),
            ]

        return [
            *(suggestion(label, {"type": "browse_category", "category": category},
                         message=f"Quero ver {label.lower()}")
              for label, category in START_SHORTCUTS),
            suggestion("Me surpreenda", {"type": "surprise"},
                       message="Me recomenda um livro?"),
        ]

    def context_for_llm(self) -> str:
        """Resumo do estado para o LLM não perder o fio da conversa."""
        parts = [f"etapa: {self.stage}"]
        if self.product:
            parts.append(
                f"livro em foco: {self.product['title']} (product_id={self.product['product_id']}, "
                f"{self.product.get('price')}, estoque {self.product.get('stock')})"
            )
        if self.quote:
            parts.append(
                f"cotação ativa: quote_id={self.quote['quote_id']}, "
                f"total {self.quote.get('total')}, desconto {self.quote.get('discount_pct')}%"
            )
        if self.customer_name:
            parts.append(f"nome do cliente: {self.customer_name}")
        if self.order:
            parts.append(f"pedido: order_id={self.order['order_id']} ({self.order.get('status')})")
        if self.candidates and self.stage == BROWSING:
            listed = ", ".join(f"{c['title']} ({c['product_id']})" for c in self.candidates[:5])
            parts.append(f"opções mostradas: {listed}")
        return "Estado da sessão — " + "; ".join(parts) + "."
