"""As funções reais que o vendedor executa, todas via backend-api.

O enunciado (§4) exige que a IA use funções reais para consultar o catálogo e
registrar a transação. Cada tool aqui é uma chamada HTTP ao backend, que é o
dono do PostgreSQL: o chatbot não tem credencial de banco nenhuma.

Regra de segurança central: **o LLM nunca define preço**. `create_quote` só
envia produtos, quantidades e um desconto *pedido*; o backend calcula o resto.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from app.adapters import backend
from app.adapters.backend import BackendError
from app.application.session import order_card, product_card, quote_card
from app.shared import metrics
from app.shared.logging import get_logger

log = get_logger(__name__)

# Espelha o limite do backend só para orientar o LLM; quem valida é o backend.
MAX_QTY_PER_ITEM = 10

# Schemas no formato de tool calling da OpenAI, aceito também por Gemini e Groq.
TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_products",
            "description": (
                "Busca livros no catálogo por título, autor, categoria ou tema "
                "(a busca entende descrições como 'livro sobre vigilância'). "
                "Use sempre que o cliente descrever o que procura."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Termos de busca"},
                    "category": {"type": "string", "description": "Filtro opcional"},
                    "max_results": {"type": "integer", "minimum": 1, "maximum": 10},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_product",
            "description": "Detalhes e sinopse de um livro pelo product_id.",
            "parameters": {
                "type": "object",
                "properties": {"product_id": {"type": "string"}},
                "required": ["product_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_availability",
            "description": "Confere o estoque atual de um livro. Use antes de criar a cotação.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {"type": "string"},
                    "quantity": {"type": "integer", "minimum": 1},
                },
                "required": ["product_id", "quantity"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_quote",
            "description": (
                "Cria uma cotação com preço e desconto calculados pelo sistema. "
                "Informe o desconto que o cliente pediu em requested_discount_pct; "
                "o sistema aplica o teto permitido."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "product_id": {"type": "string"},
                                "quantity": {"type": "integer", "minimum": 1,
                                             "maximum": MAX_QTY_PER_ITEM},
                            },
                            "required": ["product_id", "quantity"],
                        },
                    },
                    "requested_discount_pct": {"type": "integer", "minimum": 0, "maximum": 100},
                },
                "required": ["items"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "confirm_order",
            "description": (
                "Registra o pedido a partir de uma cotação. Só chame depois de o "
                "cliente confirmar explicitamente. O nome do pedido vem da conta do cliente."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "quote_id": {"type": "string"},
                },
                "required": ["quote_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_order_status",
            "description": "Consulta a situação de um pedido já registrado.",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
            },
        },
    },
]


class ToolBox:
    """Executor das tools, amarrado a uma conversa.

    Além de devolver o resultado para o LLM, registra o que a interface deve
    mostrar (cards de livro, cotação, pedido) e o histórico de ações.
    """

    def __init__(self, conversation_id: str, logged_in: bool) -> None:
        self.conversation_id = conversation_id
        self.logged_in = logged_in
        # A interface mostra "Entrar" e "Criar conta" quando isto fica verdadeiro.
        self.login_required = False
        self.actions: list[dict[str, Any]] = []
        self.products: list[dict[str, Any]] = []
        self.product: dict[str, Any] | None = None
        self.quote: dict[str, Any] | None = None
        self.order: dict[str, Any] | None = None

    @property
    def dispatch_table(self) -> dict[str, Callable[..., dict[str, Any]]]:
        return {
            "search_products": self.search_products,
            "get_product": self.get_product,
            "check_availability": self.check_availability,
            "create_quote": self.create_quote,
            "confirm_order": self.confirm_order,
            "get_order_status": self.get_order_status,
        }

    def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Executa a tool. `BackendUnavailable` propaga: o turno inteiro degrada."""
        handler = self.dispatch_table.get(name)
        if handler is None:
            result: dict[str, Any] = {"error": f"tool desconhecida: {name}"}
        else:
            try:
                with metrics.timed("ToolLatency", dim_tool=name):
                    result = handler(**arguments)
            except BackendError as exc:
                log.info("backend recusou a operação",
                         extra={"tool": name, "status": exc.status, "reason": exc.detail})
                result = {"error": exc.detail, "status": exc.status}
                self.login_required = self.login_required or exc.status == 401
            except TypeError as exc:
                # Argumentos inválidos vindos do LLM: devolver como erro é melhor
                # que estourar, porque o modelo costuma corrigir na iteração seguinte.
                log.warning("argumentos inválidos", extra={"tool": name, "error": str(exc)})
                result = {"error": f"argumentos inválidos para {name}: {exc}"}
        self.actions.append({"tool": name, "args": arguments, "result_summary": _summary(result)})
        return result

    # ------------------------------------------------------------- consultas

    def search_products(self, query: str, category: str | None = None,
                        max_results: int = 5) -> dict[str, Any]:
        found = backend.search_products(query, category, max(1, min(int(max_results), 10)))
        cards = [product_card(p) for p in found["products"]]
        self.products = cards
        return {"count": len(cards), "products": cards}

    def get_product(self, product_id: str) -> dict[str, Any]:
        product = backend.get_product(product_id)
        self.product = product_card(product)
        return {**self.product, "description": product.get("description", "")}

    def check_availability(self, product_id: str, quantity: int) -> dict[str, Any]:
        return backend.check_availability(product_id, max(1, int(quantity)))

    # -------------------------------------------------------- cotação e pedido

    def create_quote(self, items: list[dict[str, Any]],
                     requested_discount_pct: int = 0) -> dict[str, Any]:
        quote = backend.create_quote(self.conversation_id, items, int(requested_discount_pct or 0))
        self.quote = quote_card(quote)
        metrics.emit("QuotesRequested", 1, discount_pct=self.quote["discount_pct"])
        return self.quote

    def confirm_order(self, quote_id: str) -> dict[str, Any]:
        if not self.logged_in:
            self.login_required = True
            return {"error": "o cliente precisa entrar na conta para concluir o pedido",
                    "status": 401}
        order = backend.place_order(quote_id, self.conversation_id)
        self.order = order_card(order)
        if order.get("created"):
            metrics.emit("OrdersPlacedByChat", 1)
        return {**self.order, "already_existed": not order.get("created", True)}

    def get_order_status(self, order_id: str) -> dict[str, Any]:
        order = backend.get_order(order_id, self.conversation_id)
        self.order = order_card(order)
        return self.order


def _summary(result: dict[str, Any]) -> str:
    """Resumo curto para a UI e os logs, sem despejar o payload inteiro."""
    if "error" in result:
        return f"erro: {result['error']}"
    if "count" in result:
        return f"{result['count']} resultado(s)"
    for key in ("order_id", "quote_id", "available", "status", "product_id"):
        if key in result:
            return f"{key}={result[key]}"
    return "ok"
