"""Caso de uso de um turno de conversa.

Na ordem em que acontece:

1. **Rate limit** por conversa, antes de qualquer gasto de cota de LLM.
2. **Idempotência por `request_id`**: retry de rede ou duplo clique devolvem a
   resposta armazenada, sem chamar o LLM de novo e sem segundo pedido.
3. Carrega histórico e **estado da sessão**.
4. Executa o turno:
   - ação estruturada (botão) → vendedor guiado, em qualquer modo;
   - texto com `LLM_PROVIDER=mock` → vendedor guiado interpretando o texto;
   - texto com LLM real → agente com tool calling + estado da sessão.
5. Persiste mensagens (com o `ui` da resposta), estado e resposta armazenada.

Se o backend estiver fora, o turno degrada com HTTP 200 e uma explicação —
nunca 5xx — e o `request_id` é liberado para o cliente poder tentar de novo.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.adapters.backend import BackendUnavailable
from app.adapters.conversations import ConversationRepository
from app.application import agent
from app.application.seller import Seller
from app.application.session import Session
from app.application.tools import ToolBox
from app.shared import metrics
from app.shared.config import settings
from app.shared.logging import get_logger

log = get_logger(__name__)

# Turnos de histórico enviados ao LLM. Mais que isso infla o custo por token.
HISTORY_TURNS = 12

BACKEND_DOWN_REPLY = (
    "Nosso catálogo está fora do ar por alguns instantes, então não consigo consultar "
    "livros nem registrar pedidos agora. Nada do que você já fez se perdeu — tente de "
    "novo em alguns segundos."
)


class RateLimited(Exception):
    pass


class RequestInFlight(Exception):
    """Mesmo `request_id` já está sendo processado por outra task."""


@dataclass
class ChatResponse:
    conversation_id: str
    reply: str
    actions: list[dict[str, Any]] = field(default_factory=list)
    products: list[dict[str, Any]] = field(default_factory=list)
    quote: dict[str, Any] | None = None
    order: dict[str, Any] | None = None
    suggestions: list[dict[str, Any]] = field(default_factory=list)
    input_hint: str = ""
    stage: str = "idle"
    degraded: bool = False
    replayed: bool = False

    @property
    def order_id(self) -> str | None:
        return self.order["order_id"] if self.order else None

    def ui(self) -> dict[str, Any]:
        return {
            "products": self.products,
            "quote": self.quote,
            "order": self.order,
            "suggestions": self.suggestions,
            "input_hint": self.input_hint,
            "stage": self.stage,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "reply": self.reply,
            "actions": self.actions,
            "order_id": self.order_id,
            "degraded": self.degraded,
            "replayed": self.replayed,
            **self.ui(),
        }


def _to_llm_history(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"role": item["role"], "content": item.get("content", "")}
        for item in items
        if item.get("role") in ("user", "assistant")
    ]


def _cards(toolbox: ToolBox) -> list[dict[str, Any]]:
    # Livro em foco tem prioridade sobre a lista da busca: é dele que se fala.
    if toolbox.product:
        return [toolbox.product]
    return toolbox.products


def handle_turn(
    conversation_id: str,
    request_id: str,
    message: str,
    action: dict[str, Any] | None = None,
) -> ChatResponse:
    cfg = settings()
    conversations = ConversationRepository()

    if conversations.hit_rate_limit(conversation_id, cfg.rate_limit_per_minute):
        metrics.emit("RateLimited", 1)
        log.warning("conversa excedeu o rate limit")
        raise RateLimited(f"limite de {cfg.rate_limit_per_minute} mensagens por minuto excedido")

    if not conversations.claim_request(conversation_id, request_id):
        cached = conversations.get_cached_response(conversation_id, request_id)
        if cached is not None:
            log.info("request_id repetido, devolvendo resposta armazenada")
            metrics.emit("ReplayedRequests", 1)
            return ChatResponse(**{**cached, "replayed": True})
        raise RequestInFlight(request_id)

    try:
        return _run(conversations, conversation_id, request_id, message, action)
    except Exception:
        conversations.release_request(conversation_id, request_id)
        raise


def _run(
    conversations: ConversationRepository,
    conversation_id: str,
    request_id: str,
    message: str,
    action: dict[str, Any] | None,
) -> ChatResponse:
    cfg = settings()
    history = conversations.history(conversation_id, limit=HISTORY_TURNS * 2)
    session = Session.load(conversations.get_state(conversation_id))
    toolbox = ToolBox(conversation_id)
    seller = Seller(toolbox, session)
    mode = "action" if action else ("guided" if cfg.llm_provider == "mock" else "llm")
    log.info("turno recebido",
             extra={"message_chars": len(message), "history_messages": len(history),
                    "stage": session.stage, "mode": mode,
                    "action": (action or {}).get("type")})

    degraded = False
    try:
        if action:
            reply = seller.handle_action(action)
        elif cfg.llm_provider == "mock":
            reply = seller.handle_text(message)
        else:
            result = agent.run_turn(_to_llm_history(history), message, session, toolbox)
            reply, degraded = result.reply, result.degraded
            agent.sync_session(session, toolbox)
    except BackendUnavailable:
        log.error("backend indisponível, turno degradado")
        metrics.emit("DegradedTurns", 1, reason="backend")
        conversations.release_request(conversation_id, request_id)
        return ChatResponse(
            conversation_id=conversation_id, reply=BACKEND_DOWN_REPLY,
            suggestions=[{"label": "Tentar de novo", "message": message,
                          **({"action": action} if action else {})}],
            stage=session.stage, input_hint=session.input_hint(), degraded=True,
        )

    if degraded:
        metrics.emit("DegradedTurns", 1, reason="llm")

    response = ChatResponse(
        conversation_id=conversation_id,
        reply=reply,
        actions=toolbox.actions,
        products=_cards(toolbox),
        quote=toolbox.quote,
        order=toolbox.order,
        suggestions=session.suggestions(),
        input_hint=session.input_hint(),
        stage=session.stage,
        degraded=degraded,
    )

    index = len(history)
    conversations.append_message(conversation_id, index, "user", message,
                                 action=action or None)
    conversations.append_message(conversation_id, index + 1, "assistant", reply,
                                 ui=response.ui(),
                                 tools=[a["tool"] for a in toolbox.actions] or None)
    conversations.save_state(conversation_id, session.dump())
    cached = {k: v for k, v in response.to_dict().items() if k not in ("replayed", "order_id")}
    conversations.save_response(conversation_id, request_id, cached)
    metrics.emit("ChatTurns", 1, dim_mode=mode, stage=session.stage)
    return response
