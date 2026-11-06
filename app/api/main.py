"""chatbot: o vendedor conversacional (serviço ECS síncrono, FastAPI).

Dono só das conversas (DynamoDB). Catálogo, cotação e pedido são do backend,
acessados por HTTP. Assim uma instabilidade do LLM não afeta o backend, e um
pico de conversas escala este serviço sem escalar o backend.

`/health` não checa dependências (nem DynamoDB, nem backend, nem LLM): se
checasse, uma falha do backend derrubaria também todas as tasks do chatbot no
ALB. O diagnóstico fica em `/ready`.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Any, Literal

from fastapi import Depends, FastAPI, HTTPException, Path, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.adapters import backend
from app.adapters.conversations import ConversationRepository
from app.application.access import ConversationAccess, ConversationNotFound
from app.application.chat import RateLimited, RequestInFlight, handle_turn
from app.application.session import Session
from app.middlewares.validate_auth_token import SessionUser, optional_user, require_user
from app.shared import context, metrics
from app.shared.config import settings
from app.shared.logging import configure_logging, get_logger

configure_logging()
log = get_logger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    cfg = settings()
    log.info("chatbot iniciado",
             extra={"env": cfg.env, "region": cfg.region, "llm_provider": cfg.llm_provider,
                    "backend_url": cfg.backend_url})
    metrics.emit("ServiceStart", 1)
    yield
    log.info("chatbot encerrando")


app = FastAPI(title="ConvCommerce — chatbot", docs_url="/api/chat/docs",
              openapi_url="/api/chat/openapi.json", redoc_url=None, lifespan=lifespan)

_conversations = ConversationRepository()
_access = ConversationAccess(_conversations)

# Sem `#`: o identificador vira chave no DynamoDB e não pode imitar `USER#<id>`.
CONVERSATION_ID_PATTERN = r"^[\w.-]{1,64}$"
ConversationId = Annotated[str, Path(pattern=CONVERSATION_ID_PATTERN)]


@app.middleware("http")
async def correlation_middleware(request: Request, call_next: Any) -> Response:
    incoming = request.headers.get("x-request-id") or context.new_id()
    tokens = context.bind(correlation_id=incoming, traceparent=request.headers.get("traceparent"))
    started = time.perf_counter()
    try:
        response = await call_next(request)
    finally:
        context.unbind(tokens)
    response.headers["X-Request-Id"] = incoming
    if request.url.path.startswith("/api/"):
        log.info("requisição concluída",
                 extra={"method": request.method, "path": request.url.path,
                        "status": response.status_code,
                        "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                        "correlation_id": incoming})
    return response


@app.exception_handler(ConversationNotFound)
async def conversation_not_found(_: Request, __: ConversationNotFound) -> JSONResponse:
    return JSONResponse({"detail": "conversa não encontrada"}, status_code=404)


# ------------------------------------------------------------- health checks

_ready_cache: dict[str, Any] = {"at": 0.0, "body": {}, "ok": False}
_READY_TTL_S = 10


@app.get("/health", include_in_schema=False)
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready", include_in_schema=False)
def ready() -> JSONResponse:
    now = time.monotonic()
    if now - _ready_cache["at"] >= _READY_TTL_S:
        checks: dict[str, str] = {}
        try:
            _conversations.get_state("__probe__")
            checks["dynamodb"] = "ok"
        except Exception as exc:  # noqa: BLE001
            checks["dynamodb"] = type(exc).__name__
        checks["backend"] = "ok" if backend.health() else "unreachable"
        checks["llm"] = settings().llm_provider
        ok = checks["dynamodb"] == "ok" and checks["backend"] == "ok"
        _ready_cache.update(at=now, ok=ok, body=checks)
    status = 200 if _ready_cache["ok"] else 503
    return JSONResponse({"status": "ready" if status == 200 else "degraded",
                         **_ready_cache["body"]}, status_code=status)


# ---------------------------------------------------------------------- chat

ActionType = Literal[
    "select_product", "set_quantity", "change_quantity", "request_discount", "confirm",
    "set_name", "details", "order_status", "browse_category", "search", "surprise", "back",
    "cancel", "reset", "start",
]


class ChatAction(BaseModel):
    """Resposta rápida estruturada (botão). Sem ambiguidade de interpretação."""

    type: ActionType
    product_id: str | None = Field(default=None, max_length=32)
    quantity: int | None = Field(default=None, ge=1, le=10)
    discount_pct: int | None = Field(default=None, ge=0, le=100)
    category: str | None = Field(default=None, max_length=60)
    query: str | None = Field(default=None, max_length=200)
    name: str | None = Field(default=None, max_length=80)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    request_id: str = Field(min_length=8, max_length=64)
    conversation_id: str | None = Field(default=None, pattern=CONVERSATION_ID_PATTERN)
    action: ChatAction | None = None


@app.post("/api/chat")
def chat(payload: ChatRequest,
         user: SessionUser | None = Depends(optional_user)) -> dict[str, Any]:
    conversation_id = payload.conversation_id or context.new_id()
    _access.ensure_can_write(conversation_id, user)
    tokens = context.bind(conversation_id=conversation_id)
    session_token = context.bind_auth_token(user.token if user else None)
    try:
        response = handle_turn(
            conversation_id, payload.request_id, payload.message,
            payload.action.model_dump(exclude_none=True) if payload.action else None,
            user.user_id if user else None,
        )
    except RateLimited as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except RequestInFlight as exc:
        # 409: o cliente deve esperar, não reenviar com outro request_id.
        raise HTTPException(status_code=409, detail="requisição em processamento, aguarde") from exc
    finally:
        context.unbind_auth_token(session_token)
        context.unbind(tokens)
    return response.to_dict()


@app.get("/api/conversations")
def list_conversations(user: SessionUser = Depends(require_user)) -> dict[str, Any]:
    """Conversas da conta logada. Não existe rota que liste as de outra pessoa."""
    return {"conversations": _conversations.list_for_user(user.user_id)}


@app.get("/api/conversations/{conversation_id}")
def get_conversation(conversation_id: ConversationId,
                     user: SessionUser | None = Depends(optional_user)) -> dict[str, Any]:
    """Histórico com o `ui` de cada resposta, para o frontend restaurar a conversa."""
    _access.ensure_can_read(conversation_id, user)
    messages = _conversations.history(conversation_id, limit=100)
    if not messages:
        raise HTTPException(status_code=404, detail="conversa não encontrada")
    session = Session.load(_conversations.get_state(conversation_id))
    return {
        "conversation_id": conversation_id,
        "messages": [
            {"role": m["role"], "content": m.get("content", ""), "at": m["sk"][4:],
             **({"ui": m["ui"]} if m.get("ui") else {})}
            for m in messages
        ],
        "stage": session.stage,
        "order": session.order,
        "suggestions": session.suggestions(),
        "input_hint": session.input_hint(),
    }


@app.post("/api/conversations/{conversation_id}/start")
def start(conversation_id: ConversationId,
          user: SessionUser | None = Depends(optional_user)) -> dict[str, Any]:
    """Saudação inicial sem gastar turno: só as sugestões da etapa atual."""
    _access.ensure_can_read(conversation_id, user)
    session = Session.load(_conversations.get_state(conversation_id))
    return {"conversation_id": conversation_id, "stage": session.stage,
            "suggestions": session.suggestions(), "input_hint": session.input_hint()}
