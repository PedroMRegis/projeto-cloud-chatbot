"""Quem está conversando.

O chatbot não tem banco de usuários: ele confere a assinatura do cookie que o
backend emitiu, com o mesmo segredo. Sem cookie válido, a pessoa é visitante e
pode conversar, mas sem histórico de conversas.
"""

from __future__ import annotations

from dataclasses import dataclass

import jwt
from fastapi import Depends, HTTPException, Request

from app.shared.config import settings

COOKIE_NAME = "cirro_session"
ISSUER = "cirro-backend"
AUDIENCE = "cirro-web"


@dataclass(frozen=True)
class SessionUser:
    user_id: str
    token: str


def optional_user(request: Request) -> SessionUser | None:
    token = request.cookies.get(COOKIE_NAME)
    secret = settings().auth_jwt_secret
    if not token or not secret:
        return None
    try:
        payload = jwt.decode(token, secret, algorithms=["HS256"], audience=AUDIENCE,
                             issuer=ISSUER, leeway=30, options={"require": ["exp", "sub"]})
    except jwt.InvalidTokenError:
        return None
    return SessionUser(user_id=payload["sub"], token=token)


def require_user(user: SessionUser | None = Depends(optional_user)) -> SessionUser:
    if user is None:
        raise HTTPException(status_code=401, detail="faça login para continuar")
    return user
