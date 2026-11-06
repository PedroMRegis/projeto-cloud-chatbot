"""Quem pode abrir uma conversa.

Conversa de visitante não tem dono: vale o identificador, como sempre foi. A
partir do primeiro turno com login, ela passa a ser da conta, e só a conta abre.
"""

from __future__ import annotations

from app.adapters.conversations import ConversationRepository
from app.middlewares.validate_auth_token import SessionUser


class ConversationNotFound(Exception):
    """Para quem não é o dono, a conversa simplesmente não existe."""


class ConversationAccess:
    def __init__(self, conversations: ConversationRepository) -> None:
        self._conversations = conversations

    def ensure_can_read(self, conversation_id: str, user: SessionUser | None) -> None:
        owner = self._conversations.owner(conversation_id)
        if owner and (user is None or owner != user.user_id):
            raise ConversationNotFound(conversation_id)

    def ensure_can_write(self, conversation_id: str, user: SessionUser | None) -> None:
        self.ensure_can_read(conversation_id, user)
        if user and self._conversations.claim(conversation_id, user.user_id) != user.user_id:
            # Outra conta assumiu a conversa no mesmo instante.
            raise ConversationNotFound(conversation_id)
