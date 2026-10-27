"""Resiliência do LLM: retry, fallback, circuit breaker e resposta degradada."""

from __future__ import annotations

from typing import Any

import pytest

from app.adapters import llm
from app.shared import config


class FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content
        self.tool_calls = None


class FakeResponse:
    def __init__(self, content: str, provider: str) -> None:
        self.choices = [type("Choice", (), {"message": FakeMessage(content)})()]
        self.provider = provider


class HttpError(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


@pytest.fixture
def both_providers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "chave-gemini")
    monkeypatch.setenv("GROQ_API_KEY", "chave-groq")
    config.reset_settings_cache()
    llm.reset_breakers()


def stub_providers(
    monkeypatch: pytest.MonkeyPatch, behaviour: dict[str, Any]
) -> list[str]:
    """Substitui a chamada HTTP por um comportamento por provedor.

    O valor pode ser uma exceção (levantada), uma string (resposta) ou uma
    lista consumida a cada tentativa, o que permite testar o retry.
    """
    chamadas: list[str] = []

    def fake_call(provider: Any, messages: Any, tools: Any) -> Any:
        chamadas.append(provider.name)
        outcome = behaviour[provider.name]
        if isinstance(outcome, list):
            outcome = outcome.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return FakeResponse(str(outcome), provider.name)

    # Substitui apenas a camada de transporte: retry, breaker e fallback
    # continuam sendo o código real sob teste.
    monkeypatch.setattr(llm, "_call_provider", fake_call)
    return chamadas


class TestFallback:
    def test_primario_respondendo_nao_aciona_o_fallback(
        self, both_providers: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        chamadas = stub_providers(
            monkeypatch, {"gemini": "ok do gemini", "groq": "ok do groq"}
        )
        response = llm.complete([{"role": "user", "content": "oi"}])

        assert response.provider == "gemini"
        assert chamadas == ["gemini"]

    def test_falha_do_primario_cai_para_o_groq(
        self, both_providers: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        chamadas = stub_providers(
            monkeypatch, {"gemini": HttpError(500), "groq": "ok do groq"}
        )
        response = llm.complete([{"role": "user", "content": "oi"}])

        assert response.provider == "groq"
        assert chamadas == ["gemini", "groq"]

    def test_falha_dos_dois_levanta_llm_unavailable(
        self, both_providers: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        stub_providers(
            monkeypatch, {"gemini": HttpError(500), "groq": HttpError(429)}
        )
        with pytest.raises(llm.LLMUnavailable):
            llm.complete([{"role": "user", "content": "oi"}])

    def test_sem_chave_configurada_nao_ha_provedor(
        self, monkeypatch: pytest.MonkeyPatch, reset_caches: None
    ) -> None:
        monkeypatch.setenv("LLM_PROVIDER", "gemini")
        monkeypatch.setenv("GEMINI_API_KEY", "")
        monkeypatch.setenv("GROQ_API_KEY", "")
        config.reset_settings_cache()

        with pytest.raises(llm.LLMUnavailable):
            llm.complete([{"role": "user", "content": "oi"}])

    def test_provider_groq_inverte_a_ordem_de_tentativa(
        self, both_providers: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LLM_PROVIDER", "groq")
        config.reset_settings_cache()
        chamadas = stub_providers(
            monkeypatch, {"gemini": "g", "groq": "ok do groq"}
        )

        llm.complete([{"role": "user", "content": "oi"}])

        assert chamadas == ["groq"]


class TestCircuitBreaker:
    def test_abre_depois_do_limite_de_falhas(self) -> None:
        breaker = llm.CircuitBreaker("teste")
        for _ in range(llm.BREAKER_FAILURE_THRESHOLD - 1):
            breaker.record_failure()
        assert breaker.is_open() is False

        breaker.record_failure()
        assert breaker.is_open() is True

    def test_sucesso_zera_o_contador(self) -> None:
        breaker = llm.CircuitBreaker("teste")
        for _ in range(llm.BREAKER_FAILURE_THRESHOLD - 1):
            breaker.record_failure()
        breaker.record_success()
        breaker.record_failure()

        assert breaker.is_open() is False

    def test_fecha_depois_da_janela(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Half-open: passada a janela, uma tentativa sonda o provedor."""
        breaker = llm.CircuitBreaker("teste")
        for _ in range(llm.BREAKER_FAILURE_THRESHOLD):
            breaker.record_failure()
        assert breaker.is_open() is True

        agora = breaker.opened_at or 0.0
        monkeypatch.setattr(
            llm.time, "monotonic", lambda: agora + llm.BREAKER_OPEN_S + 1
        )
        assert breaker.is_open() is False

    def test_circuito_aberto_no_primario_vai_direto_ao_fallback(
        self, both_providers: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Economiza o timeout de 20 s em uma dependência já conhecida como quebrada."""
        for _ in range(llm.BREAKER_FAILURE_THRESHOLD):
            llm.breaker("gemini").record_failure()
        chamadas = stub_providers(
            monkeypatch, {"gemini": HttpError(500), "groq": "ok do groq"}
        )

        response = llm.complete([{"role": "user", "content": "oi"}])

        assert response.provider == "groq"
        assert "gemini" not in chamadas


class TestClassificacaoDeErro:
    @pytest.mark.parametrize("status", [429, 500, 502, 503])
    def test_erros_transitorios_permitem_retry(self, status: int) -> None:
        assert llm._is_retryable(HttpError(status)) is True

    @pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
    def test_erros_permanentes_nao_permitem_retry(self, status: int) -> None:
        """Repetir um 401 só queima tempo: a chave não vai ficar válida."""
        assert llm._is_retryable(HttpError(status)) is False

    def test_timeout_sem_status_permite_retry(self) -> None:
        assert llm._is_retryable(TimeoutError("estourou")) is True
