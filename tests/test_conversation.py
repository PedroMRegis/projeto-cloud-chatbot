"""Fluxos de conversa ponta a ponta (API → vendedor → tools → backend falso).

O primeiro teste reproduz o bug relatado: o vendedor sugeria um livro,
perguntava se o cliente queria, o cliente dizia "sim" e o "sim" virava busca.
"""

from __future__ import annotations

from typing import Any

from tests.conftest import Chat, FakeBackend


def labels(response: dict[str, Any]) -> list[str]:
    return [s["label"] for s in response["suggestions"]]


class TestContexto:
    def test_sim_responde_a_pergunta_anterior(self, chat: Chat, fake_backend: FakeBackend) -> None:
        r = chat.say("quero o dom casmurro")
        assert r["stage"] == "product"
        assert "Quer levar?" in r["reply"]
        assert r["products"][0]["product_id"] == "BK001"

        r = chat.say("sim")
        assert r["stage"] == "quoted", r["reply"]
        assert "não encontrei" not in r["reply"].lower()
        assert r["quote"]["items"][0]["product_id"] == "BK001"
        assert "search" not in fake_backend.calls[2:]  # o "sim" não virou busca

        r = chat.say("pode")
        assert r["stage"] == "awaiting_name"
        assert "nome" in r["reply"].lower()

        r = chat.say("Pedro Regis")
        assert r["stage"] == "ordered"
        assert r["order"]["status"] == "PENDING"
        assert "Pedido registrado, Pedro!" in r["reply"]
        assert fake_backend.orders[r["order_id"]]["customer_name"] == "Pedro Regis"

    def test_estado_sobrevive_entre_requisicoes(self, chat: Chat, api: Any) -> None:
        chat.say("1984")
        restored = api.get(f"/api/conversations/{chat.conversation_id}").json()
        assert restored["stage"] == "product"
        assert restored["messages"][-1]["ui"]["products"][0]["product_id"] == "BK016"
        assert "Quero 1 exemplar" in [s["label"] for s in restored["suggestions"]]

    def test_nome_dito_antes_nao_e_perguntado_de_novo(self, chat: Chat) -> None:
        chat.say("meu nome é Ana Lima, quero o dom casmurro")
        chat.say("sim")
        r = chat.say("fechado")
        assert r["stage"] == "ordered"
        assert "Ana" in r["reply"]

    def test_status_do_pedido(self, chat: Chat) -> None:
        chat.say("dom casmurro")
        chat.say("2 exemplares")
        chat.say("sim")
        chat.say("Maria Souza")
        r = chat.say("como está meu pedido?")
        assert "confirmação de estoque" in r["reply"]


class TestEscolha:
    def test_lista_e_escolha_por_ordinal(self, chat: Chat) -> None:
        r = chat.say("tolkien")
        assert r["stage"] == "browsing"
        assert "2 opções" in r["reply"]
        assert labels(r)[:2] == ["1. O Hobbit", "2. O Senhor dos Anéis: A Sociedade do Anel"]

        r = chat.say("o segundo")
        assert r["products"][0]["product_id"] == "BK026"

        r = chat.say("não, quero ver outras")
        assert r["stage"] == "browsing"

        r = chat.say("1")
        assert r["products"][0]["product_id"] == "BK025"

    def test_sim_na_lista_escolhe_o_primeiro(self, chat: Chat) -> None:
        chat.say("orwell")
        r = chat.say("sim")
        assert r["stage"] == "product"
        assert r["products"][0]["product_id"] == "BK016"

    def test_busca_sem_resultado(self, chat: Chat) -> None:
        r = chat.say("astrofísica quântica")
        assert r["stage"] == "idle"
        assert "Não encontrei" in r["reply"]
        assert "Clássicos" in labels(r)

    def test_livro_esgotado(self, chat: Chat) -> None:
        r = chat.say("Esgotado")
        assert "esgotado" in r["reply"]
        assert r["stage"] != "product"


class TestQuantidadeEDesconto:
    def test_frase_completa_vai_direto_para_a_cotacao(self, chat: Chat) -> None:
        r = chat.say("quero 3 unidades de Dom Casmurro com 10% de desconto")
        assert r["stage"] == "quoted"
        assert r["quote"]["items"][0] == {"product_id": "BK001", "title": "Dom Casmurro",
                                          "quantity": 3, "unit_price": "R$ 39,90"}
        assert r["quote"]["discount_pct"] == 5
        r = chat.say("confirmo, meu nome é Maria Silva")
        assert r["stage"] == "ordered"

    def test_tres_exemplares_liberam_desconto(self, chat: Chat) -> None:
        chat.say("dom casmurro")
        r = chat.say("quero 3 exemplares")
        assert r["quote"]["max_discount_pct"] == 5
        assert "Pedir 5% de desconto" in labels(r)

        r = chat.click("Pedir 5%")
        assert r["quote"]["discount_pct"] == 5

        r = chat.say("e um desconto de 20%?")
        assert "máximo" in r["reply"]

    def test_poucos_exemplares_explica_a_politica(self, chat: Chat) -> None:
        chat.say("dom casmurro")
        chat.say("sim")
        r = chat.say("tem desconto?")
        assert "A partir de 3 exemplares" in r["reply"]

    def test_pedido_de_desconto_acima_do_teto_e_limitado(self, chat: Chat) -> None:
        chat.say("dom casmurro")
        chat.say("5 unidades")
        r = chat.say("faz 30%?")
        assert r["quote"]["discount_pct"] == 10
        assert "máximo de 10%" in r["reply"]

    def test_estoque_insuficiente_oferece_o_que_tem(self, chat: Chat) -> None:
        chat.say("vidas secas")
        r = chat.say("quero 5 unidades")
        assert "Tenho só 3" in r["reply"]
        r = chat.say("sim")
        assert r["quote"]["items"][0]["quantity"] == 3

    def test_mudar_quantidade_na_cotacao(self, chat: Chat) -> None:
        chat.say("1984")
        chat.say("sim")
        r = chat.say("2")
        assert r["quote"]["items"][0]["quantity"] == 2


class TestBotoes:
    def test_compra_inteira_so_com_botoes(self, chat: Chat, fake_backend: FakeBackend) -> None:
        r = chat.say("Quero ver fantasia",
                     {"type": "browse_category", "category": "fantasia"})
        assert r["stage"] == "browsing"
        chat.click("1.")
        chat.click("Quero 2")
        r = chat.click("Confirmar pedido")
        assert r["stage"] == "awaiting_name"
        r = chat.say("Joana Prado")
        assert r["stage"] == "ordered"
        assert len(fake_backend.orders) == 1
        r = chat.click("Ver status")
        assert "confirmação" in r["reply"]
        r = chat.click("Comprar outro")
        assert r["stage"] == "idle"

    def test_nome_lembrado_vira_botao(self, chat: Chat) -> None:
        chat.say("dom casmurro")
        chat.say("sim")
        chat.say("sim")
        chat.say("Pedro Regis")
        chat.click("Comprar outro")
        chat.say("1984")
        chat.say("sim")
        r = chat.click("Confirmar")
        # O nome já é conhecido: registra direto, sem perguntar de novo.
        assert r["stage"] == "ordered"

    def test_cancelar_volta_ao_inicio(self, chat: Chat) -> None:
        chat.say("dom casmurro")
        chat.say("sim")
        r = chat.click("Cancelar")
        assert r["stage"] == "idle"
        assert r["quote"] is None

    def test_sinopse(self, chat: Chat) -> None:
        chat.say("dom casmurro")
        r = chat.click("Sobre o que é?")
        assert "Sinopse longa de Dom Casmurro" in r["reply"]
        assert r["stage"] == "product"


class TestResiliencia:
    def test_mesmo_request_id_nao_repete_o_pedido(
        self, chat: Chat, fake_backend: FakeBackend
    ) -> None:
        chat.say("dom casmurro")
        chat.say("sim")
        chat.say("sim")
        first = chat.say("Carla Dias", request_id="req-repetido-123")
        again = chat.say("Carla Dias", request_id="req-repetido-123")
        assert again["replayed"] is True
        assert again["order_id"] == first["order_id"]
        assert len(fake_backend.orders) == 1

    def test_backend_fora_degrada_sem_5xx(self, chat: Chat, fake_backend: FakeBackend) -> None:
        fake_backend.down = True
        r = chat.say("dom casmurro", request_id="req-backend-fora")
        assert r["degraded"] is True
        assert "fora do ar" in r["reply"]
        assert labels(r) == ["Tentar de novo"]

        # O request_id foi liberado: o retry do cliente funciona.
        fake_backend.down = False
        r = chat.say("dom casmurro", request_id="req-backend-fora")
        assert r["degraded"] is False
        assert r["stage"] == "product"

    def test_ready_mostra_backend_fora(self, api: Any, fake_backend: FakeBackend) -> None:
        fake_backend.down = True
        body = api.get("/ready")
        assert body.status_code == 503
        assert body.json()["backend"] == "unreachable"

    def test_rate_limit(self, chat: Chat, api: Any, monkeypatch: Any) -> None:
        from app.shared.config import reset_settings_cache

        monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "2")
        reset_settings_cache()
        chat.say("oi")
        chat.say("oi")
        resp = api.post("/api/chat", json={"message": "oi", "request_id": "r" * 10,
                                           "conversation_id": chat.conversation_id})
        assert resp.status_code == 429


def test_saudacao_mostra_atalhos(chat: Chat) -> None:
    r = chat.say("oi")
    assert "livreiro" in r["reply"]
    assert labels(r) == ["Clássicos", "Ficção", "Fantasia", "Não-ficção", "Tecnologia",
                         "Me surpreenda"]


def test_start_sem_conversa(api: Any) -> None:
    body = api.post("/api/conversations/nova/start").json()
    assert body["stage"] == "idle"
    assert body["suggestions"]
