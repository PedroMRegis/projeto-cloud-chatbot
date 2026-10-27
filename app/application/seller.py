"""Vendedor guiado: determinístico, com contexto, sobre as mesmas tools do LLM.

Atende em dois papéis:

1. **Ações estruturadas** (botões de sugestão do frontend). São comandos sem
   ambiguidade — escolher livro, quantidade, confirmar — e por isso são
   executadas aqui em qualquer modo, inclusive com LLM. Efeito colateral bom:
   a compra continua possível mesmo com o LLM fora do ar.
2. **Texto livre no modo `mock`** (sem LLM, usado em teste de carga e quando
   não há chave). As frases são interpretadas *de acordo com a etapa*: o mesmo
   "sim" significa "quero esse livro" na etapa `product` e "pode fechar" na
   etapa `quoted`. Esse era o bug do vendedor antigo, que tratava "sim" como
   busca.
"""

from __future__ import annotations

import random
import re
import time
import unicodedata
from typing import Any

from app.application.session import (
    AWAITING_NAME,
    BROWSING,
    IDLE,
    ORDERED,
    PRODUCT,
    QUOTED,
    START_SHORTCUTS,
    Session,
)
from app.application.tools import ToolBox
from app.shared import metrics
from app.shared.config import settings

MAX_QTY = 10


def fold(text: str) -> str:
    """Minúsculas sem acento, para comparar texto."""
    normalized = unicodedata.normalize("NFD", text.lower())
    return "".join(c for c in normalized if unicodedata.category(c) != "Mn")


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9%]+", fold(text))


NUM_WORDS = {"um": 1, "uma": 1, "dois": 2, "duas": 2, "tres": 3, "quatro": 4, "cinco": 5,
             "seis": 6, "sete": 7, "oito": 8, "nove": 9, "dez": 10}
QTY_UNITS = r"(?:unidades?|exemplares?|livros?|copias?|volumes?|x)"
_NUM = r"(\d{1,2}|um|uma|dois|duas|tres|quatro|cinco|seis|sete|oito|nove|dez)"
QTY_WITH_UNIT_RE = re.compile(rf"\b{_NUM}\s*{QTY_UNITS}\b")
QTY_VERB_RE = re.compile(
    r"\b(?:quero|levo|levar|leva|manda|compro|comprar|seriam|serao|sao|fico com|me ve|me da)\s+"
    r"(\d{1,2}|dois|duas|tres|quatro|cinco|seis|sete|oito|nove|dez)\b"
)
BARE_NUMBER_RE = re.compile(r"^\s*(?:o\s+|a\s+|numero\s+)?(\d{1,2})\s*[.!?]?\s*$")
PCT_RE = re.compile(r"(\d{1,2})\s*(?:%|por\s*cento)")
NAME_RE = re.compile(
    r"(?:meu nome (?:é|e)|me chamo|pode colocar|coloca|sou (?:o|a)|nome:?)\s+"
    r"([A-Za-zÀ-ÿ][A-Za-zÀ-ÿ'.\- ]{1,58})",
    re.IGNORECASE,
)

AFFIRM_START = {"sim", "s", "ss", "quero", "queria", "pode", "isso", "claro", "ok", "okay",
                "beleza", "blz", "bora", "fechado", "fecha", "fechar", "confirmo", "confirmar",
                "confirma", "aceito", "perfeito", "manda", "vamos", "com", "top", "otimo",
                "show", "uhum", "aham", "yes", "positivo", "certo", "exato", "levo", "compro",
                "esse", "essa", "este", "esta", "vou"}
AFFIRM_FILLER = AFFIRM_START | {"ser", "mesmo", "certeza", "por", "favor", "pfv", "pf", "o",
                                "a", "ele", "ela", "levar", "comprar", "pedido", "isso", "sim",
                                "e", "de", "livro", "registrar", "registra", "fazer", "faz",
                                "pode", "obrigado", "obrigada", "quero", "q", "ai", "entao",
                                "querer", "agora", "ja", "um", "uma", *NUM_WORDS,
                                "unidade", "unidades", "exemplar", "exemplares", "livros",
                                "copia", "copias", "x"}
DENY_START = {"nao", "n", "nope", "cancela", "cancelar", "desisto", "deixa", "esquece",
              "nenhum", "nenhuma", "nem", "negativo"}
OTHER_RE = re.compile(r"\b(outr[oa]s?|mais opcoes|alternativas?|diferente)\b")
DISCOUNT_RE = re.compile(r"(desconto|abatimento|mais barato|barat|melhor preco|melhora|"
                         r"faz por menos|promocao|\d{1,2}\s*%|por\s*cento)")
STATUS_RE = re.compile(r"(status|situacao|andamento|meu pedido|o pedido|chegou|"
                       r"foi confirmado|ja confirmou|deu certo)")
DETAILS_RE = re.compile(r"(sobre o que|do que se trata|sinopse|resumo|descricao|conta mais|"
                        r"fala mais|mais detalhes|detalhes|e bom\??$|vale a pena)")
GREETING_START = {"oi", "ola", "opa", "salve", "hey", "hello", "eai", "bom", "boa", "alo"}
THANKS = {"obrigado", "obrigada", "valeu", "vlw", "brigado", "grato", "grata"}
SURPRISE_RE = re.compile(r"(surpreend|recomend|indica|sugest|sugere|sugira|qualquer um)")
HELP_RE = re.compile(r"^(ajuda|help|como funciona|o que (voce|vc) faz|menu)\b")
ORDINALS = {"primeiro": 1, "primeira": 1, "1o": 1, "1a": 1, "segundo": 2, "segunda": 2,
            "2o": 2, "2a": 2, "terceiro": 3, "terceira": 3, "3o": 3, "3a": 3, "quarto": 4,
            "quarta": 4, "4o": 4, "quinto": 5, "quinta": 5, "5o": 5, "ultimo": -1, "ultima": -1}
SEARCH_STOPWORDS = {"quero", "queria", "procuro", "procurando", "gostaria", "comprar", "livro",
                    "livros", "tem", "voces", "vc", "vcs", "voce", "algum", "alguma", "me",
                    "mostra", "mostre", "busca", "buscar", "procurar", "ver", "um", "uma", "de",
                    "do", "da", "o", "a", "os", "as", "por", "favor", "e", "sobre", "ai",
                    "aí", "chamado", "titulo", "autor", "escrito", "tipo", "ler"}


def _parse_quantity(text: str) -> int | None:
    folded = fold(text)
    for regex in (QTY_WITH_UNIT_RE, QTY_VERB_RE):
        match = regex.search(folded)
        if match:
            raw = match.group(1)
            return int(raw) if raw.isdigit() else NUM_WORDS.get(raw)
    return None


def _is_pure_affirm(text: str) -> bool:
    tokens = words(text)
    if not tokens or tokens[0] not in AFFIRM_START:
        return False
    return all(t in AFFIRM_FILLER or t.isdigit() for t in tokens)


def _is_deny(text: str) -> bool:
    tokens = words(text)
    return bool(tokens) and tokens[0] in DENY_START


def _extract_name(text: str) -> str | None:
    match = NAME_RE.search(text)
    if not match:
        return None
    name = re.split(r"[,.!?;]| e (?:quero|pode)", match.group(1))[0]
    return _clean_name(name)


def _clean_name(raw: str) -> str | None:
    name = " ".join(raw.strip().split()[:5])
    if not 2 <= len(name) <= 60 or any(ch.isdigit() for ch in name):
        return None
    if not re.fullmatch(r"[A-Za-zÀ-ÿ'.\- ]+", name):
        return None
    tokens = words(name)
    if tokens and (tokens[0] in AFFIRM_START or tokens[0] in DENY_START):
        return None
    return name


def _first_name(name: str | None) -> str:
    return (name or "").split()[0] if name else ""


def _short_id(order_id: str) -> str:
    return order_id.split("-")[0].upper()


QUERY_NOISE = {"unidade", "unidades", "exemplar", "exemplares", "copia", "copias", "desconto",
               "com", "sem", "cento", "x", *NUM_WORDS}


def _clean_query(text: str) -> str:
    """Tira da busca o que é pedido (quantidade, desconto) e não livro."""
    tokens = [
        t for t in re.findall(r"[\wÀ-ÿ]+", re.sub(r"\d{1,2}\s*%", " ", text))
        if fold(t) not in SEARCH_STOPWORDS and fold(t) not in QUERY_NOISE and not t.isdigit()
    ]
    return " ".join(tokens) or text.strip()


def _strong_match(query: str, product: dict[str, Any]) -> bool:
    """O primeiro resultado é claramente o livro pedido (o título foi citado)?

    Palavras curtas e genéricas ("amor") não contam: aí o cliente vê a lista.
    """
    q = fold(query).strip()
    title = fold(product.get("title") or "")
    if q == title:
        return True
    return len(q) >= 5 and (q in title or title in q)


class Seller:
    def __init__(self, toolbox: ToolBox, session: Session) -> None:
        self.tools = toolbox
        self.s = session

    # ================================================================ ações

    def handle_action(self, action: dict[str, Any]) -> str:
        kind = action.get("type")
        metrics.emit("GuidedActions", 1, dim_action=str(kind))
        if kind == "select_product" and action.get("product_id"):
            return self.focus(str(action["product_id"]))
        if kind == "set_quantity":
            return self.quote(int(action.get("quantity") or 1))
        if kind == "change_quantity":
            return self.ask_quantity()
        if kind == "request_discount":
            return self.request_discount(action.get("discount_pct"))
        if kind == "confirm":
            return self.confirm()
        if kind == "set_name" and action.get("name"):
            name = _clean_name(str(action["name"]))
            return self.place(name) if name else self.ask_name()
        if kind == "details":
            return self.details()
        if kind == "order_status":
            return self.order_status()
        if kind == "browse_category" and action.get("category"):
            return self.browse_category(str(action["category"]))
        if kind == "search" and action.get("query"):
            return self.search(str(action["query"]))
        if kind == "surprise":
            return self.surprise()
        if kind == "back":
            return self.back()
        if kind == "cancel":
            return self.cancel()
        if kind == "reset":
            self.s.reset()
            return "Vamos lá! O que você gostaria de ler agora? Diga um título, autor ou assunto."
        return self.welcome()

    # ========================================================== texto livre

    def handle_text(self, text: str) -> str:
        _simulate_latency()
        s = self.s
        tokens = words(text)
        folded = fold(text).strip()

        if not tokens:
            return "Não entendi. Pode repetir?"

        if HELP_RE.search(folded):
            return self.help()

        if tokens[0] in GREETING_START and len(tokens) <= 4 and not _parse_quantity(text):
            return self.welcome()

        if all(t in THANKS or t in {"muito", "obg", "brigadao"} for t in tokens):
            return "Imagina! Se quiser mais alguma coisa, é só chamar."

        if s.order and STATUS_RE.search(folded) and s.stage in (ORDERED, IDLE):
            return self.order_status()

        if s.stage == AWAITING_NAME:
            if _is_deny(text):
                return self.back()
            name = _extract_name(text) or _clean_name(text)
            return self.place(name) if name else self.ask_name(retry=True)

        name = _extract_name(text)
        if name:
            s.customer_name = name

        if s.product and DETAILS_RE.search(folded) and s.stage in (PRODUCT, QUOTED):
            return self.details()

        if DISCOUNT_RE.search(folded) and s.stage in (PRODUCT, QUOTED):
            pct = PCT_RE.search(folded)
            return self.request_discount(int(pct.group(1)) if pct else None)

        if s.stage == BROWSING:
            chosen = self._choose_candidate(text)
            if chosen:
                return self.focus(chosen)

        quantity = _parse_quantity(text)
        bare = BARE_NUMBER_RE.match(folded)
        if quantity is None and bare and s.stage in (PRODUCT, QUOTED):
            quantity = int(bare.group(1))
        if quantity is not None and s.stage in (PRODUCT, QUOTED):
            return self.quote(quantity)

        if _is_pure_affirm(text):
            return self._affirm(quantity)

        if _is_deny(text) or (OTHER_RE.search(folded) and len(tokens) <= 5):
            return self._deny()

        if SURPRISE_RE.search(folded) and len(tokens) <= 8:
            return self.surprise()

        if s.stage == QUOTED and name and re.search(r"(fecha|confirm|registr|pode)", folded):
            return self.confirm()

        reply = self.search(_clean_query(text))
        # Frase completa ("quero 3 do Dom Casmurro com 10%"): se a busca achou
        # o livro, a quantidade e o desconto pedidos já viram cotação.
        if s.stage == PRODUCT and quantity:
            pct = PCT_RE.search(folded)
            return self.quote(quantity, int(pct.group(1)) if pct else 0)
        return reply

    def _affirm(self, quantity: int | None) -> str:
        s = self.s
        if s.stage == PRODUCT:
            return self.quote(quantity or s.quantity or 1)
        if s.stage == QUOTED:
            return self.confirm()
        if s.stage == BROWSING and s.candidates:
            return self.focus(s.candidates[0]["product_id"])
        if s.stage == ORDERED:
            s.reset()
            return "Ótimo! Qual é o próximo livro? Diga um título, autor ou assunto."
        return "Claro! Me diga o título, o autor ou um assunto, ou escolha uma categoria abaixo."

    def _deny(self) -> str:
        s = self.s
        if s.stage in (PRODUCT, QUOTED):
            return self.back()
        if s.stage == BROWSING:
            s.reset()
            return "Sem problemas. Me diga outro título, autor ou assunto."
        return "Tudo bem! Se quiser procurar algo, é só me dizer."

    def _choose_candidate(self, text: str) -> str | None:
        candidates = self.s.candidates
        folded = fold(text)
        for token in words(text):
            if token in ORDINALS:
                index = ORDINALS[token]
                return candidates[index if index < 0 else index - 1]["product_id"] \
                    if abs(index) <= len(candidates) else None
        bare = BARE_NUMBER_RE.match(folded.strip())
        if bare and 1 <= int(bare.group(1)) <= len(candidates):
            return candidates[int(bare.group(1)) - 1]["product_id"]
        for candidate in candidates:
            title = fold(candidate["title"])
            if len(folded.strip()) >= 4 and (folded.strip() in title or title in folded):
                return candidate["product_id"]
        return None

    # ========================================================= operações

    def welcome(self) -> str:
        s = self.s
        if s.stage in (PRODUCT, QUOTED) and s.product:
            return (f"Olá! Seguimos com “{s.product['title']}”? Posso cotar, dar desconto "
                    "ou buscar outro.")
        if s.stage not in (AWAITING_NAME,):
            s.stage = IDLE
        return ("Olá! Sou o livreiro da Cirro. Posso procurar por título, autor ou assunto, "
                "montar uma cotação com desconto e registrar o seu pedido. Por onde começamos?")

    def help(self) -> str:
        return ("Funciona assim: você me diz o que procura (\"um romance brasileiro\", "
                "\"Tolkien\", \"livro sobre hábitos\"), eu mostro as opções, você escolhe a "
                "quantidade, eu monto a cotação — a partir de 3 exemplares tem desconto — e, com "
                "o seu nome, registro o pedido. Também dá para usar os botões abaixo.")

    def search(self, query: str, *, category: str | None = None) -> str:
        s = self.s
        result = self.tools.call("search_products",
                                 {"query": query, "category": category, "max_results": 5})
        if "error" in result:
            return f"Não consegui buscar agora: {result['error']}"
        products = result["products"]
        if not products:
            s.stage, s.candidates = IDLE, []
            return (f"Não encontrei nada para “{query}”. Tente pelo título, pelo autor ou por "
                    "um assunto — ou escolha uma categoria abaixo.")

        s.candidates = [{"product_id": p["product_id"], "title": p["title"]} for p in products]
        s.last_query = query
        if len(products) == 1 or _strong_match(query, products[0]):
            return self.focus(products[0]["product_id"])

        s.stage = BROWSING
        lines = [f"{i}. “{p['title']}”, de {p['author']} — {p['price']}"
                 for i, p in enumerate(products[:5], start=1)]
        return (f"Encontrei {len(products)} opções para “{query}”:\n" + "\n".join(lines)
                + "\nQual delas te interessa?")

    def browse_category(self, category: str) -> str:
        label = next((lbl for lbl, cat in START_SHORTCUTS if cat == category), category)
        s = self.s
        result = self.tools.call("search_products",
                                 {"query": "", "category": category, "max_results": 6})
        if "error" in result:
            return f"Não consegui listar agora: {result['error']}"
        products = result["products"]
        if not products:
            return f"Ainda não temos livros em {label.lower()}."
        s.stage = BROWSING
        s.candidates = [{"product_id": p["product_id"], "title": p["title"]} for p in products]
        lines = [f"{i}. “{p['title']}”, de {p['author']} — {p['price']}"
                 for i, p in enumerate(products, start=1)]
        return f"Alguns títulos de {label.lower()} que temos:\n" + "\n".join(lines) + \
            "\nQuer ver algum deles?"

    def surprise(self) -> str:
        _label, category = random.choice(START_SHORTCUTS)
        result = self.tools.call("search_products",
                                 {"query": "", "category": category, "max_results": 10})
        options = [p for p in result.get("products", []) if p.get("in_stock")]
        if not options:
            return self.welcome()
        pick = random.choice(options)
        self.s.candidates = [{"product_id": p["product_id"], "title": p["title"]} for p in options]
        return self.focus(pick["product_id"], intro="Que tal este? ")

    def focus(self, product_id: str, intro: str = "") -> str:
        s = self.s
        product = self.tools.call("get_product", {"product_id": product_id})
        if "error" in product:
            return f"Não achei esse livro: {product['error']}"
        s.product = {k: product.get(k) for k in (
            "product_id", "title", "author", "category", "tagline", "original_year", "price",
            "price_cents", "stock", "in_stock")}
        s.quote, s.quantity = None, 1
        year = f" ({product['original_year']})" if product.get("original_year") else ""
        if not product.get("in_stock"):
            s.stage = BROWSING if len(s.candidates) > 1 else IDLE
            return (f"{intro}“{product['title']}”, de {product['author']}{year}, está esgotado "
                    "no momento. Quer ver outra opção?")
        s.stage = PRODUCT
        return (f"{intro}“{product['title']}”, de {product['author']}{year}. "
                f"{product.get('tagline', '')} Sai por {product['price']} e temos "
                f"{product['stock']} em estoque. Quer levar? É só dizer quantos exemplares.")

    def details(self) -> str:
        s = self.s
        if not s.product:
            return "De qual livro você quer saber mais?"
        product = self.tools.call("get_product", {"product_id": s.product["product_id"]})
        if "error" in product:
            return f"Não consegui abrir a sinopse: {product['error']}"
        follow = "Posso registrar o pedido?" if s.stage == QUOTED else "Quer levar?"
        return f"{product.get('description') or product.get('tagline')}\n\n{follow}"

    def ask_quantity(self) -> str:
        s = self.s
        if not s.product:
            return self.welcome()
        s.stage = PRODUCT
        return f"Quantos exemplares de “{s.product['title']}” você quer? (até {MAX_QTY})"

    def quote(self, quantity: int, discount_pct: int = 0) -> str:
        s = self.s
        if not s.product:
            return "Primeiro me diga qual livro você quer. Pode ser título, autor ou assunto."
        if quantity < 1:
            return "A quantidade precisa ser de pelo menos 1 exemplar."
        if quantity > MAX_QTY:
            return (f"Por pedido, consigo até {MAX_QTY} exemplares do mesmo livro. "
                    "Quantos você quer?")

        product = s.product
        availability = self.tools.call(
            "check_availability", {"product_id": product["product_id"], "quantity": quantity})
        if "error" in availability:
            return f"Não consegui conferir o estoque: {availability['error']}"
        if not availability.get("available"):
            stock = int(availability.get("stock") or 0)
            s.product = {**product, "stock": stock, "in_stock": stock > 0}
            if stock == 0:
                s.stage = BROWSING if len(s.candidates) > 1 else IDLE
                return f"“{product['title']}” acabou de esgotar. Quer ver outra opção?"
            s.stage, s.quantity = PRODUCT, stock
            return (f"Tenho só {stock} exemplar(es) de “{product['title']}”. "
                    f"Quer levar {stock}?")

        quote = self.tools.call("create_quote", {
            "items": [{"product_id": product["product_id"], "quantity": quantity}],
            "requested_discount_pct": discount_pct,
        })
        if "error" in quote:
            return f"Não consegui montar a cotação: {quote['error']}"

        s.quote, s.quantity, s.stage = quote, quantity, QUOTED
        unit = quote["items"][0]["unit_price"]
        parts = [f"Cotação pronta: {quantity} × “{product['title']}” ({unit}) "
                 f"= {quote['subtotal']}."]
        if quote["discount_pct"]:
            parts.append(f"Com {quote['discount_pct']}% de desconto, fica {quote['total']}.")
        else:
            parts.append(f"Total: {quote['total']}.")
        if quote.get("note"):
            parts.append(quote["note"])
        parts.append("Posso registrar o pedido?")
        return " ".join(parts)

    def request_discount(self, pct: int | None) -> str:
        s = self.s
        if not s.product:
            return self.welcome()
        if not s.quote:
            return self.quote(s.quantity or 1, pct or 10)
        ceiling = int(s.quote.get("max_discount_pct") or 0)
        current = int(s.quote.get("discount_pct") or 0)
        if ceiling == 0:
            return ("Para 1 ou 2 exemplares o preço já é o final. A partir de 3 exemplares dou 5%, "
                    "e a partir de 5 exemplares (ou acima de R$ 300) chego a 10%.")
        if current >= ceiling:
            return (f"Já apliquei o máximo que posso para esta quantidade: {current}%. "
                    "Posso registrar o pedido?")
        return self.quote(s.quantity, pct or ceiling)

    def confirm(self) -> str:
        s = self.s
        if s.stage != QUOTED or not s.quote:
            if s.stage == PRODUCT:
                return self.quote(s.quantity or 1)
            return "Ainda não temos uma cotação. Qual livro você quer?"
        if not s.customer_name:
            return self.ask_name()
        return self.place(s.customer_name)

    def ask_name(self, retry: bool = False) -> str:
        self.s.stage = AWAITING_NAME
        if retry:
            return "Não consegui entender o nome. Pode digitar só o nome e o sobrenome?"
        return "Perfeito! Para registrar, qual nome coloco no pedido?"

    def place(self, name: str | None) -> str:
        s = self.s
        if not name:
            return self.ask_name(retry=True)
        if not s.quote:
            s.stage = IDLE
            return "Não encontrei a cotação. Vamos montar de novo? Qual livro você quer?"
        result = self.tools.call("confirm_order",
                                 {"quote_id": s.quote["quote_id"], "customer_name": name})
        if "error" in result:
            if result.get("status") == 409:  # cotação expirou: refaz com preço atual
                discount = int(s.quote.get("discount_pct") or 0)
                s.customer_name = name
                reply = self.quote(s.quantity, discount)
                return "A cotação anterior expirou e eu refiz com os preços de agora. " + reply
            s.stage = IDLE
            return f"Não consegui registrar o pedido: {result['error']}"

        s.customer_name, s.order, s.stage = name, result, ORDERED
        short = _short_id(result["order_id"])
        if result.get("already_existed"):
            return f"Esse pedido já estava registrado: nº {short}, total {result['total']}."
        return (f"Pedido registrado, {_first_name(name)}! Número {short}, total {result['total']}. "
                "Agora o estoque é confirmado em alguns segundos — acompanhe o status aqui.")

    def order_status(self) -> str:
        s = self.s
        if not s.order:
            return "Ainda não há pedido nesta conversa."
        result = self.tools.call("get_order_status", {"order_id": s.order["order_id"]})
        if "error" in result:
            return f"Não consegui consultar o pedido: {result['error']}"
        s.order = result
        short = _short_id(result["order_id"])
        if result["status"] == "CONFIRMED":
            return (f"Pedido {short} confirmado: estoque reservado e tudo certo. "
                    "Obrigado pela compra!")
        if result["status"] == "REJECTED":
            return (f"Infelizmente o pedido {short} não pôde ser confirmado. "
                    f"{result.get('explanation') or ''} Quer escolher outro livro?")
        return f"O pedido {short} está em confirmação de estoque. Leva só alguns segundos."

    def back(self) -> str:
        s = self.s
        if s.stage == AWAITING_NAME and s.quote:
            s.stage = QUOTED
            return (f"Certo. A cotação de {s.quote['total']} continua de pé. "
                    "Posso ajudar em algo mais?")
        if len(s.candidates) > 1:
            s.stage, s.product, s.quote = BROWSING, None, None
            lines = [f"{i}. “{c['title']}”" for i, c in enumerate(s.candidates[:5], start=1)]
            return "Estas eram as outras opções:\n" + "\n".join(lines) + "\nQual te interessa?"
        s.reset()
        return "Sem problemas. Me diga outro título, autor ou assunto."

    def cancel(self) -> str:
        self.s.reset()
        return "Cancelado, sem problemas. Quer procurar outro livro?"


def _simulate_latency() -> None:
    """No modo mock, imita a latência de um LLM para o k6 ver o mesmo perfil de I/O."""
    if settings().llm_provider != "mock":
        return
    base = settings().mock_latency_ms
    if base <= 0:
        return
    delay_ms = max(0.0, base + random.uniform(-0.25, 0.25) * base)
    time.sleep(delay_ms / 1000)
    metrics.emit("LLMLatency", round(delay_ms, 2), "Milliseconds", dim_provider="mock")
