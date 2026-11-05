# projeto-cloud-chatbot

O **livreiro** da Cirro Livros: o vendedor conversacional do projeto semestral
de Computação em Nuvem (Insper). Ele conversa em português, consulta o
catálogo, monta a cotação com desconto e registra o pedido, tudo por **funções
reais** do backend.

| Repositório | Serviço ECS | Responsabilidade |
|---|---|---|
| [projeto-cloud-backend](https://github.com/PedroMRegis/projeto-cloud-backend) | `backend-api`, `order-worker` | Catálogo, cotação, pedido, confirmação assíncrona |
| **projeto-cloud-chatbot** (este) | `chatbot` | Vendedor conversacional |
| [projeto-cloud-frontend](https://github.com/PedroMRegis/projeto-cloud-frontend) | `frontend` | Loja em React |
| [projeto-cloud-insper](https://github.com/joselsm3/projeto-cloud-insper) | — | Plano, infraestrutura, docs e compose da plataforma |

## O que mudou em relação à versão anterior

O vendedor anterior não guardava contexto. Ele sugeria um livro, perguntava "quer
levar?", o cliente respondia **"sim"** e o "sim" virava uma busca por "sim".

Agora cada conversa tem um **estado de sessão** persistido no DynamoDB:

```
idle ──busca──► browsing ──escolhe──► product ──quantidade/"sim"──► quoted
  ▲                                     │                            │ "sim"
  │                                     ◄──────── "não"/outras ──────┤
  │                                                                  ▼
  └──────────── comprar outro ◄────── ordered ◄──── nome ──── awaiting_name
```

O mesmo "sim" significa "quero esse livro" na etapa `product` e "pode fechar" na
etapa `quoted`. Cada etapa devolve também **sugestões clicáveis** (com uma
`action` estruturada), e o cliente pode comprar só clicando. Texto livre continua
funcionando.

## Três modos de atender

| Entrada | Quem responde | Observação |
|---|---|---|
| Botão (`action`) | Vendedor guiado (`seller.py`) | Sem ambiguidade. Funciona até com o LLM fora do ar |
| Texto, `LLM_PROVIDER=mock` | Vendedor guiado, interpretando o texto pela etapa | Sem custo de LLM; usado em teste de carga |
| Texto, `LLM_PROVIDER=bedrock`/`gemini`/`groq` | LLM com tool calling (`agent.py`) | Recebe histórico **e** um resumo do estado (livro em foco, `quote_id`, nome) |

As tools (`search_products`, `get_product`, `check_availability`,
`create_quote`, `confirm_order`, `get_order_status`) são chamadas HTTP ao
backend. **O LLM nunca define preço**: só pede desconto, e o backend aplica o
teto.

## Resiliência

- **LLM:** timeout de 20 s, retry com jitter, circuit breaker por provedor e fallback Bedrock → Gemini → Groq (Gemini e Groq só com chave configurada). Se tudo falhar, a resposta é degradada (HTTP 200) e os botões continuam vendendo.
- **Backend:** timeout de 5 s, retry em erro de conexão e 502/503/504, circuit breaker (5 falhas em 30 s abrem o circuito por 15 s). Com o backend fora, o turno degrada com uma explicação e o `request_id` é liberado para o cliente tentar de novo.
- **Idempotência:** o `request_id` repetido devolve a resposta guardada, sem chamar o LLM de novo e sem segundo pedido.
- **Rate limit:** 20 mensagens/min por conversa, para proteger a cota do LLM.
- **Contexto distribuído:** `X-Request-Id`, `traceparent` e `X-Conversation-Id` seguem para o backend, então os logs dos dois serviços se correlacionam.

## Dados (DynamoDB `conversations`)

Uma partição por conversa (`conversation_id`), com itens pela sort key:

| `sk` | Conteúdo |
|---|---|
| `MSG#<ts>#<n>` | Uma mensagem. A do vendedor guarda também o `ui` (cards, cotação, sugestões) para o frontend restaurar a conversa |
| `STATE` | Estado da sessão de venda |
| `REQ#<request_id>` | Resposta guardada (idempotência) |
| `RATE#<minuto>` | Contador do rate limit |

Tudo tem TTL de 7 dias.

## API

| Rota | O que faz |
|---|---|
| `POST /api/chat` | `{message, request_id, conversation_id?, action?}` → resposta + `products`, `quote`, `order`, `suggestions`, `input_hint`, `stage` |
| `GET /api/conversations/{id}` | Histórico com o `ui` de cada resposta, além do estado e das sugestões atuais |
| `POST /api/conversations/{id}/start` | Sugestões iniciais, sem gastar um turno |
| `GET /health` / `GET /ready` | Vivo / dependências (DynamoDB e backend) |

## Rodando

```bash
docker compose up -d --build     # DynamoDB Local + chatbot (precisa do backend em :8001)
```

Para usar um LLM real, copie `.env.example` para `.env` e preencha as chaves. Para a
plataforma inteira, use o compose do `projeto-cloud-insper`.

### Bedrock

Com `LLM_PROVIDER=bedrock`, o vendedor usa o Claude Haiku 5.5 pela API Converse do
`bedrock-runtime`. O modelo só aceita perfil de inferência, e o padrão é o global,
que funciona tanto em Ohio quanto em São Paulo.

| Variável | Padrão | Para quê |
|---|---|---|
| `BEDROCK_MODEL` | `global.anthropic.claude-haiku-5-5` | Perfil de inferência |
| `BEDROCK_REGION` | vazio (usa `AWS_REGION`) | Região do Bedrock, se for diferente da do serviço |
| `BEDROCK_EFFORT` | `medium` | `low` a `max`: quanto o modelo raciocina |
| `BEDROCK_MAX_TOKENS` | `16000` | Teto da resposta, incluindo o raciocínio |

Na AWS não há chave: a autenticação é a task role do ECS, que precisa de
`bedrock:InvokeModel` no perfil de inferência e no modelo em todas as regiões do
perfil. O compose local usa credenciais AWS falsas para o DynamoDB Local, então
continua em `mock`. Para testar o Bedrock em local, rode o `uvicorn` fora do compose,
com `AWS_PROFILE` definido.

### Testes

```bash
python -m venv .venv && .venv/Scripts/pip install -r requirements-dev.txt   # Windows
.venv/Scripts/python -m pytest      # 66 testes, sem AWS e sem Docker (moto + backend falso)
.venv/Scripts/python -m ruff check .
```

`tests/test_conversation.py` começa reproduzindo o bug do "sim".
