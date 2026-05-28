# email-rag

Local-first RAG over an IMAP mailbox. Embeddings and vectors stay on your
machine; only the top-k retrieved snippets are sent to Claude to compose an
answer.

## Prerequisites
- Python 3.11+, [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com) running locally: `ollama pull nomic-embed-text`
- `ANTHROPIC_API_KEY` in your environment (for `ask`)

## Setup
```bash
uv sync
uv run email-rag init          # prompts for IMAP host/user/password
uv run email-rag sync          # first run: ~3 years per config 'since'
```

## Use
```bash
uv run email-rag ask "what did Alice commit to about the dataset?"
uv run email-rag search "budget spreadsheet"   # retrieval only, no LLM
uv run email-rag status
```

## Backfill older mail
```bash
uv run email-rag sync --since 2018-01-01
```
