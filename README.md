# ClarityAI

A grounded conversational assistant: multi-turn chat, retrieval over your uploaded documents, optional live web research, source citations, safety layer, and feedback capture.

## To run it

**Read [STARTUP.md](./STARTUP.md).** That's the only doc you need. It's 3 steps.

## What's actually here

```
clarityai/
├── start.bat                  ← one-click Windows launcher
├── STARTUP.md                 ← read this
├── backend/                   ← FastAPI + SQLAlchemy + retrieval + LLM provider
│   ├── .env                   ← put your API key here (LLM_API_KEY=...)
│   ├── app/
│   │   ├── main.py            ← FastAPI app, lifespan, health endpoint
│   │   ├── api/routes/        ← chat, sessions, knowledge, feedback
│   │   ├── services/
│   │   │   ├── chat_engine.py ← orchestrates retrieval → routing → LLM → fallback
│   │   │   ├── retrieval.py   ← TF-IDF + optional dense embeddings
│   │   │   ├── routing.py     ← decides local / research / hybrid per turn
│   │   │   ├── providers.py   ← OpenAI-compatible LLM client (Groq, OpenAI, etc.)
│   │   │   ├── web_research.py← Tavily client (optional)
│   │   │   ├── safety.py      ← prompt-injection + risk patterns
│   │   │   ├── prompts.py     ← system prompt + user prompt builder
│   │   │   ├── seeder.py      ← auto-loads sample_knowledge on first run
│   │   │   ├── chunker.py     ← document chunking
│   │   │   └── documents.py   ← PDF / CSV / JSON / text extraction
│   │   ├── db/                ← SQLAlchemy models (sessions, messages, docs, chunks)
│   │   └── schemas/           ← Pydantic request schemas
│   ├── scripts/               ← dataset export, eval suite (optional)
│   └── tests/                 ← pytest tests for routing, safety, chunking
├── frontend/                  ← React + TypeScript + Vite
│   └── src/
│       ├── App.tsx            ← top-level state, session management
│       ├── lib/api.ts         ← REST client + SSE streaming
│       └── components/        ← Sidebar, ChatView, Composer, MessageBubble, RightPanel
├── sample_knowledge/          ← seed docs auto-loaded on first run
└── docker-compose.yml         ← optional Docker stack (dev)
└── docker-compose.production.yml ← optional Docker stack (prod, with PostgreSQL)
```

## Honest scope

This app is **not** a model trained from scratch on the internet — that's not how anyone builds production AI apps in 2026, regardless of what tutorials suggest. It's a retrieval-augmented chat app on top of a strong existing LLM (Groq/OpenAI/Anthropic via API). That's the architecture every "AI app" you've used is actually running.

What that gives you:
- Real LLM intelligence for reasoning and tone
- Your own documents as the knowledge base (this is what makes it *your* app)
- Live web sources via Tavily when the LLM needs current info
- Citations on every claim, so users can verify
- Per-turn routing between local-only, web-research, or hybrid

What it doesn't promise:
- 100% accurate answers (no honest LLM app does)
- Perfect domain knowledge without you uploading documents
- Original training of foundation models (a $10M+ effort outside the scope of any single app)

## Optional: Docker

If you have Docker Desktop installed and don't want to install Python/Node locally:

```bash
docker compose up --build
```

Open http://localhost:5173.

For a production-flavored stack with PostgreSQL and a built frontend:

```bash
docker compose -f docker-compose.production.yml up --build
```

Open http://localhost:8080.
