# IPL Cricket AI — two builds

AI-powered, text-to-text IPL assistant (data: 2008 → 31 Dec 2025). LLM never
invents statistics — every number comes from a database built from Cricsheet
ball-by-ball data.

| Folder | What | Stack |
|---|---|---|
| `ipl-intelligence/` | **v2 — production build (recommended)** | PostgreSQL+pgvector, Redis, intent router, agent, MCP server, RAG, FastAPI, chat frontend, Docker Compose |
| `ipl-ai-assistant/` | v1 — single-file build | SQLite, Qdrant, BGE-M3, Gemini/Claude agent, FastMCP, FastAPI |

## Quick start (v2)

```bash
cd ipl-intelligence
cp .env.example .env     # put your GEMINI_API_KEY (free: aistudio.google.com)
docker compose up --build
# open http://localhost:8000
```

Each folder's README has full architecture, verification results, and
deployment docs. `ipl-ai-assistant/INTERVIEW_GUIDE.md` explains the whole
system for interviews.
