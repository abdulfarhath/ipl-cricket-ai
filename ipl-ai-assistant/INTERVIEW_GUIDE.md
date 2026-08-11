# Interview Guide — IPL AI Cricket Assistant

How to explain this project to an HR screen, a team lead, or a technical interviewer.
Read top to bottom once; before an interview, re-read sections 1, 7, and 8.

---

## 1. The 30-second pitch (memorize this)

> "I built a text-to-text AI assistant for IPL cricket covering 2008 to end of 2025.
> The core design idea is that the LLM never generates statistics — it only decides
> which tool to call. All numbers come from a SQLite database built from ball-by-ball
> Cricsheet data, and a vector database handles meaning-based search. The agent can
> consume its tools directly or over MCP, it runs on Gemini or Claude behind one
> interface, and the whole thing deploys with one Docker Compose command. I also
> built a golden-QA evaluation harness — it currently scores 6/6 on grounded
> correctness with ~2-second warm latency."

Every sentence there is a hook the interviewer can pull on. Know the story behind each.

## 2. Buzzwords — one line each (HR-safe definitions)

| Term | Plain meaning | Where it is in this project |
|---|---|---|
| LLM | The language model that reads the question and writes the answer | Gemini Flash (free) or Claude Opus 5, switchable by env var |
| Agent | An LLM in a loop that decides actions (tool calls) until it can answer | `answer()` in app.py |
| Tool calling | The LLM outputs "call function X with args Y" instead of text; code runs it and feeds the result back | 13 functions in `TOOL_FUNCS` |
| MCP | Model Context Protocol — a standard plug so ANY AI app can use your tools | `python app.py mcp`; agent can also consume tools through it |
| RAG | Retrieval-Augmented Generation — fetch relevant documents, let the LLM answer from them | Qdrant + BGE-M3 over 1,169 match summaries |
| Embeddings | Turning text into number vectors so similar meanings are near each other | BGE-M3, 1024 dimensions |
| Vector DB | A database that searches by meaning-distance instead of keywords | Qdrant |
| ETL | Extract-Transform-Load: raw files → clean database | Cricsheet JSON → SQLite |
| Docker Compose | Config file that starts the whole system (4 services) with one command | docker-compose.yml |

## 3. The request flow — tell it as a story

"A user asks: *Who took the Purple Cap in 2025?*

1. FastAPI receives it at POST /chat.
2. The agent sends the question + a list of 13 tool schemas to the LLM.
3. The LLM doesn't know the answer — and my system prompt forbids guessing — so it
   replies with a tool call: `top_wicket_takers(season=2025)`.
4. My code executes that: a SQL query over an aggregation view built on 278,205
   ball-by-ball rows. Result: M Prasidh Krishna, 25 wickets.
5. The result goes back to the LLM, which writes the final sentence around it.
6. The API returns the answer plus `latency_ms`.

If the question is qualitative — *'show me last-ball thrillers'* — the LLM picks the
semantic tool instead: the query is embedded with BGE-M3 and Qdrant returns the
nearest match summaries. Exact numbers = SQL; vibes = vectors. Never the other way."

## 4. Architecture decisions and WHY (the senior-engineer answers)

| Decision | Why (say this) |
|---|---|
| SQL for stats, not RAG | Embeddings are lossy and can't aggregate. Asking a vector DB "how many runs" is using a similarity engine for arithmetic. RAG only for meaning-search. `ENABLE_RAG=0` proves the system stands without it. |
| SQLite, not Postgres | 278K read-only rows after ETL. SQLite = zero infra, milliseconds per query. Tool layer is plain SQL, so Postgres later is a connection-string change. Don't build for scale you don't have. |
| Qdrant | Rust-fast, best production default in 2026 comparisons, embedded local mode for dev and a Docker image for deploy — same client API for both. |
| BGE-M3 embeddings | Top open-source retrieval model, MIT license, runs on CPU. Qwen3-8B scores higher on leaderboards but needs ~5 GB for zero practical gain on 1,169 short docs. |
| No LangChain | One agent, 13 tools. The provider SDK runs the loop in ~30 lines. A framework earns its place at multi-agent graphs, not before. Fewer dependencies = fewer 3 AM surprises. |
| Two LLM providers | Business reality: Gemini has a free tier, Claude has the best tool use. One `answer()` interface, provider picked by env var. Swapping LLMs is a config change, not a rewrite. |
| MCP | Decouples tools from the brain. Claude Desktop or any MCP client can query my cricket DB without touching my API. The agent itself can also run as an MCP client — same tools, over the protocol. |
| Views, not stored aggregates | `batting_by_season` etc. computed live — can't go stale, and 278K rows aggregate in milliseconds. Materialize only when profiling says so. |

## 5. Where everything is stored

Local (no Docker):

```
ipl-ai-assistant/
├── app.py            code — the whole system
├── .env              secrets (API key) — never committed
└── data/
    ├── raw/          downloaded Cricsheet zip + 1,243 JSON files
    ├── db/ipl.db     SQLite — matches, deliveries, players, bios + views
    └── qdrant/       embedded vector store (local mode)
```

Docker: named volumes, survive container restarts/rebuilds:

| Volume | Holds |
|---|---|
| `app_data` | the `data/` tree above (SQLite + raw) |
| `qdrant_data` | Qdrant collections |
| `hf_cache` | the 2 GB BGE-M3 model download (so rebuilds don't re-download) |

Images are built from the `Dockerfile`; code is COPY'd in at build time — that's why
code changes need `docker compose up --build`.

## 6. Deployment — the exact process

```
docker compose up --build
```

What actually happens, in order:

1. **build** — Docker creates the app image: python:3.12-slim + CPU torch + requirements + app.py.
2. **qdrant** starts (vector DB service, port 6333).
3. **bootstrap** runs once: `python app.py all` → downloads Cricsheet data → builds
   SQLite → embeds 1,169 docs → loads Qdrant → exits 0. Compose gates on
   `service_completed_successfully`.
4. **mcp** starts: tool server over streamable-http on :8765.
5. **api** starts: FastAPI on :8000, agent configured with `MCP_URL=http://mcp:8765/mcp`
   so every tool call travels through the MCP service.

Services talk over Compose's internal network by service name (`http://qdrant:6333`,
`http://mcp:8765`). Only 8000 (and 8765/6333 if wanted) are published to the host.

### Access from OTHER PCs/laptops on the same Wi-Fi/LAN

1. On the machine running Docker, find its LAN IP: `ipconfig` (Windows) → IPv4 Address,
   e.g. `192.168.1.42` (`ip addr` on Linux).
2. Allow the port through the firewall (Windows: Defender Firewall → Advanced →
   Inbound Rules → New Rule → Port 8000 → Allow).
3. Any device on the same network:

```
curl -X POST http://192.168.1.42:8000/chat -H "content-type: application/json" -d "{\"question\":\"Who won IPL 2019?\"}"
```

Phone browser test: `http://192.168.1.42:8000/health`.

Internet-wide access = different league: rent a small VPS (or a tunnel like
ngrok/Cloudflare Tunnel for demos), same compose file, plus reverse proxy with HTTPS
and an auth token — say this in interviews as "what I'd add for production."

## 7. Correctness + latency — the measurable story

### How do you KNOW answers are correct?

Three layers (interviewers love layered answers):

1. **Grounding by construction** — the system prompt forbids stating numbers without
   a tool result; the tools read a database built from official ball-by-ball data.
2. **Golden-QA eval harness** — `python app.py eval` runs known-answer questions and
   string-checks the response. Current score: **6/6**. Re-run after every change =
   regression safety net.
3. **Spot audits** — season leaders cross-checked against public records (RCB's 2025
   title, Prasidh Krishna's 25 wickets, Kohli's 973-run 2016).

Honest caveat to volunteer: the LLM can still phrase things misleadingly around
correct numbers; the eval catches wrong facts, not wrong tone. Next step would be an
answer-validator that extracts every number and asserts it appeared in tool output.

### Latency: measured, explained, reduced

Measured (free Gemini tier, in-process tools):

```
warm answers:      1.6 – 2.7 s
cold/rate-limited: up to ~70 s   ← this is WAITING on free-tier limits, not compute
avg over eval set: 16.6 s (dragged by one rate-limited outlier)
```

Where time goes per answer: 1–3 LLM round-trips (~0.5–1.5 s each on Flash-Lite) +
SQL under 10 ms + occasional embedder load (~10 s one-time if a semantic question
arrives first).

Levers to reduce (in impact order):

1. **Paid tier / higher quota** — kills the 20–60 s retry waits. Biggest lever by far.
2. **Fewer LLM rounds** — the name-resolution hop (search_player → stats tool) costs a
   full round-trip; caching resolved names or accepting fuzzy names in tools removes it.
3. **In-process tools instead of MCP hop** — unset MCP_URL; saves network overhead per call.
4. **`ENABLE_RAG=0`** if semantic search unused — avoids the 2 GB model load entirely.
5. **Answer caching** — sports questions repeat heavily; a dict/Redis cache on
   normalized questions gives sub-100 ms repeats.

Observability built in: every `/chat` response carries `latency_ms`; `GET /metrics`
returns count/avg/p50/p95/max over the last 500 requests.

## 8. Likely interview questions — prepared answers

**Q: Why didn't you just fine-tune a model on cricket data?**
Fine-tuning teaches style, not reliable facts — it still hallucinates numbers and
costs GPU money every data update. Tool-grounding gives exact answers and updating
data is just re-running ETL.

**Q: What breaks if I ask about IPL 2026?**
System prompt scopes to 2008–2025; ETL drops post-cutoff matches (74 were skipped).
Agent politely refuses. Guardrail at both data layer and prompt layer.

**Q: What was the hardest bug?**
`from __future__ import annotations` turned all type hints into strings; the Gemini
SDK crashed on every tool call, and the model silently answered from memory instead —
confidently wrong stats. Fixed the import, added the eval harness so silent tool
failure can never masquerade as success again. (Great story — tells them you debug
at the integration layer and add regression protection.)

**Q: How do you handle the LLM being down or rate-limited?**
Retry with exponential backoff on 429/503; two providers behind one interface, so
flipping an env var switches Gemini↔Claude. Claude path also has server-side fallback.

**Q: SQL injection? The LLM writes SQL!**
`run_sql` is read-only by construction: comments stripped, single statement enforced,
must start SELECT/WITH, write/DDL keywords blocked, row-capped. DB file can also be
mounted read-only in Docker. Defense in depth.

**Q: How would you scale this to 10,000 users?**
Cache answers (questions repeat), move SQLite→Postgres + pgvector or keep Qdrant,
run several API replicas behind a load balancer — the API is stateless so it scales
horizontally — and put the LLM on a paid tier with rate budgeting per user.

**Q: Why is one file 1,000 lines? Is that good engineering?**
Deliberate for a v1: one file, six labeled sections, zero import spaghetti. The tool
registry pattern means splitting into a package later is mechanical. Ship first,
modularize when tests demand it.

**Q: What would you build next?**
Answer-validator (every number must appear in tool output), conversation memory in
Redis, live-season ingestion behind a feature flag, LLM-as-judge eval for phrasing
quality, tracing with OpenTelemetry.

## 9. War stories (drop one when asked "tell me about a challenge")

1. **The silent hallucination** — annotations bug above. Lesson: integration failures
   can look like model failures; always trace the actual tool calls.
2. **Tool-name mismatch** — prompt said `run_sql`, registry exposed `tool_run_sql`;
   model called the prompt's name → KeyError. Fixed with a single shared registry
   consumed by Claude, Gemini AND MCP. Lesson: one source of truth for contracts.
3. **Model deprecation mid-project** — `gemini-2.5-flash` closed to new accounts the
   week I integrated it. Fixed with `-latest` aliases + model as config. Lesson:
   never hardcode model IDs.
4. **SDK major-version break** — mcp 2.0 renamed FastMCP and changed client returns;
   google-genai couldn't digest 2.0 sessions. Pinned `<2`, wrote a version-proof
   manual MCP bridge. Lesson: pin dependencies, read changelogs, own your protocol
   layer when SDKs wobble.

## 10. Glossary of numbers to remember

| Number | What |
|---|---|
| 1,169 | matches in DB (2008 → 2025) |
| 278,205 | ball-by-ball delivery rows |
| 767 | players |
| 33 | curated bios |
| 13 | agent/MCP tools |
| 1,024 | embedding dimensions (BGE-M3) |
| 6/6 | golden-QA accuracy |
| ~2 s | warm answer latency |
| 4 | Docker services (qdrant, bootstrap, mcp, api) |
