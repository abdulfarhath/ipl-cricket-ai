"""FastAPI gateway (spec §23, §14, §25).

- POST /api/chat      — agent answer (Redis-cached, request IDs, latency + trace)
- GET  /api/players/{name}/stats, /api/seasons/{year}, /api/records, /api/health
- GET  /api/metrics   — latency percentiles, cache hit rate
- GET  /             — chat frontend

Run: uvicorn api.main:app --host 0.0.0.0 --port 8000
"""
import hashlib
import json
import logging
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from agent.graph import answer as agent_answer
from core.config import CACHE_TTL_S, REDIS_URL
from stats import engine

logging.basicConfig(level=logging.INFO,
                    format='{"lvl":"%(levelname)s","logger":"%(name)s","msg":%(message)r}')
log = logging.getLogger("api")

app = FastAPI(title="IPL Intelligence Agent", version="1.0.0")
_lat: list[int] = []
_cache_stats = {"hits": 0, "misses": 0}


def _redis():
    import redis
    return redis.from_url(REDIS_URL, decode_responses=True, socket_timeout=2)


class ChatRequest(BaseModel):
    question: str = Field(min_length=2, max_length=500)
    history: list[dict] = []


@app.post("/api/chat")
def chat(req: ChatRequest):
    rid = uuid.uuid4().hex[:8]
    t0 = time.perf_counter()
    key = "ipl:chat:" + hashlib.sha1(
        req.question.strip().lower().encode()).hexdigest()

    cached = None
    if not req.history:  # only cache stateless questions
        try:
            cached = _redis().get(key)
        except Exception:  # noqa: BLE001 — cache is best-effort, never fatal
            log.warning("redis unavailable, skipping cache")
    if cached:
        _cache_stats["hits"] += 1
        ms = round((time.perf_counter() - t0) * 1000)
        _lat.append(ms)
        log.info("rid=%s cache=hit ms=%s", rid, ms)
        return {**json.loads(cached), "cached": True, "latency_ms": ms,
                "request_id": rid}

    _cache_stats["misses"] += 1
    result = agent_answer(req.question, req.history)
    if not req.history:
        try:
            _redis().setex(key, CACHE_TTL_S, json.dumps(
                {"answer": result["answer"], "intent": result["intent"]}))
        except Exception:  # noqa: BLE001
            pass
    _lat.append(result["latency_ms"])
    del _lat[:-1000]
    log.info("rid=%s cache=miss intent=%s ms=%s tools=%s", rid,
             result["intent"], result["latency_ms"],
             [t["tool"] for t in result["trace"]])
    return {**result, "cached": False, "request_id": rid}


@app.get("/api/players/{name}/stats")
def player_stats(name: str):
    result = {"batting": engine.batting_stats(name),
              "bowling": engine.bowling_stats(name),
              "fielding": engine.fielding_stats(name)}
    if all("error" in v for v in result.values()):
        raise HTTPException(404, f"no data for player '{name}'")
    return result


@app.get("/api/seasons/{year}")
def season(year: int):
    result = engine.season_summary(year)
    if "error" in result:
        raise HTTPException(404, result["error"])
    return result


@app.get("/api/records")
def records():
    return engine.records()


@app.get("/api/health")
def health():
    from core.db import q1
    try:
        db_ok = q1("SELECT 1 AS ok")["ok"] == 1
    except Exception:  # noqa: BLE001
        db_ok = False
    try:
        redis_ok = _redis().ping()
    except Exception:  # noqa: BLE001
        redis_ok = False
    status = "ok" if db_ok else "degraded"
    return {"status": status, "postgres": db_ok, "redis": redis_ok}


@app.get("/api/metrics")
def metrics():
    s = sorted(_lat)
    total = _cache_stats["hits"] + _cache_stats["misses"]
    return {
        "requests": len(s),
        "latency_ms_p50": s[len(s) // 2] if s else None,
        "latency_ms_p95": s[int(len(s) * 0.95) - 1] if len(s) >= 2 else (s[-1] if s else None),
        "latency_ms_max": s[-1] if s else None,
        "cache_hit_rate": round(_cache_stats["hits"] / total, 3) if total else None,
        **_cache_stats,
    }


@app.get("/", response_class=HTMLResponse)
def frontend():
    return (Path(__file__).resolve().parent.parent / "frontend" / "index.html"
            ).read_text(encoding="utf-8")
