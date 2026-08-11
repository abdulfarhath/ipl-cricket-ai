"""Agent orchestration (spec §2, §13, §14).

Supervisor pattern as a small deterministic graph:

    question -> route() -> STATS  (tool loop over the statistics engine)
                        -> KNOWLEDGE (RAG retrieve -> LLM)
                        -> OUT_OF_SCOPE / CURRENT_DATA (canned, zero LLM calls)

Hand-rolled rather than LangGraph: same supervisor/worker pattern, zero extra
dependencies, lower latency (documented deviation — see README §deviations).
"""
import json
import logging
import sys
import time
from enum import Enum
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import DATA_CUTOFF
from core.llm import LLMProvider, ToolSpec, get_provider
from stats import engine

log = logging.getLogger("agent")


class Intent(str, Enum):
    STATS = "stats"
    KNOWLEDGE = "knowledge"
    CURRENT_DATA = "current_data"
    OUT_OF_SCOPE = "out_of_scope"


SYSTEM_PROMPT = f"""You are an IPL-only AI assistant covering the Indian Premier
League from 2008 through {DATA_CUTOFF}. Rules:
1. NEVER state a statistic without fetching it from a tool. Numbers come from
   the database; you explain and format them.
2. Player names are stored in Cricsheet initial form ("V Kohli"). Resolve full
   names with search_player first, then use the exact returned name.
3. Cite season/opponent/venue context and mention the source when giving stats.
4. For comparisons, present a clean table.
5. If a tool reports data unavailable (e.g. coaches), say so honestly.
6. Refuse anything not about IPL, and anything after {DATA_CUTOFF}."""

OUT_OF_SCOPE_REPLY = ("I specialize in IPL cricket (2008 through Dec 2025) — "
                      "player stats, teams, matches, records, and IPL history. "
                      "Please ask me an IPL-related question!")
CURRENT_DATA_REPLY = (f"My data ends on {DATA_CUTOFF}, so I can't answer about "
                      "live or current events. Ask me anything about IPL 2008-2025!")

_OOS_MARKERS = ("football", "soccer", "nba", "quantum", "python", "prime minister",
                "weather", "stock", "bitcoin", "movie", "recipe", "translate")
_CURRENT_MARKERS = ("2026", "today", "tomorrow", "live score", "next match",
                    "upcoming", "current squad", "latest news")
_KNOWLEDGE_MARKERS = ("why", "history", "founded", "started", "rule", "explain",
                      "impact", "how did", "what is the ipl", "when did")


def route(question: str) -> Intent:
    """Deterministic keyword router (spec §14: fast path, no LLM call).
    Anything ambiguous falls through to STATS, whose tool loop self-corrects."""
    ql = question.lower()
    if any(m in ql for m in _OOS_MARKERS):
        return Intent.OUT_OF_SCOPE
    if any(m in ql for m in _CURRENT_MARKERS):
        return Intent.CURRENT_DATA
    if any(m in ql for m in _KNOWLEDGE_MARKERS) and not any(
            s in ql for s in ("runs", "wickets", "average", "score", "record")):
        return Intent.KNOWLEDGE
    return Intent.STATS


# ---- tool registry for the stats path -------------------------------------
def _spec(name, desc, props, required):
    return ToolSpec(name, desc, {"type": "object", "properties": props,
                                 "required": required})


TOOLS: dict[str, tuple] = {
    "search_player": (engine.search_player, _spec(
        "search_player", "Resolve canonical player name from fragment/surname.",
        {"fragment": {"type": "string"}}, ["fragment"])),
    "get_player_batting_stats": (engine.batting_stats, _spec(
        "get_player_batting_stats", "Career+season batting stats for exact name.",
        {"player": {"type": "string"}, "season": {"type": "integer"}}, ["player"])),
    "get_player_bowling_stats": (engine.bowling_stats, _spec(
        "get_player_bowling_stats", "Career+season bowling stats for exact name.",
        {"player": {"type": "string"}, "season": {"type": "integer"}}, ["player"])),
    "get_player_fielding_stats": (engine.fielding_stats, _spec(
        "get_player_fielding_stats", "Catches/stumpings/run outs as fielder.",
        {"player": {"type": "string"}}, ["player"])),
    "get_player_highest_score": (engine.highest_score, _spec(
        "get_player_highest_score",
        "Top innings with opponent, venue, season, balls, SR.",
        {"player": {"type": "string"}}, ["player"])),
    "get_player_phase_stats": (engine.phase_stats, _spec(
        "get_player_phase_stats", "Powerplay/middle/death splits.",
        {"player": {"type": "string"},
         "discipline": {"type": "string", "enum": ["batting", "bowling"]}},
        ["player"])),
    "get_player_vs_team": (engine.player_vs_team, _spec(
        "get_player_vs_team", "Player's record against one team.",
        {"player": {"type": "string"}, "team": {"type": "string"}},
        ["player", "team"])),
    "get_batter_vs_bowler": (engine.batter_vs_bowler, _spec(
        "get_batter_vs_bowler", "Ball-by-ball duel between batter and bowler.",
        {"batter": {"type": "string"}, "bowler": {"type": "string"}},
        ["batter", "bowler"])),
    "get_head_to_head": (engine.head_to_head, _spec(
        "get_head_to_head", "All-time wins between two teams.",
        {"team_a": {"type": "string"}, "team_b": {"type": "string"}},
        ["team_a", "team_b"])),
    "get_team_season_stats": (engine.team_season, _spec(
        "get_team_season_stats", "Team wins/losses in one season.",
        {"team": {"type": "string"}, "season": {"type": "integer"}},
        ["team", "season"])),
    "get_season_stats": (engine.season_summary, _spec(
        "get_season_stats", "Season final, champion, orange/purple cap.",
        {"season": {"type": "integer"}}, ["season"])),
    "get_venue_stats": (engine.venue_stats, _spec(
        "get_venue_stats", "Matches and average score at a venue.",
        {"venue": {"type": "string"}}, ["venue"])),
    "get_ipl_records": (engine.records, _spec(
        "get_ipl_records", "All-time IPL records, computed live.", {}, [])),
    "get_team_staff": (engine.team_staff, _spec(
        "get_team_staff", "Coaches/staff (reports unavailability honestly).",
        {"team": {"type": "string"}, "season": {"type": "integer"}}, ["team"])),
}


def _stats_agent(provider: LLMProvider, question: str, history: list[dict],
                 trace: list) -> str:
    specs = [spec for _, spec in TOOLS.values()]
    messages = list(history) + [{"role": "user", "content": question}]
    for _ in range(8):
        resp = provider.chat(SYSTEM_PROMPT, messages, tools=specs)
        if not resp.tool_calls:
            return resp.text
        messages.append({"role": "assistant", "content": resp.text,
                         "tool_calls": resp.tool_calls, "raw": resp.raw})
        for tc in resp.tool_calls:
            t0 = time.perf_counter()
            fn = TOOLS.get(tc.name, (None,))[0]
            result = json.dumps(fn(**tc.args)) if fn else \
                json.dumps({"error": f"unknown tool {tc.name}"})
            trace.append({"tool": tc.name, "args": tc.args,
                          "ms": round((time.perf_counter() - t0) * 1000)})
            messages.append({"role": "tool", "name": tc.name, "content": result})
    return "Sorry — tool-call limit reached without a final answer."


def _knowledge_agent(provider: LLMProvider, question: str, trace: list) -> str:
    from rag.store import search
    t0 = time.perf_counter()
    docs = search(question, top_k=4)
    trace.append({"tool": "search_ipl_knowledge", "args": {"query": question},
                  "ms": round((time.perf_counter() - t0) * 1000)})
    context = "\n\n".join(f"[{d['title']} — {d['source']}]\n{d['content']}"
                          for d in docs) or "no documents found"
    resp = provider.chat(
        SYSTEM_PROMPT,
        [{"role": "user", "content":
          f"Answer using ONLY these retrieved documents. Cite the document "
          f"titles you used.\n\nDOCUMENTS:\n{context}\n\nQUESTION: {question}"}])
    return resp.text


def answer(question: str, history: list[dict] | None = None) -> dict:
    """Entry point. Returns {answer, intent, trace, latency_ms}."""
    t0 = time.perf_counter()
    intent = route(question)
    trace: list = []
    if intent == Intent.OUT_OF_SCOPE:
        text = OUT_OF_SCOPE_REPLY
    elif intent == Intent.CURRENT_DATA:
        text = CURRENT_DATA_REPLY
    else:
        provider = get_provider()
        if intent == Intent.KNOWLEDGE:
            text = _knowledge_agent(provider, question, trace)
        else:
            text = _stats_agent(provider, question, history or [], trace)
    return {"answer": text, "intent": intent.value, "trace": trace,
            "latency_ms": round((time.perf_counter() - t0) * 1000)}
