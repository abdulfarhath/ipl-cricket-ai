"""IPL MCP server (spec §12). Typed tools over the statistics engine + RAG.
No arbitrary SQL is exposed. Run: python -m mcp_server.server
(stdio default; MCP_TRANSPORT=streamable-http for network use).
"""
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import MCP_PORT
from stats import engine

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("mcp")


def _wrap(fn, *args, **kwargs) -> str:
    """Uniform error handling + logging for every tool."""
    try:
        result = fn(*args, **kwargs)
    except Exception as e:  # noqa: BLE001 — tool boundary must not crash the server
        log.exception("tool %s failed", fn.__name__)
        result = {"error": f"internal error in {fn.__name__}: {e}"}
    log.info("tool=%s args=%s", fn.__name__, args or kwargs)
    return json.dumps(result)


def build_server():
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("ipl-intelligence", host="0.0.0.0", port=MCP_PORT)

    @mcp.tool()
    def search_player(name_fragment: str) -> str:
        """Resolve a player's canonical name from a partial name or surname."""
        return _wrap(engine.search_player, name_fragment)

    @mcp.tool()
    def get_player_batting_stats(player: str, season: int | None = None) -> str:
        """Career + per-season batting stats (runs, average, SR, 50s/100s, highest)."""
        return _wrap(engine.batting_stats, player, season)

    @mcp.tool()
    def get_player_bowling_stats(player: str, season: int | None = None) -> str:
        """Career + per-season bowling stats (wickets, economy, best figures)."""
        return _wrap(engine.bowling_stats, player, season)

    @mcp.tool()
    def get_player_fielding_stats(player: str) -> str:
        """Catches, stumpings, run outs credited as fielder."""
        return _wrap(engine.fielding_stats, player)

    @mcp.tool()
    def get_player_highest_score(player: str) -> str:
        """Top 5 innings with opponent, venue, season, balls, strike rate."""
        return _wrap(engine.highest_score, player)

    @mcp.tool()
    def get_player_phase_stats(player: str, discipline: str = "batting") -> str:
        """Powerplay / middle / death splits. discipline: batting|bowling."""
        return _wrap(engine.phase_stats, player, discipline)

    @mcp.tool()
    def get_player_vs_team(player: str, team: str) -> str:
        """A player's batting and bowling record against one team."""
        return _wrap(engine.player_vs_team, player, team)

    @mcp.tool()
    def get_batter_vs_bowler(batter: str, bowler: str) -> str:
        """Ball-by-ball duel: balls, runs, dismissals, strike rate."""
        return _wrap(engine.batter_vs_bowler, batter, bowler)

    @mcp.tool()
    def get_head_to_head(team_a: str, team_b: str) -> str:
        """All-time win counts between two teams."""
        return _wrap(engine.head_to_head, team_a, team_b)

    @mcp.tool()
    def get_team_season_stats(team: str, season: int) -> str:
        """One team's wins/losses in one season."""
        return _wrap(engine.team_season, team, season)

    @mcp.tool()
    def get_season_stats(season: int) -> str:
        """Season overview: final, champion, most runs, most wickets."""
        return _wrap(engine.season_summary, season)

    @mcp.tool()
    def get_venue_stats(venue: str) -> str:
        """Matches hosted and average innings score at a venue."""
        return _wrap(engine.venue_stats, venue)

    @mcp.tool()
    def get_ipl_records() -> str:
        """All-time records computed live (most runs/wickets, highest scores...)."""
        return _wrap(engine.records)

    @mcp.tool()
    def get_match_scorecard(match_id: str) -> str:
        """Result + top batting/bowling performances for one match id."""
        return _wrap(engine.match_scorecard, match_id)

    @mcp.tool()
    def get_team_staff(team: str, season: int | None = None) -> str:
        """Coaches/support staff. Honestly reports data unavailability."""
        return _wrap(engine.team_staff, team, season)

    @mcp.tool()
    def get_team_squad(team: str, season: int) -> str:
        """All players who appeared for a team in one season (from playing XIs)."""
        return _wrap(engine.team_squad, team, season)

    @mcp.tool()
    def get_playing_xi(match_id: str) -> str:
        """Named players per side for one match."""
        return _wrap(engine.playing_xi, match_id)

    @mcp.tool()
    def get_match_officials(match_id: str) -> str:
        """Umpires, TV umpire, match referee for one match."""
        return _wrap(engine.match_officials, match_id)

    @mcp.tool()
    def get_umpire_record(name: str) -> str:
        """Matches officiated by an umpire/referee, by role and season span."""
        return _wrap(engine.umpire_record, name)

    @mcp.tool()
    def search_ipl_knowledge(query: str, top_k: int = 4) -> str:
        """Semantic search over curated IPL knowledge documents (history, rules)."""
        from rag.store import search
        return _wrap(search, query, top_k)

    return mcp


if __name__ == "__main__":
    build_server().run(transport=os.environ.get("MCP_TRANSPORT", "stdio"))
