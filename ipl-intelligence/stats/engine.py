"""Statistics engine (spec §9, §10). Every function is deterministic SQL over
validated data — the LLM formats these results, never computes them.

All functions return plain dicts/lists (JSON-safe). Errors return {"error": ...}.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.db import q, q1

BOWLER_KINDS = "('bowled','caught','caught and bowled','lbw','stumped','hit wicket')"
PHASE = "CASE WHEN d.over_no < 6 THEN 'powerplay' WHEN d.over_no < 15 THEN 'middle' ELSE 'death' END"


def _sr(runs, balls):
    return round(100.0 * runs / balls, 2) if balls else None


def _econ(runs, balls):
    return round(6.0 * runs / balls, 2) if balls else None


def search_player(fragment: str) -> list[dict]:
    return q("""SELECT p.cricsheet_name AS name, p.full_name, p.role
                FROM players p WHERE p.cricsheet_name ILIKE '%%' || %s || '%%'
                   OR p.full_name ILIKE '%%' || %s || '%%'
                ORDER BY p.cricsheet_name LIMIT 15""", (fragment, fragment))


def _pid(name: str) -> int | None:
    """Fuzzy identity resolution: exact -> alias -> unique substring/surname
    match. Lets the agent pass 'Virat Kohli' directly (saves an LLM round)."""
    row = q1("SELECT id FROM players WHERE cricsheet_name = %s", (name,))
    if row:
        return row["id"]
    row = q1("SELECT player_id AS id FROM player_aliases WHERE alias = %s", (name,))
    if row:
        return row["id"]
    def _pick(hits: list[dict]) -> int | None:
        if not hits:
            return None
        if len(hits) == 1:
            return hits[0]["id"]
        # ambiguous ("R Sharma" vs "RG Sharma"): pick the most-capped candidate
        row = q1("""SELECT p.id, (SELECT COUNT(*) FROM deliveries d
                                  WHERE d.batter_id = p.id OR d.bowler_id = p.id) AS n
                    FROM players p WHERE p.id = ANY(%s)
                    ORDER BY n DESC LIMIT 1""", ([h["id"] for h in hits],))
        return row["id"] if row else None

    parts = name.strip().split()
    pid = _pick(q("SELECT id FROM players WHERE cricsheet_name ILIKE '%%' || %s || '%%' "
                  "LIMIT 5", (name,)))
    if pid:
        return pid
    if len(parts) >= 2:  # "Virat Kohli" -> initial 'V' + surname 'Kohli' -> "V Kohli"
        pid = _pick(q("SELECT id FROM players WHERE cricsheet_name ILIKE %s LIMIT 5",
                      (parts[0][0] + "%" + parts[-1],)))
        if pid:
            return pid
    return _pick(q("SELECT id FROM players WHERE cricsheet_name ILIKE '%%' || %s || '%%' "
                   "LIMIT 5", (parts[-1] if parts else "",)))


def batting_stats(player: str, season: int | None = None) -> dict:
    pid = _pid(player)
    if not pid:
        return {"error": f"unknown player '{player}' — use search_player"}
    where, params = "batter_id = %s", [pid]
    if season:
        where += " AND season = %s"
        params.append(season)
    rows = q(f"""
        SELECT season, COUNT(*) AS innings, SUM(runs) AS runs, SUM(balls) AS balls,
               SUM(fours) AS fours, SUM(sixes) AS sixes,
               SUM(CASE WHEN dismissed THEN 1 ELSE 0 END) AS dismissals,
               SUM(CASE WHEN NOT dismissed THEN 1 ELSE 0 END) AS not_outs,
               MAX(runs) AS highest,
               SUM(CASE WHEN runs >= 100 THEN 1 ELSE 0 END) AS hundreds,
               SUM(CASE WHEN runs BETWEEN 50 AND 99 THEN 1 ELSE 0 END) AS fifties
        FROM batting_innings WHERE {where} GROUP BY season ORDER BY season""", tuple(params))
    if not rows:
        return {"error": f"no batting data for '{player}'"}
    career = {k: sum(r[k] for r in rows) for k in
              ("innings", "runs", "balls", "fours", "sixes", "dismissals",
               "not_outs", "hundreds", "fifties")}
    career["highest"] = max(r["highest"] for r in rows)
    career["strike_rate"] = _sr(career["runs"], career["balls"])
    career["average"] = round(career["runs"] / career["dismissals"], 2) \
        if career["dismissals"] else None
    for r in rows:
        r["strike_rate"] = _sr(r["runs"], r["balls"])
    return {"player": player, "career": career, "by_season": rows,
            "source": "cricsheet ball-by-ball (computed)"}


def bowling_stats(player: str, season: int | None = None) -> dict:
    pid = _pid(player)
    if not pid:
        return {"error": f"unknown player '{player}' — use search_player"}
    where, params = "bowler_id = %s", [pid]
    if season:
        where += " AND season = %s"
        params.append(season)
    rows = q(f"""
        SELECT season, COUNT(*) AS innings, SUM(balls) AS balls,
               SUM(runs_conceded) AS runs_conceded, SUM(wickets) AS wickets,
               SUM(dot_balls) AS dot_balls
        FROM bowling_innings WHERE {where} GROUP BY season ORDER BY season""", tuple(params))
    if not rows:
        return {"error": f"no bowling data for '{player}'"}
    career = {k: sum(r[k] for r in rows) for k in
              ("innings", "balls", "runs_conceded", "wickets", "dot_balls")}
    career["economy"] = _econ(career["runs_conceded"], career["balls"])
    career["average"] = round(career["runs_conceded"] / career["wickets"], 2) \
        if career["wickets"] else None
    career["bowling_strike_rate"] = round(career["balls"] / career["wickets"], 2) \
        if career["wickets"] else None
    best = q1("""SELECT wickets, runs_conceded, season FROM bowling_innings
                 WHERE bowler_id = %s ORDER BY wickets DESC, runs_conceded ASC
                 LIMIT 1""", (pid,))
    career["best_figures"] = f"{best['wickets']}/{best['runs_conceded']} ({best['season']})"
    for r in rows:
        r["economy"] = _econ(r["runs_conceded"], r["balls"])
    return {"player": player, "career": career, "by_season": rows,
            "source": "cricsheet ball-by-ball (computed)"}


def fielding_stats(player: str) -> dict:
    pid = _pid(player)
    if not pid:
        return {"error": f"unknown player '{player}' — use search_player"}
    row = q1(f"""
        SELECT SUM(CASE WHEN dismissal_kind = 'caught' THEN 1 ELSE 0 END) AS catches,
               SUM(CASE WHEN dismissal_kind = 'stumped' THEN 1 ELSE 0 END) AS stumpings,
               SUM(CASE WHEN dismissal_kind = 'run out' THEN 1 ELSE 0 END) AS run_outs
        FROM deliveries WHERE fielder_id = %s""", (pid,))
    if not row or row["catches"] is None:
        return {"error": f"no fielding data for '{player}'"}
    return {"player": player, **row, "source": "cricsheet ball-by-ball (computed)"}


def highest_score(player: str) -> dict:
    pid = _pid(player)
    if not pid:
        return {"error": f"unknown player '{player}' — use search_player"}
    rows = q("""
        SELECT bi.runs, bi.balls, bi.dismissed, bi.season, m.match_date,
               v.name AS venue, t.name AS opponent
        FROM batting_innings bi
        JOIN matches m ON m.id = bi.match_id
        LEFT JOIN venues v ON v.id = m.venue_id
        LEFT JOIN teams t ON t.id = bi.opponent_id
        WHERE bi.batter_id = %s ORDER BY bi.runs DESC LIMIT 5""", (pid,))
    if not rows:
        return {"error": f"no innings for '{player}'"}
    for r in rows:
        r["strike_rate"] = _sr(r["runs"], r["balls"])
        r["not_out"] = not r.pop("dismissed")
    return {"player": player, "top_innings": rows}


def phase_stats(player: str, discipline: str = "batting") -> dict:
    pid = _pid(player)
    if not pid:
        return {"error": f"unknown player '{player}' — use search_player"}
    if discipline == "bowling":
        rows = q(f"""
            SELECT {PHASE} AS phase,
                   SUM(CASE WHEN d.wides = 0 AND d.noballs = 0 THEN 1 ELSE 0 END) AS balls,
                   SUM(d.runs_total - d.byes - d.legbyes) AS runs_conceded,
                   SUM(CASE WHEN d.dismissal_kind IN {BOWLER_KINDS} THEN 1 ELSE 0 END) AS wickets,
                   SUM(CASE WHEN d.runs_total = 0 THEN 1 ELSE 0 END) AS dots
            FROM deliveries d WHERE d.bowler_id = %s GROUP BY 1""", (pid,))
        for r in rows:
            r["economy"] = _econ(r["runs_conceded"], r["balls"])
            r["dot_pct"] = round(100.0 * r["dots"] / r["balls"], 1) if r["balls"] else None
    else:
        rows = q(f"""
            SELECT {PHASE} AS phase, SUM(d.runs_batter) AS runs,
                   SUM(CASE WHEN d.wides = 0 THEN 1 ELSE 0 END) AS balls,
                   SUM(CASE WHEN d.runs_batter IN (4,6) THEN 1 ELSE 0 END) AS boundaries,
                   SUM(CASE WHEN d.runs_batter = 0 AND d.runs_extras = 0 THEN 1 ELSE 0 END) AS dots
            FROM deliveries d WHERE d.batter_id = %s GROUP BY 1""", (pid,))
        for r in rows:
            r["strike_rate"] = _sr(r["runs"], r["balls"])
            r["dot_pct"] = round(100.0 * r["dots"] / r["balls"], 1) if r["balls"] else None
            r["boundary_pct"] = round(100.0 * r["boundaries"] / r["balls"], 1) if r["balls"] else None
    if not rows:
        return {"error": f"no {discipline} data for '{player}'"}
    return {"player": player, "discipline": discipline,
            "phases": {r.pop("phase"): r for r in rows}}


def player_vs_team(player: str, team: str) -> dict:
    pid = _pid(player)
    if not pid:
        return {"error": f"unknown player '{player}' — use search_player"}
    bat = q1("""
        SELECT COUNT(DISTINCT d.match_id) AS matches, SUM(d.runs_batter) AS runs,
               SUM(CASE WHEN d.wides = 0 THEN 1 ELSE 0 END) AS balls,
               SUM(CASE WHEN d.player_out_id = d.batter_id THEN 1 ELSE 0 END) AS dismissals
        FROM deliveries d JOIN teams t ON t.id = d.bowling_team_id
        WHERE d.batter_id = %s AND t.name ILIKE '%%' || %s || '%%'""", (pid, team))
    bowl = q1(f"""
        SELECT COUNT(DISTINCT d.match_id) AS matches,
               SUM(CASE WHEN d.wides = 0 AND d.noballs = 0 THEN 1 ELSE 0 END) AS balls,
               SUM(d.runs_total - d.byes - d.legbyes) AS runs_conceded,
               SUM(CASE WHEN d.dismissal_kind IN {BOWLER_KINDS} THEN 1 ELSE 0 END) AS wickets
        FROM deliveries d JOIN teams t ON t.id = d.batting_team_id
        WHERE d.bowler_id = %s AND t.name ILIKE '%%' || %s || '%%'""", (pid, team))
    out = {"player": player, "vs_team": team}
    if bat and bat["balls"]:
        bat["strike_rate"] = _sr(bat["runs"], bat["balls"])
        bat["average"] = round(bat["runs"] / bat["dismissals"], 2) if bat["dismissals"] else None
        out["batting"] = bat
    if bowl and bowl["balls"]:
        bowl["economy"] = _econ(bowl["runs_conceded"], bowl["balls"])
        out["bowling"] = bowl
    return out if len(out) > 2 else {"error": f"no data for '{player}' vs '{team}'"}


def batter_vs_bowler(batter: str, bowler: str) -> dict:
    bid, wid = _pid(batter), _pid(bowler)
    if not bid or not wid:
        return {"error": "unknown player name(s) — use search_player"}
    row = q1(f"""
        SELECT SUM(CASE WHEN wides = 0 THEN 1 ELSE 0 END) AS balls,
               SUM(runs_batter) AS runs,
               SUM(CASE WHEN player_out_id = %s AND dismissal_kind IN {BOWLER_KINDS}
                   THEN 1 ELSE 0 END) AS dismissals
        FROM deliveries WHERE batter_id = %s AND bowler_id = %s""", (bid, bid, wid))
    if not row or not row["balls"]:
        return {"error": f"no deliveries between '{batter}' and '{bowler}'"}
    row["strike_rate"] = _sr(row["runs"], row["balls"])
    return {"batter": batter, "bowler": bowler, **row}


def head_to_head(team_a: str, team_b: str) -> dict:
    rows = q("""
        SELECT COALESCE(w.name, 'tie/no result') AS winner, COUNT(*) AS wins
        FROM matches m
        JOIN teams t1 ON t1.id = m.team1_id
        JOIN teams t2 ON t2.id = m.team2_id
        LEFT JOIN teams w ON w.id = m.winner_id
        WHERE (t1.name ILIKE '%%'||%s||'%%' AND t2.name ILIKE '%%'||%s||'%%')
           OR (t1.name ILIKE '%%'||%s||'%%' AND t2.name ILIKE '%%'||%s||'%%')
        GROUP BY 1 ORDER BY wins DESC""", (team_a, team_b, team_b, team_a))
    return {"head_to_head": rows} if rows else \
        {"error": f"no matches between '{team_a}' and '{team_b}'"}


def team_season(team: str, season: int) -> dict:
    row = q1("""
        SELECT t.name AS team,
               COUNT(*) AS matches,
               SUM(CASE WHEN m.winner_id = t.id THEN 1 ELSE 0 END) AS wins,
               SUM(CASE WHEN m.winner_id IS NOT NULL AND m.winner_id != t.id
                   THEN 1 ELSE 0 END) AS losses,
               SUM(CASE WHEN m.winner_id IS NULL THEN 1 ELSE 0 END) AS no_result_or_tie
        FROM matches m JOIN teams t ON t.id IN (m.team1_id, m.team2_id)
        WHERE t.name ILIKE '%%'||%s||'%%' AND m.season = %s GROUP BY t.name""",
        (team, season))
    return row or {"error": f"no data for '{team}' in {season}"}


def season_summary(season: int) -> dict:
    final = q1("""
        SELECT m.match_date, v.name AS venue, t1.name AS team1, t2.name AS team2,
               w.name AS winner, m.win_by_type, m.win_by_margin,
               p.cricsheet_name AS player_of_match, m.event_stage
        FROM matches m
        LEFT JOIN venues v ON v.id = m.venue_id
        LEFT JOIN teams t1 ON t1.id = m.team1_id
        LEFT JOIN teams t2 ON t2.id = m.team2_id
        LEFT JOIN teams w ON w.id = m.winner_id
        LEFT JOIN players p ON p.id = m.player_of_match_id
        WHERE m.season = %s ORDER BY m.match_date DESC LIMIT 1""", (season,))
    if not final:
        return {"error": f"no data for IPL {season} (coverage 2008-2025)"}
    orange = q1("""SELECT p.cricsheet_name AS player, SUM(bi.runs) AS runs
                   FROM batting_innings bi JOIN players p ON p.id = bi.batter_id
                   WHERE bi.season = %s GROUP BY 1 ORDER BY runs DESC LIMIT 1""", (season,))
    purple = q1("""SELECT p.cricsheet_name AS player, SUM(bo.wickets) AS wickets
                   FROM bowling_innings bo JOIN players p ON p.id = bo.bowler_id
                   WHERE bo.season = %s GROUP BY 1 ORDER BY wickets DESC LIMIT 1""", (season,))
    n = q1("SELECT COUNT(*) AS c FROM matches WHERE season = %s", (season,))
    return {"season": season, "matches": n["c"], "final": final,
            "most_runs_orange_cap": orange, "most_wickets_purple_cap": purple,
            "source": "computed from cricsheet ball-by-ball"}


def venue_stats(venue: str) -> dict:
    row = q1("""
        SELECT v.name, v.city, COUNT(DISTINCT m.id) AS matches,
               ROUND(AVG(inn.total), 1) AS avg_innings_score
        FROM venues v
        JOIN matches m ON m.venue_id = v.id
        JOIN (SELECT match_id, innings, SUM(runs_total) AS total
              FROM deliveries GROUP BY match_id, innings) inn ON inn.match_id = m.id
        WHERE v.name ILIKE '%%'||%s||'%%' OR v.city ILIKE '%%'||%s||'%%'
        GROUP BY v.name, v.city ORDER BY matches DESC LIMIT 1""", (venue, venue))
    return row or {"error": f"no venue matching '{venue}'"}


def records() -> dict:
    return {
        "most_career_runs": q("""
            SELECT p.cricsheet_name AS player, SUM(bi.runs) AS runs
            FROM batting_innings bi JOIN players p ON p.id = bi.batter_id
            GROUP BY 1 ORDER BY runs DESC LIMIT 5"""),
        "most_career_wickets": q("""
            SELECT p.cricsheet_name AS player, SUM(bo.wickets) AS wickets
            FROM bowling_innings bo JOIN players p ON p.id = bo.bowler_id
            GROUP BY 1 ORDER BY wickets DESC LIMIT 5"""),
        "highest_individual_scores": q("""
            SELECT p.cricsheet_name AS player, bi.runs, bi.balls, bi.season
            FROM batting_innings bi JOIN players p ON p.id = bi.batter_id
            ORDER BY bi.runs DESC LIMIT 3"""),
        "best_bowling_figures": q("""
            SELECT p.cricsheet_name AS player, bo.wickets, bo.runs_conceded, bo.season
            FROM bowling_innings bo JOIN players p ON p.id = bo.bowler_id
            ORDER BY bo.wickets DESC, bo.runs_conceded ASC LIMIT 3"""),
        "highest_team_totals": q("""
            SELECT t.name AS team, x.total, m.season
            FROM (SELECT match_id, innings, batting_team_id, SUM(runs_total) AS total
                  FROM deliveries GROUP BY 1,2,3) x
            JOIN teams t ON t.id = x.batting_team_id
            JOIN matches m ON m.id = x.match_id
            ORDER BY x.total DESC LIMIT 3"""),
        "most_career_sixes": q("""
            SELECT p.cricsheet_name AS player, SUM(bi.sixes) AS sixes
            FROM batting_innings bi JOIN players p ON p.id = bi.batter_id
            GROUP BY 1 ORDER BY sixes DESC LIMIT 3"""),
        "largest_win_by_runs": q("""
            SELECT w.name AS winner, m.win_by_margin AS runs, m.season
            FROM matches m JOIN teams w ON w.id = m.winner_id
            WHERE m.win_by_type = 'runs' ORDER BY m.win_by_margin DESC LIMIT 3"""),
        "fastest_fifties_by_balls": q("""
            WITH cum AS (
                SELECT d.match_id, d.innings, d.batter_id,
                       SUM(d.runs_batter) OVER w AS cum_runs,
                       ROW_NUMBER() OVER w AS balls_faced
                FROM deliveries d WHERE d.wides = 0
                WINDOW w AS (PARTITION BY d.match_id, d.innings, d.batter_id
                             ORDER BY d.over_no, d.ball_no)
            )
            SELECT p.cricsheet_name AS player, MIN(c.balls_faced) AS balls, m.season
            FROM cum c
            JOIN players p ON p.id = c.batter_id
            JOIN matches m ON m.id = c.match_id
            WHERE c.cum_runs >= 50
            GROUP BY p.cricsheet_name, c.match_id, c.innings, m.season
            ORDER BY balls ASC LIMIT 3"""),
        "most_player_of_match_awards": q("""
            SELECT p.cricsheet_name AS player, COUNT(*) AS awards
            FROM matches m JOIN players p ON p.id = m.player_of_match_id
            GROUP BY 1 ORDER BY awards DESC LIMIT 3"""),
        "fastest_centuries_by_balls": q("""
            WITH cum AS (
                SELECT d.match_id, d.innings, d.batter_id,
                       SUM(d.runs_batter) OVER w AS cum_runs,
                       ROW_NUMBER() OVER w AS balls_faced
                FROM deliveries d WHERE d.wides = 0
                WINDOW w AS (PARTITION BY d.match_id, d.innings, d.batter_id
                             ORDER BY d.over_no, d.ball_no))
            SELECT p.cricsheet_name AS player, MIN(c.balls_faced) AS balls, m.season
            FROM cum c JOIN players p ON p.id = c.batter_id
            JOIN matches m ON m.id = c.match_id
            WHERE c.cum_runs >= 100
            GROUP BY p.cricsheet_name, c.match_id, c.innings, m.season
            ORDER BY balls ASC LIMIT 3"""),
        "most_centuries": q("""
            SELECT p.cricsheet_name AS player,
                   SUM(CASE WHEN bi.runs >= 100 THEN 1 ELSE 0 END) AS hundreds
            FROM batting_innings bi JOIN players p ON p.id = bi.batter_id
            GROUP BY 1 HAVING SUM(CASE WHEN bi.runs >= 100 THEN 1 ELSE 0 END) > 0
            ORDER BY hundreds DESC LIMIT 3"""),
        "most_fifties": q("""
            SELECT p.cricsheet_name AS player,
                   SUM(CASE WHEN bi.runs BETWEEN 50 AND 99 THEN 1 ELSE 0 END) AS fifties
            FROM batting_innings bi JOIN players p ON p.id = bi.batter_id
            GROUP BY 1 ORDER BY fifties DESC LIMIT 3"""),
        "best_career_economy_min_300_balls": q("""
            SELECT p.cricsheet_name AS player,
                   ROUND(6.0 * SUM(bo.runs_conceded) / SUM(bo.balls), 2) AS economy,
                   SUM(bo.balls) AS balls
            FROM bowling_innings bo JOIN players p ON p.id = bo.bowler_id
            GROUP BY 1 HAVING SUM(bo.balls) >= 300
            ORDER BY economy ASC LIMIT 3"""),
        "best_career_strike_rate_min_500_balls": q("""
            SELECT p.cricsheet_name AS player,
                   ROUND(100.0 * SUM(bi.runs) / SUM(bi.balls), 2) AS strike_rate,
                   SUM(bi.runs) AS runs
            FROM batting_innings bi JOIN players p ON p.id = bi.batter_id
            GROUP BY 1 HAVING SUM(bi.balls) >= 500
            ORDER BY strike_rate DESC LIMIT 3"""),
        "most_maidens": q("""
            WITH overs AS (
                SELECT d.match_id, d.bowler_id, d.over_no,
                       SUM(d.runs_total - d.byes - d.legbyes) AS conceded,
                       COUNT(*) FILTER (WHERE d.wides = 0 AND d.noballs = 0) AS legal
                FROM deliveries d GROUP BY 1, 2, 3)
            SELECT p.cricsheet_name AS player, COUNT(*) AS maidens
            FROM overs o JOIN players p ON p.id = o.bowler_id
            WHERE o.conceded = 0 AND o.legal >= 6
            GROUP BY 1 ORDER BY maidens DESC LIMIT 3"""),
        "most_dot_balls_bowled": q("""
            SELECT p.cricsheet_name AS player, SUM(bo.dot_balls) AS dot_balls
            FROM bowling_innings bo JOIN players p ON p.id = bo.bowler_id
            GROUP BY 1 ORDER BY dot_balls DESC LIMIT 3"""),
        "most_career_catches": q("""
            SELECT p.cricsheet_name AS player, COUNT(*) AS catches
            FROM deliveries d JOIN players p ON p.id = d.fielder_id
            WHERE d.dismissal_kind = 'caught'
            GROUP BY 1 ORDER BY catches DESC LIMIT 3"""),
        "lowest_completed_team_totals": q("""
            SELECT t.name AS team, x.total, m.season
            FROM (SELECT match_id, innings, batting_team_id,
                         SUM(runs_total) AS total,
                         SUM(CASE WHEN is_wicket THEN 1 ELSE 0 END) AS wkts,
                         COUNT(*) FILTER (WHERE wides = 0 AND noballs = 0) AS legal
                  FROM deliveries GROUP BY 1, 2, 3) x
            JOIN teams t ON t.id = x.batting_team_id
            JOIN matches m ON m.id = x.match_id
            WHERE x.wkts >= 10 OR x.legal >= 120
            ORDER BY x.total ASC LIMIT 3"""),
        "largest_successful_chases": q("""
            SELECT t.name AS team, x.total, m.season, opp.name AS opponent
            FROM (SELECT match_id, batting_team_id, MIN(bowling_team_id) AS bowl_id,
                         SUM(runs_total) AS total
                  FROM deliveries WHERE innings = 2 GROUP BY 1, 2) x
            JOIN matches m ON m.id = x.match_id AND m.winner_id = x.batting_team_id
            JOIN teams t ON t.id = x.batting_team_id
            JOIN teams opp ON opp.id = x.bowl_id
            ORDER BY x.total DESC LIMIT 3"""),
        "best_partnerships": q("""
            SELECT p1.cricsheet_name AS batter_a, p2.cricsheet_name AS batter_b,
                   x.runs, m.season
            FROM (SELECT match_id, innings,
                         LEAST(batter_id, non_striker_id) AS a,
                         GREATEST(batter_id, non_striker_id) AS b,
                         SUM(runs_total) AS runs
                  FROM deliveries GROUP BY 1, 2, 3, 4) x
            JOIN players p1 ON p1.id = x.a
            JOIN players p2 ON p2.id = x.b
            JOIN matches m ON m.id = x.match_id
            ORDER BY x.runs DESC LIMIT 3"""),
        "source": "computed live from cricsheet ball-by-ball; never hard-coded",
    }


def match_scorecard(match_id: str) -> dict:
    info = q1("""
        SELECT m.id, m.season, m.match_date, v.name AS venue, t1.name AS team1,
               t2.name AS team2, w.name AS winner, m.win_by_type, m.win_by_margin
        FROM matches m
        LEFT JOIN venues v ON v.id = m.venue_id
        LEFT JOIN teams t1 ON t1.id = m.team1_id
        LEFT JOIN teams t2 ON t2.id = m.team2_id
        LEFT JOIN teams w ON w.id = m.winner_id
        WHERE m.id = %s""", (match_id,))
    if not info:
        return {"error": f"unknown match id '{match_id}'"}
    batting = q("""
        SELECT bi.match_id, p.cricsheet_name AS batter, bi.runs, bi.balls
        FROM batting_innings bi JOIN players p ON p.id = bi.batter_id
        WHERE bi.match_id = %s ORDER BY bi.runs DESC LIMIT 8""", (match_id,))
    bowling = q("""
        SELECT p.cricsheet_name AS bowler, bo.wickets, bo.runs_conceded, bo.balls
        FROM bowling_innings bo JOIN players p ON p.id = bo.bowler_id
        WHERE bo.match_id = %s ORDER BY bo.wickets DESC LIMIT 8""", (match_id,))
    return {"match": info, "top_batting": batting, "top_bowling": bowling}


def points_table(season: int) -> dict:
    """League-stage points table with wins/losses/points and computed NRR
    (all-out innings counted as full 20 overs, per official method)."""
    inn = q("""
        WITH innings_agg AS (
            SELECT d.match_id, d.innings, d.batting_team_id, d.bowling_team_id,
                   SUM(d.runs_total) AS runs,
                   SUM(CASE WHEN d.wides = 0 AND d.noballs = 0 THEN 1 ELSE 0 END) AS balls,
                   SUM(CASE WHEN d.is_wicket AND d.dismissal_kind NOT IN
                       ('retired hurt','retired out') THEN 1 ELSE 0 END) AS wickets
            FROM deliveries d
            JOIN matches m ON m.id = d.match_id
            WHERE m.season = %s AND m.event_stage IS NULL
            GROUP BY 1,2,3,4)
        SELECT * FROM innings_agg""", (season,))
    if not inn:
        return {"error": f"no league-stage data for IPL {season}"}
    stats: dict[int, dict] = {}
    for r in inn:
        overs = 20.0 if r["wickets"] >= 10 else r["balls"] / 6.0
        f = stats.setdefault(r["batting_team_id"],
                             {"runs_for": 0, "overs_faced": 0.0,
                              "runs_against": 0, "overs_bowled": 0.0})
        f["runs_for"] += r["runs"]
        f["overs_faced"] += overs
        a = stats.setdefault(r["bowling_team_id"],
                             {"runs_for": 0, "overs_faced": 0.0,
                              "runs_against": 0, "overs_bowled": 0.0})
        a["runs_against"] += r["runs"]
        a["overs_bowled"] += overs
    results = q("""
        SELECT t.id, t.name,
               COUNT(*) AS played,
               SUM(CASE WHEN m.winner_id = t.id THEN 1 ELSE 0 END) AS wins,
               SUM(CASE WHEN m.winner_id IS NOT NULL AND m.winner_id != t.id
                   THEN 1 ELSE 0 END) AS losses,
               SUM(CASE WHEN m.winner_id IS NULL THEN 1 ELSE 0 END) AS no_results
        FROM matches m JOIN teams t ON t.id IN (m.team1_id, m.team2_id)
        WHERE m.season = %s AND m.event_stage IS NULL
        GROUP BY t.id, t.name""", (season,))
    table = []
    for r in results:
        s = stats.get(r["id"], {})
        nrr = None
        if s.get("overs_faced") and s.get("overs_bowled"):
            nrr = round(s["runs_for"] / s["overs_faced"]
                        - s["runs_against"] / s["overs_bowled"], 3)
        table.append({"team": r["name"], "played": r["played"], "wins": r["wins"],
                      "losses": r["losses"], "no_results": r["no_results"],
                      "points": 2 * r["wins"] + r["no_results"], "nrr": nrr})
    table.sort(key=lambda x: (-x["points"], -(x["nrr"] or -99)))
    return {"season": season, "league_stage_table": table,
            "note": "computed from ball-by-ball; NRR uses full 20 overs for all-out innings"}


def team_history(team: str) -> dict:
    """Titles, final appearances, playoff appearances, per-season W/L."""
    seasons = q("""
        SELECT m.season,
               COUNT(*) AS played,
               SUM(CASE WHEN m.winner_id = t.id THEN 1 ELSE 0 END) AS wins,
               SUM(CASE WHEN m.winner_id IS NOT NULL AND m.winner_id != t.id
                   THEN 1 ELSE 0 END) AS losses
        FROM matches m JOIN teams t ON t.id IN (m.team1_id, m.team2_id)
        WHERE t.name ILIKE '%%'||%s||'%%'
        GROUP BY m.season ORDER BY m.season""", (team,))
    if not seasons:
        return {"error": f"no team matching '{team}'"}
    finals = q("""
        SELECT m.season, (m.winner_id = t.id) AS won
        FROM matches m JOIN teams t ON t.id IN (m.team1_id, m.team2_id)
        WHERE t.name ILIKE '%%'||%s||'%%' AND m.event_stage = 'Final'
        ORDER BY m.season""", (team,))
    playoffs = q("""
        SELECT DISTINCT m.season
        FROM matches m JOIN teams t ON t.id IN (m.team1_id, m.team2_id)
        WHERE t.name ILIKE '%%'||%s||'%%' AND m.event_stage IS NOT NULL""", (team,))
    return {"team": team,
            "titles": [f["season"] for f in finals if f["won"]],
            "final_appearances": [f["season"] for f in finals],
            "playoff_seasons": sorted(p["season"] for p in playoffs),
            "seasons": seasons,
            "note": "finals/playoffs detected from match stage data (2011+ tagging is most complete)"}


def player_profile(player: str) -> dict:
    """Profile: full name, nationality, role, styles (curated for famous
    players), plus teams played for and career span from match data."""
    pid = _pid(player)
    if not pid:
        return {"error": f"unknown player '{player}' — use search_player"}
    p = q1("SELECT * FROM players WHERE id = %s", (pid,))
    teams = q("""
        SELECT t.name AS team, MIN(m.season) AS first, MAX(m.season) AS last,
               COUNT(*) AS matches
        FROM match_players mp
        JOIN teams t ON t.id = mp.team_id
        JOIN matches m ON m.id = mp.match_id
        WHERE mp.player_id = %s GROUP BY t.name ORDER BY MIN(m.season)""", (pid,))
    pom = q1("SELECT COUNT(*) AS c FROM matches WHERE player_of_match_id = %s", (pid,))
    out = {"player": p["cricsheet_name"], "teams": teams,
           "player_of_match_awards": pom["c"]}
    for k in ("full_name", "nationality", "role", "batting_style", "bowling_style"):
        out[k] = p[k] or "not on file (curated bios cover the most famous players)"
    return out


def player_team_history(player: str) -> dict:
    """Season-by-season franchise history — team changes show as transfers/auction moves."""
    pid = _pid(player)
    if not pid:
        return {"error": f"unknown player '{player}' — use search_player"}
    rows = q("""
        SELECT m.season, t.name AS team, COUNT(*) AS matches
        FROM match_players mp
        JOIN teams t ON t.id = mp.team_id
        JOIN matches m ON m.id = mp.match_id
        WHERE mp.player_id = %s
        GROUP BY m.season, t.name ORDER BY m.season""", (pid,))
    moves = []
    prev = None
    for r in rows:
        if prev and r["team"] != prev:
            moves.append(f"{r['season']}: moved to {r['team']}")
        prev = r["team"]
    return {"player": player, "by_season": rows, "team_changes": moves}


def match_partnerships(match_id: str) -> dict:
    """Partnership runs per batting pair in one match."""
    rows = q("""
        SELECT d.innings, t.name AS batting_team,
               p1.cricsheet_name AS batter_a, p2.cricsheet_name AS batter_b,
               SUM(d.runs_total) AS partnership_runs, COUNT(*) AS balls
        FROM deliveries d
        JOIN players p1 ON p1.id = LEAST(d.batter_id, d.non_striker_id)
        JOIN players p2 ON p2.id = GREATEST(d.batter_id, d.non_striker_id)
        JOIN teams t ON t.id = d.batting_team_id
        WHERE d.match_id = %s
        GROUP BY 1, 2, 3, 4 ORDER BY partnership_runs DESC""", (match_id,))
    return {"match_id": match_id, "partnerships": rows[:12]} if rows else \
        {"error": f"unknown match id '{match_id}'"}


def fall_of_wickets(match_id: str) -> dict:
    """Score at each dismissal, in order, per innings."""
    rows = q("""
        SELECT d.innings, d.over_no, d.ball_no,
               SUM(d.runs_total) OVER (PARTITION BY d.innings
                   ORDER BY d.over_no, d.ball_no) AS team_score,
               d.is_wicket, p.cricsheet_name AS player_out, d.dismissal_kind
        FROM deliveries d
        LEFT JOIN players p ON p.id = d.player_out_id
        WHERE d.match_id = %s ORDER BY d.innings, d.over_no, d.ball_no""",
        (match_id,))
    if not rows:
        return {"error": f"unknown match id '{match_id}'"}
    fow: dict[int, list] = {}
    counts: dict[int, int] = {}
    for r in rows:
        if r["is_wicket"] and r["player_out"]:
            n = counts.get(r["innings"], 0) + 1
            counts[r["innings"]] = n
            fow.setdefault(r["innings"], []).append(
                f"{n}-{r['team_score']} ({r['player_out']}, "
                f"{r['over_no']}.{r['ball_no']} ov, {r['dismissal_kind']})")
    return {"match_id": match_id, "fall_of_wickets": fow}


def match_officials(match_id: str) -> dict:
    """Umpires, TV umpire, match referee for one match."""
    rows = q("SELECT person, role FROM officials WHERE match_id = %s ORDER BY role",
             (match_id,))
    return {"match_id": match_id, "officials": rows} if rows else \
        {"error": f"no officials recorded for match '{match_id}'"}


def umpire_record(name: str) -> dict:
    """How many IPL matches a match official has officiated, by role and season span."""
    rows = q("""SELECT o.role, COUNT(*) AS matches,
                       MIN(m.season) AS first_season, MAX(m.season) AS last_season
                FROM officials o JOIN matches m ON m.id = o.match_id
                WHERE o.person ILIKE '%%' || %s || '%%'
                GROUP BY o.role ORDER BY matches DESC""", (name,))
    return {"official": name, "record": rows} if rows else \
        {"error": f"no official matching '{name}'"}


def team_squad(team: str, season: int) -> dict:
    """Every player who appeared for a team in a season (from playing XIs)."""
    rows = q("""SELECT p.cricsheet_name AS player, COUNT(*) AS matches
                FROM match_players mp
                JOIN matches m ON m.id = mp.match_id
                JOIN teams t ON t.id = mp.team_id
                JOIN players p ON p.id = mp.player_id
                WHERE t.name ILIKE '%%' || %s || '%%' AND m.season = %s
                GROUP BY p.cricsheet_name ORDER BY matches DESC""", (team, season))
    return {"team": team, "season": season, "squad_size": len(rows),
            "players": rows} if rows else \
        {"error": f"no squad found for '{team}' in {season}"}


def playing_xi(match_id: str) -> dict:
    """The 11 (12 with Impact Player era) named players per side for one match."""
    rows = q("""SELECT t.name AS team, p.cricsheet_name AS player
                FROM match_players mp
                JOIN teams t ON t.id = mp.team_id
                JOIN players p ON p.id = mp.player_id
                WHERE mp.match_id = %s ORDER BY t.name""", (match_id,))
    if not rows:
        return {"error": f"no XI recorded for match '{match_id}'"}
    out: dict[str, list] = {}
    for r in rows:
        out.setdefault(r["team"], []).append(r["player"])
    return {"match_id": match_id, "teams": out}


def team_staff(team: str, season: int | None = None) -> dict:
    where, params = "t.name ILIKE '%%'||%s||'%%'", [team]
    if season:
        where += " AND ts.season = %s"
        params.append(season)
    rows = q(f"""SELECT t.name AS team, ts.season, ts.person, ts.role
                 FROM team_staff ts JOIN teams t ON t.id = ts.team_id
                 WHERE {where} ORDER BY ts.season, ts.role""", tuple(params))
    if not rows:
        return {"error": "coach/support-staff data is not loaded for this query: "
                         "no free licensed source provides it, and this system "
                         "never fabricates data. Rows can be added via the manual "
                         "CSV importer (ingest/staff_csv.py) with provenance."}
    return {"staff": rows, "source": "manually verified CSV (see data_sources)"}
