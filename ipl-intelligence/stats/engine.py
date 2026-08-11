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
