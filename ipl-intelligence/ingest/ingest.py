"""Ingestion pipeline (spec §5):
raw Cricsheet JSON -> validation -> normalization -> identity resolution
-> insertion -> derived stats (materialized views) -> validation report.

Run: python -m ingest.ingest
"""
import json
import logging
import sys
import zipfile
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import config
from core.db import execute, pool, q, q1

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
log = logging.getLogger("ingest")

CUTOFF = datetime.strptime(config.DATA_CUTOFF, "%Y-%m-%d")
BOWLER_KINDS = ("bowled", "caught", "caught and bowled", "lbw", "stumped", "hit wicket")


def download() -> Path:
    raw = Path(config.RAW_DIR)
    match_dir = raw / "ipl_matches"
    if match_dir.exists() and list(match_dir.glob("*.json")):
        log.info("raw data already present at %s", match_dir)
        return match_dir
    raw.mkdir(parents=True, exist_ok=True)
    zip_path = raw / "ipl_json.zip"
    if not zip_path.exists():
        import requests
        log.info("downloading %s", config.CRICSHEET_URL)
        r = requests.get(config.CRICSHEET_URL, timeout=180)
        r.raise_for_status()
        zip_path.write_bytes(r.content)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(match_dir)
    return match_dir


class Ids:
    """Identity resolution caches: name -> surrogate id (insert-on-miss)."""

    def __init__(self):
        self.teams: dict[str, int] = {}
        self.players: dict[str, int] = {}
        self.venues: dict[str, int] = {}

    def team(self, name: str | None) -> int | None:
        if not name:
            return None
        if name not in self.teams:
            row = q1("INSERT INTO teams (name) VALUES (%s) "
                     "ON CONFLICT (name) DO UPDATE SET name = EXCLUDED.name "
                     "RETURNING id", (name,))
            self.teams[name] = row["id"]
        return self.teams[name]

    def player(self, name: str | None) -> int | None:
        if not name:
            return None
        if name not in self.players:
            row = q1("INSERT INTO players (cricsheet_name) VALUES (%s) "
                     "ON CONFLICT (cricsheet_name) DO UPDATE SET cricsheet_name = EXCLUDED.cricsheet_name "
                     "RETURNING id", (name,))
            self.players[name] = row["id"]
        return self.players[name]

    def venue(self, name: str | None, city: str | None) -> int | None:
        if not name:
            return None
        if name not in self.venues:
            row = q1("INSERT INTO venues (name, city) VALUES (%s, %s) "
                     "ON CONFLICT (name) DO UPDATE SET name = EXCLUDED.name "
                     "RETURNING id", (name, city))
            self.venues[name] = row["id"]
        return self.venues[name]


def validate_file(data: dict) -> str | None:
    """Return rejection reason or None if valid."""
    info = data.get("info", {})
    if not info.get("dates"):
        return "missing dates"
    try:
        dt = datetime.strptime(info["dates"][0], "%Y-%m-%d")
    except ValueError:
        return "bad date format"
    if dt > CUTOFF:
        return "post-cutoff"
    if len(info.get("teams", [])) != 2:
        return "team count != 2"
    return None


def ingest_match(fp: Path, ids: Ids) -> tuple[int, int] | None:
    if q1("SELECT 1 AS x FROM matches WHERE id = %s", (fp.stem,)):
        return None  # idempotent re-runs: never double-insert deliveries
    data = json.loads(fp.read_text(encoding="utf-8"))
    reason = validate_file(data)
    if reason:
        return None
    info = data["info"]
    dt = info["dates"][0]
    teams = info["teams"]
    outcome = info.get("outcome", {})
    by = outcome.get("by", {})
    toss = info.get("toss", {})
    pom = (info.get("player_of_match") or [None])[0]
    season = int(dt[:4])

    execute("INSERT INTO seasons (year) VALUES (%s) ON CONFLICT DO NOTHING", (season,))
    execute("""
        INSERT INTO matches (id, season, match_date, venue_id, team1_id, team2_id,
            toss_winner_id, toss_decision, winner_id, win_by_type, win_by_margin,
            result_type, player_of_match_id, event_stage)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        ON CONFLICT (id) DO NOTHING""", (
        fp.stem, season, dt, ids.venue(info.get("venue"), info.get("city")),
        ids.team(teams[0]), ids.team(teams[1]),
        ids.team(toss.get("winner")), toss.get("decision"),
        ids.team(outcome.get("winner")),
        "runs" if "runs" in by else ("wickets" if "wickets" in by else None),
        by.get("runs") or by.get("wickets"),
        outcome.get("result", "normal"), ids.player(pom),
        (info.get("event") or {}).get("stage")))

    rows = []
    for inn_idx, inning in enumerate(data.get("innings", []), start=1):
        batting = inning.get("team")
        bowling = teams[0] if batting == teams[1] else teams[1]
        bat_id, bowl_id = ids.team(batting), ids.team(bowling)
        for over in inning.get("overs", []):
            for ball_idx, d in enumerate(over.get("deliveries", []), start=1):
                runs, extras = d.get("runs", {}), d.get("extras", {})
                wickets = d.get("wickets", [])
                fielder = None
                if wickets and wickets[0].get("fielders"):
                    fielder = wickets[0]["fielders"][0].get("name")
                rows.append((
                    fp.stem, inn_idx, over.get("over"), ball_idx, bat_id, bowl_id,
                    ids.player(d.get("batter")), ids.player(d.get("bowler")),
                    ids.player(d.get("non_striker")),
                    runs.get("batter", 0), runs.get("extras", 0), runs.get("total", 0),
                    extras.get("wides", 0), extras.get("noballs", 0),
                    extras.get("byes", 0), extras.get("legbyes", 0),
                    bool(wickets),
                    ids.player(wickets[0].get("player_out")) if wickets else None,
                    wickets[0].get("kind") if wickets else None,
                    ids.player(fielder)))
    with pool().connection() as conn:
        with conn.cursor() as cur:
            cur.executemany("""
                INSERT INTO deliveries (match_id, innings, over_no, ball_no,
                    batting_team_id, bowling_team_id, batter_id, bowler_id,
                    non_striker_id, runs_batter, runs_extras, runs_total,
                    wides, noballs, byes, legbyes, is_wicket, player_out_id,
                    dismissal_kind, fielder_id)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                rows)
    return 1, len(rows)


def validation_report() -> dict:
    """Spec §26 — integrity checks over the loaded data."""
    checks = {
        "matches": "SELECT COUNT(*) c FROM matches",
        "deliveries": "SELECT COUNT(*) c FROM deliveries",
        "players": "SELECT COUNT(*) c FROM players",
        "negative_runs": "SELECT COUNT(*) c FROM deliveries WHERE runs_total < 0",
        "orphan_deliveries": """SELECT COUNT(*) c FROM deliveries d
            LEFT JOIN matches m ON m.id = d.match_id WHERE m.id IS NULL""",
        "duplicate_deliveries": """SELECT COUNT(*) c FROM (
            SELECT match_id, innings, over_no, ball_no, COUNT(*)
            FROM deliveries GROUP BY 1,2,3,4 HAVING COUNT(*) > 1) x""",
        "matches_without_winner_or_result": """SELECT COUNT(*) c FROM matches
            WHERE winner_id IS NULL AND result_type = 'normal'""",
        "invalid_overs": "SELECT COUNT(*) c FROM deliveries WHERE over_no NOT BETWEEN 0 AND 19",
    }
    return {name: q1(sql)["c"] for name, sql in checks.items()}


def main() -> None:
    schema = (Path(__file__).resolve().parent.parent / "db" / "schema.sql").read_text()
    with pool().connection() as conn:
        conn.execute(schema)
    match_dir = download()

    ids = Ids()
    n_matches = n_deliveries = n_skipped = 0
    for fp in sorted(match_dir.glob("*.json")):
        result = ingest_match(fp, ids)
        if result is None:
            n_skipped += 1
            continue
        n_matches += result[0]
        n_deliveries += result[1]

    execute("REFRESH MATERIALIZED VIEW batting_innings")
    execute("REFRESH MATERIALIZED VIEW bowling_innings")
    execute("""UPDATE seasons s SET matches =
               (SELECT COUNT(*) FROM matches m WHERE m.season = s.year)""")
    execute("""INSERT INTO data_sources (name, url, version, notes) VALUES
               ('cricsheet', %s, %s, 'ball-by-ball IPL JSON, CC BY 4.0')""",
            (config.CRICSHEET_URL, config.DATA_CUTOFF))

    log.info("loaded %s matches, %s deliveries (skipped %s)",
             n_matches, n_deliveries, n_skipped)
    report = validation_report()
    log.info("validation report: %s", json.dumps(report, indent=2))
    bad = {k: v for k, v in report.items()
           if k.startswith(("negative", "orphan", "duplicate", "invalid")) and v}
    if bad:
        log.error("VALIDATION FAILURES: %s", bad)
        sys.exit(1)
    log.info("validation clean")


if __name__ == "__main__":
    main()
