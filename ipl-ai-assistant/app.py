"""
IPL AI Assistant — single-file implementation.

Text-to-text cricket assistant covering the Indian Premier League from the
first season (2008) through 2025-12-31. Combines:

  * SQLite          — exact statistics (runs, wickets, head-to-head, seasons)
  * Qdrant + BGE-M3 — semantic (RAG) retrieval over per-match summaries
  * Claude Opus 5   — agent that picks tools, never invents numbers
  * MCP (FastMCP)   — same tools exposed to any MCP client (Claude Desktop, etc.)
  * FastAPI         — HTTP chat endpoint for deployment

Run modes (see README.md for the full workflow):

    python app.py etl      # download Cricsheet zip, build SQLite
    python app.py index    # build match documents, embed, load into Qdrant
    python app.py all      # etl + index
    python app.py chat     # interactive terminal chat (needs ANTHROPIC_API_KEY)
    python app.py api      # FastAPI server on :8000
    python app.py mcp      # MCP server (stdio by default; MCP_TRANSPORT=streamable-http for Docker)

Environment (all optional, sane defaults):
    DATA_DIR, CRICSHEET_URL, CUTOFF_DATE, EMBED_MODEL, QDRANT_URL,
    QDRANT_COLLECTION, ANTHROPIC_MODEL, API_HOST, API_PORT, MCP_TRANSPORT
"""

import argparse
import json
import os
import re
import sqlite3
import sys
import zipfile
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def _load_dotenv() -> None:
    """Tiny dependency-free .env loader. Real env vars always win (setdefault)."""
    env_file = Path(__file__).resolve().parent / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv()

DATA_DIR = Path(os.environ.get("DATA_DIR", "./data"))
RAW_DIR = DATA_DIR / "raw" / "ipl_matches"
DB_PATH = DATA_DIR / "db" / "ipl.db"
CRICSHEET_URL = os.environ.get("CRICSHEET_URL", "https://cricsheet.org/downloads/ipl_json.zip")
CUTOFF_DATE = datetime.strptime(os.environ.get("CUTOFF_DATE", "2025-12-31"), "%Y-%m-%d")

EMBED_MODEL = os.environ.get("EMBED_MODEL", "BAAI/bge-m3")
QDRANT_URL = os.environ.get("QDRANT_URL", "")          # e.g. http://qdrant:6333; empty -> embedded local mode
QDRANT_LOCAL_PATH = str(DATA_DIR / "qdrant")
COLLECTION = os.environ.get("QDRANT_COLLECTION", "ipl_matches")

ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-opus-5")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-flash-latest")
# "gemini" | "claude"; empty -> auto-detect from which API key env var is set.
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "").lower()
# When set (e.g. http://localhost:8765/mcp) the agent calls tools over MCP
# instead of in-process — full Agent -> MCP server -> DB chain.
MCP_URL = os.environ.get("MCP_URL", "")
# RAG is optional here: exact stats never need it (SQL answers those); it only
# powers semantic match search. Set ENABLE_RAG=0 to run a pure SQL-tool agent.
ENABLE_RAG = os.environ.get("ENABLE_RAG", "1").lower() not in ("0", "false", "no")
API_HOST = os.environ.get("API_HOST", "0.0.0.0")
API_PORT = int(os.environ.get("API_PORT", "8000"))

# Dismissal kinds credited to the bowler (run outs etc. are not).
BOWLER_WICKETS = ("bowled", "caught", "caught and bowled", "lbw", "stumped", "hit wicket")

SYSTEM_PROMPT = f"""You are an expert IPL cricket assistant. Your knowledge covers the
Indian Premier League ONLY, from the first season (2008) through 31 December 2025.

Hard rules:
1. NEVER state a statistic (runs, wickets, averages, results) without fetching it
   from a tool first. If a tool returns nothing, say the data is unavailable.
2. Player names in the database use Cricsheet's initial form (e.g. "V Kohli",
   "MS Dhoni", "JJ Bumrah"). When the user gives a full name, call search_player
   with the surname first to resolve the canonical name, then use that exact
   string in other tools.
3. Every IPL season from 2008 to 2025 inclusive is IN scope — always answer these
   using tools (e.g. "IPL 2025 final" is answerable). Only refuse questions that are
   not about IPL cricket at all, or that ask about events after 31 December 2025
   ("I only cover IPL 2008 through the end of 2025").
4. Cite the season/match context for every number you report.
5. For qualitative questions ("close finishes at Chinnaswamy", "biggest chases")
   use {"find_matches (semantic search) and/or run_sql" if ENABLE_RAG else
        "match_results and run_sql"}. For "X vs Y bowler" duels
   use batter_vs_bowler; for filtered result lists use match_results; for "who is
   X" background use player_bio (only famous players have bios — if missing, say
   so and still give stats).
6. run_sql is read-only SELECT over the schema returned by get_schema. Prefer the
   purpose-built tools; drop to SQL only for questions they cannot answer.
"""


# ---------------------------------------------------------------------------
# Section 1 — ETL: Cricsheet JSON -> SQLite
# ---------------------------------------------------------------------------

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS matches (
    match_id        TEXT PRIMARY KEY,
    season          INTEGER,            -- normalized: year of the match date
    season_label    TEXT,               -- raw Cricsheet label, e.g. "2007/08"
    match_date      TEXT,
    venue           TEXT,
    city            TEXT,
    team1           TEXT,
    team2           TEXT,
    toss_winner     TEXT,
    toss_decision   TEXT,
    winner          TEXT,
    win_by_type     TEXT,               -- 'runs' | 'wickets'
    win_by_margin   INTEGER,
    result_type     TEXT,               -- 'normal' | 'tie' | 'no result'
    player_of_match TEXT,
    event_stage     TEXT                -- 'Final', 'Qualifier 1', ... when present
);

CREATE TABLE IF NOT EXISTS deliveries (
    match_id      TEXT,
    innings       INTEGER,
    over          INTEGER,
    ball          INTEGER,
    batting_team  TEXT,
    bowling_team  TEXT,
    batter        TEXT,
    bowler        TEXT,
    non_striker   TEXT,
    runs_batter   INTEGER,
    runs_extras   INTEGER,
    runs_total    INTEGER,
    wides         INTEGER DEFAULT 0,
    noballs       INTEGER DEFAULT 0,
    byes          INTEGER DEFAULT 0,
    legbyes       INTEGER DEFAULT 0,
    is_wicket     INTEGER DEFAULT 0,
    player_out    TEXT,
    dismissal_kind TEXT,
    FOREIGN KEY (match_id) REFERENCES matches(match_id)
);

CREATE TABLE IF NOT EXISTS players (
    player_name      TEXT PRIMARY KEY,
    teams_played_for TEXT
);

-- Supplementary hand-curated bios for the most-searched players.
-- Cricsheet carries no bio/award data, so this is a small static layer.
CREATE TABLE IF NOT EXISTS player_bios (
    player_name   TEXT PRIMARY KEY,   -- Cricsheet canonical name, e.g. "V Kohli"
    full_name     TEXT,
    country       TEXT,
    role          TEXT,               -- batter | bowler | all-rounder | wicketkeeper-batter
    batting_style TEXT,
    bowling_style TEXT,
    notes         TEXT
);

CREATE INDEX IF NOT EXISTS idx_del_match  ON deliveries(match_id);
CREATE INDEX IF NOT EXISTS idx_del_batter ON deliveries(batter);
CREATE INDEX IF NOT EXISTS idx_del_bowler ON deliveries(bowler);
CREATE INDEX IF NOT EXISTS idx_m_season   ON matches(season);

-- Aggregation views used by the tool layer.
CREATE VIEW IF NOT EXISTS batting_by_season AS
SELECT d.batter AS player, m.season AS season,
       COUNT(DISTINCT d.match_id)                          AS matches,
       SUM(d.runs_batter)                                  AS runs,
       SUM(CASE WHEN d.wides = 0 THEN 1 ELSE 0 END)        AS balls,
       SUM(CASE WHEN d.runs_batter = 4 THEN 1 ELSE 0 END)  AS fours,
       SUM(CASE WHEN d.runs_batter = 6 THEN 1 ELSE 0 END)  AS sixes
FROM deliveries d JOIN matches m ON m.match_id = d.match_id
GROUP BY d.batter, m.season;

CREATE VIEW IF NOT EXISTS dismissals_by_season AS
SELECT d.player_out AS player, m.season AS season, COUNT(*) AS outs
FROM deliveries d JOIN matches m ON m.match_id = d.match_id
WHERE d.player_out IS NOT NULL
GROUP BY d.player_out, m.season;

CREATE VIEW IF NOT EXISTS bowling_by_season AS
SELECT d.bowler AS player, m.season AS season,
       COUNT(DISTINCT d.match_id)                                      AS matches,
       SUM(CASE WHEN d.wides = 0 AND d.noballs = 0 THEN 1 ELSE 0 END)  AS balls,
       SUM(d.runs_total - d.byes - d.legbyes)                          AS runs_conceded,
       SUM(CASE WHEN d.dismissal_kind IN
           ('bowled','caught','caught and bowled','lbw','stumped','hit wicket')
           THEN 1 ELSE 0 END)                                          AS wickets
FROM deliveries d JOIN matches m ON m.match_id = d.match_id
GROUP BY d.bowler, m.season;
"""


# (full_name, country, role, batting_style, bowling_style, notes) keyed by
# Cricsheet canonical name. ~30 most-searched players only, by design.
PLAYER_BIOS = {
    "V Kohli": ("Virat Kohli", "India", "batter", "right-hand bat", None,
                "RCB one-club man; all-time leading IPL run scorer; 2016 record 973-run season; 2025 champion"),
    "RG Sharma": ("Rohit Sharma", "India", "batter", "right-hand bat", None,
                  "Captained Mumbai Indians to five IPL titles"),
    "MS Dhoni": ("Mahendra Singh Dhoni", "India", "wicketkeeper-batter", "right-hand bat", None,
                 "CSK talisman and captain for five IPL titles; famed finisher"),
    "JJ Bumrah": ("Jasprit Bumrah", "India", "bowler", "right-hand bat", "right-arm fast",
                  "MI death-overs specialist, unorthodox action"),
    "SK Raina": ("Suresh Raina", "India", "batter", "left-hand bat", "off-spin",
                 "'Mr IPL' — CSK middle-order mainstay"),
    "DA Warner": ("David Warner", "Australia", "batter", "left-hand bat", None,
                  "Three-time Orange Cap; led SRH to the 2016 title"),
    "CH Gayle": ("Chris Gayle", "West Indies", "batter", "left-hand bat", "off-spin",
                 "Holds highest IPL individual score, 175* for RCB in 2013"),
    "AB de Villiers": ("Abraham Benjamin de Villiers", "South Africa", "batter", "right-hand bat", None,
                       "'Mr 360' — RCB great, one of the most destructive IPL batters"),
    "R Ashwin": ("Ravichandran Ashwin", "India", "bowler", "right-hand bat", "off-spin",
                 "Canny powerplay off-spinner across CSK, PBKS, DC, RR"),
    "YS Chahal": ("Yuzvendra Chahal", "India", "bowler", "right-hand bat", "leg-spin",
                  "All-time leading IPL wicket-taker"),
    "RA Jadeja": ("Ravindra Jadeja", "India", "all-rounder", "left-hand bat", "left-arm orthodox",
                  "CSK all-rounder; hit the winning runs in the 2023 final"),
    "HH Pandya": ("Hardik Pandya", "India", "all-rounder", "right-hand bat", "right-arm medium-fast",
                  "Captained Gujarat Titans to the 2022 title in their debut season"),
    "KL Rahul": ("Kannaur Lokesh Rahul", "India", "wicketkeeper-batter", "right-hand bat", None,
                 "Orange Cap 2020; captained PBKS and LSG"),
    "S Dhawan": ("Shikhar Dhawan", "India", "batter", "left-hand bat", None,
                 "Second on the all-time IPL run list at retirement"),
    "RR Pant": ("Rishabh Pant", "India", "wicketkeeper-batter", "left-hand bat", None,
                "Delhi Capitals captain; record IPL auction price with LSG in 2025"),
    "SV Samson": ("Sanju Samson", "India", "wicketkeeper-batter", "right-hand bat", None,
                  "Rajasthan Royals captain"),
    "SA Yadav": ("Suryakumar Yadav", "India", "batter", "right-hand bat", None,
                 "MI 360-degree stroke-maker; IPL 2025 MVP"),
    "Shubman Gill": ("Shubman Gill", "India", "batter", "right-hand bat", None,
                     "Orange Cap 2023 with 890 runs; Gujarat Titans captain"),
    "B Sai Sudharsan": ("Bharathidasan Sai Sudharsan", "India", "batter", "left-hand bat", None,
                        "Orange Cap 2025 with 759 runs for Gujarat Titans"),
    "M Prasidh Krishna": ("Prasidh Krishna", "India", "bowler", "right-hand bat", "right-arm fast",
                          "Purple Cap 2025 with 25 wickets for Gujarat Titans"),
    "JC Buttler": ("Jos Buttler", "England", "wicketkeeper-batter", "right-hand bat", None,
                   "Orange Cap 2022 with four centuries for RR"),
    "SP Narine": ("Sunil Narine", "West Indies", "all-rounder", "left-hand bat", "mystery spin",
                  "KKR mystery spinner and pinch-hitting opener"),
    "AD Russell": ("Andre Russell", "West Indies", "all-rounder", "right-hand bat", "right-arm fast",
                   "KKR power hitter, among the highest strike rates in IPL history"),
    "Rashid Khan": ("Rashid Khan", "Afghanistan", "bowler", "right-hand bat", "leg-spin",
                    "Elite T20 leg-spinner for SRH and GT"),
    "F du Plessis": ("Faf du Plessis", "South Africa", "batter", "right-hand bat", None,
                     "CSK opener turned RCB captain"),
    "Q de Kock": ("Quinton de Kock", "South Africa", "wicketkeeper-batter", "left-hand bat", None,
                  "Opener for MI and LSG"),
    "SL Malinga": ("Lasith Malinga", "Sri Lanka", "bowler", "right-hand bat", "right-arm fast",
                   "Yorker specialist; long held the all-time IPL wickets record"),
    "DJ Bravo": ("Dwayne Bravo", "West Indies", "all-rounder", "right-hand bat", "right-arm medium-fast",
                 "Death-overs specialist, multiple Purple Caps with CSK"),
    "SR Tendulkar": ("Sachin Tendulkar", "India", "batter", "right-hand bat", None,
                     "MI icon; Orange Cap 2010"),
    "G Gambhir": ("Gautam Gambhir", "India", "batter", "left-hand bat", None,
                  "Captained KKR to the 2012 and 2014 titles"),
    "V Sehwag": ("Virender Sehwag", "India", "batter", "right-hand bat", None,
                 "Explosive opener for DD and PBKS"),
    "TA Boult": ("Trent Boult", "New Zealand", "bowler", "right-hand bat", "left-arm fast",
                 "New-ball strike bowler for MI, RR, MI again"),
    "Arshdeep Singh": ("Arshdeep Singh", "India", "bowler", "left-hand bat", "left-arm fast-medium",
                       "PBKS left-arm quick; 2025 runners-up campaign"),
}


def _seed_bios(conn: sqlite3.Connection) -> int:
    conn.executemany(
        "INSERT OR REPLACE INTO player_bios VALUES (?, ?, ?, ?, ?, ?, ?)",
        [(name, *bio) for name, bio in PLAYER_BIOS.items()],
    )
    return len(PLAYER_BIOS)


def download_data() -> None:
    import requests

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = DATA_DIR / "raw" / "ipl_json.zip"
    if zip_path.exists():
        print(f"[etl] {zip_path} already present, skipping download")
    else:
        print(f"[etl] downloading {CRICSHEET_URL} ...")
        resp = requests.get(CRICSHEET_URL, timeout=120)
        resp.raise_for_status()
        zip_path.write_bytes(resp.content)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(RAW_DIR)
    print(f"[etl] extracted {len(list(RAW_DIR.glob('*.json')))} match files")


def _parse_match(fp: Path):
    data = json.loads(fp.read_text(encoding="utf-8"))
    info = data.get("info", {})
    dates = info.get("dates") or []
    if not dates:
        return None
    try:
        dt = datetime.strptime(dates[0], "%Y-%m-%d")
    except ValueError:
        return None
    if dt > CUTOFF_DATE:
        return None

    teams = info.get("teams", [None, None])
    outcome = info.get("outcome", {})
    by = outcome.get("by", {})
    toss = info.get("toss", {})
    pom = (info.get("player_of_match") or [None])[0]
    result_type = outcome.get("result", "tie" if "eliminator" in outcome else "normal")

    match_row = (
        fp.stem, dt.year, str(info.get("season", "")), dates[0],
        info.get("venue"), info.get("city"),
        teams[0], teams[1] if len(teams) > 1 else None,
        toss.get("winner"), toss.get("decision"),
        outcome.get("winner"),
        "runs" if "runs" in by else ("wickets" if "wickets" in by else None),
        by.get("runs") or by.get("wickets"),
        result_type, pom,
        (info.get("event") or {}).get("stage"),
    )

    delivery_rows, player_teams = [], {}
    for inn_idx, inning in enumerate(data.get("innings", []), start=1):
        batting = inning.get("team")
        bowling = teams[0] if batting == teams[1] else (teams[1] if len(teams) > 1 else None)
        for over in inning.get("overs", []):
            for ball_idx, d in enumerate(over.get("deliveries", []), start=1):
                runs = d.get("runs", {})
                extras = d.get("extras", {})
                wickets = d.get("wickets", [])
                delivery_rows.append((
                    fp.stem, inn_idx, over.get("over"), ball_idx, batting, bowling,
                    d.get("batter"), d.get("bowler"), d.get("non_striker"),
                    runs.get("batter", 0), runs.get("extras", 0), runs.get("total", 0),
                    extras.get("wides", 0), extras.get("noballs", 0),
                    extras.get("byes", 0), extras.get("legbyes", 0),
                    1 if wickets else 0,
                    wickets[0].get("player_out") if wickets else None,
                    wickets[0].get("kind") if wickets else None,
                ))
                for p, team in ((d.get("batter"), batting), (d.get("non_striker"), batting),
                                (d.get("bowler"), bowling)):
                    if p and team:
                        player_teams.setdefault(p, set()).add(team)
    return match_row, delivery_rows, player_teams


def run_etl() -> None:
    download_data()
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.executescript(SCHEMA_SQL)
    cur.execute("DELETE FROM deliveries")
    cur.execute("DELETE FROM matches")
    cur.execute("DELETE FROM players")

    all_player_teams: dict[str, set] = {}
    n_matches = n_deliveries = n_skipped = 0
    for fp in sorted(RAW_DIR.glob("*.json")):
        parsed = _parse_match(fp)
        if parsed is None:
            n_skipped += 1
            continue
        match_row, delivery_rows, player_teams = parsed
        cur.execute(f"INSERT OR REPLACE INTO matches VALUES ({','.join('?' * 16)})", match_row)
        cur.executemany(f"INSERT INTO deliveries VALUES ({','.join('?' * 19)})", delivery_rows)
        for p, teams in player_teams.items():
            all_player_teams.setdefault(p, set()).update(teams)
        n_matches += 1
        n_deliveries += len(delivery_rows)

    cur.executemany(
        "INSERT OR REPLACE INTO players VALUES (?, ?)",
        [(p, ",".join(sorted(t))) for p, t in all_player_teams.items()],
    )
    n_bios = _seed_bios(conn)
    conn.commit()
    conn.close()
    print(f"[etl] loaded {n_matches} matches, {n_deliveries} deliveries, "
          f"{len(all_player_teams)} players, {n_bios} bios "
          f"(skipped {n_skipped} post-cutoff/malformed)")


# ---------------------------------------------------------------------------
# Section 2 — RAG: match documents -> BGE-M3 embeddings -> Qdrant
# ---------------------------------------------------------------------------

def _db() -> sqlite3.Connection:
    if not DB_PATH.exists():
        raise RuntimeError(f"database not found at {DB_PATH} — run `python app.py etl` first")
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _qdrant():
    from qdrant_client import QdrantClient

    if QDRANT_URL:
        return QdrantClient(url=QDRANT_URL, timeout=30)
    Path(QDRANT_LOCAL_PATH).mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=QDRANT_LOCAL_PATH)


_embedder = None


def _embed(texts: list[str]):
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        print(f"[embed] loading {EMBED_MODEL} ...", file=sys.stderr)
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder.encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False)


def build_match_documents() -> list[dict]:
    """One retrieval document per match: result line + standout performances."""
    conn = _db()
    docs = []
    for m in conn.execute("SELECT * FROM matches ORDER BY match_date"):
        top_bat = conn.execute(
            "SELECT batter, SUM(runs_batter) r FROM deliveries WHERE match_id=? "
            "GROUP BY batter ORDER BY r DESC LIMIT 1", (m["match_id"],)).fetchone()
        top_bowl = conn.execute(
            "SELECT bowler, SUM(CASE WHEN dismissal_kind IN "
            "('bowled','caught','caught and bowled','lbw','stumped','hit wicket') "
            "THEN 1 ELSE 0 END) w, SUM(runs_total - byes - legbyes) rc "
            "FROM deliveries WHERE match_id=? GROUP BY bowler ORDER BY w DESC, rc ASC LIMIT 1",
            (m["match_id"],)).fetchone()

        result = (f"{m['winner']} won by {m['win_by_margin']} {m['win_by_type']}"
                  if m["winner"] else f"result: {m['result_type']}")
        stage = f" ({m['event_stage']})" if m["event_stage"] else ""
        text = (
            f"IPL {m['season']}{stage}, {m['match_date']}: {m['team1']} vs {m['team2']} "
            f"at {m['venue']}, {m['city']}. Toss: {m['toss_winner']} chose to {m['toss_decision']}. "
            f"{result}. Player of the match: {m['player_of_match']}. "
            f"Top scorer: {top_bat['batter']} ({top_bat['r']} runs). "
            f"Best bowler: {top_bowl['bowler']} ({top_bowl['w']} wickets for {top_bowl['rc']} runs)."
        )
        docs.append({
            "text": text,
            "match_id": m["match_id"],
            "season": m["season"],
            "date": m["match_date"],
            "teams": f"{m['team1']} vs {m['team2']}",
            "venue": m["venue"],
        })
    conn.close()
    return docs


def run_index() -> None:
    from qdrant_client.models import Distance, PointStruct, VectorParams

    docs = build_match_documents()
    print(f"[index] embedding {len(docs)} match documents with {EMBED_MODEL}")
    vectors = _embed([d["text"] for d in docs])

    client = _qdrant()
    dim = len(vectors[0])
    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
    client.create_collection(COLLECTION, vectors_config=VectorParams(size=dim, distance=Distance.COSINE))
    client.upsert(COLLECTION, points=[
        PointStruct(id=i, vector=vectors[i].tolist(), payload=docs[i]) for i in range(len(docs))
    ])
    print(f"[index] loaded {len(docs)} vectors (dim={dim}) into collection '{COLLECTION}'")


# ---------------------------------------------------------------------------
# Section 3 — Tool layer (shared by the agent and the MCP server)
# ---------------------------------------------------------------------------

_SQL_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|attach|pragma|vacuum|replace|reindex)\b", re.I)


def get_schema() -> str:
    """Tables/views + columns for run_sql."""
    conn = _db()
    rows = conn.execute(
        "SELECT name, sql FROM sqlite_master WHERE type IN ('table','view') AND sql IS NOT NULL").fetchall()
    conn.close()
    return "\n\n".join(r["sql"] for r in rows)


def run_sql(sql: str, limit: int = 50) -> str:
    """Run one read-only SELECT/WITH statement against the IPL database.
    Call get_schema first to see available tables and views.

    Args:
        sql: A single SELECT or WITH statement (no writes, no PRAGMA).
        limit: Row cap on the result (default 50).
    """
    stripped = re.sub(r"--.*?$|/\*.*?\*/", "", sql, flags=re.S | re.M).strip().rstrip(";")
    if ";" in stripped or _SQL_FORBIDDEN.search(stripped) or \
            not re.match(r"^\s*(select|with)\b", stripped, re.I):
        return json.dumps({"error": "only a single read-only SELECT/WITH statement is allowed"})
    conn = _db()
    try:
        rows = conn.execute(f"SELECT * FROM ({stripped}) LIMIT {int(limit)}").fetchall()
        return json.dumps([dict(r) for r in rows], default=str)
    except sqlite3.Error as e:
        return json.dumps({"error": str(e)})
    finally:
        conn.close()


def search_player(name_fragment: str) -> str:
    """Resolve a player's canonical database name from a partial name or surname.

    Args:
        name_fragment: Part of the name, e.g. "Kohli" or "Bumrah".
    """
    conn = _db()
    rows = conn.execute(
        "SELECT player_name, teams_played_for FROM players "
        "WHERE player_name LIKE '%' || ? || '%' LIMIT 15", (name_fragment,)).fetchall()
    conn.close()
    return json.dumps([dict(r) for r in rows])


def player_batting(player: str, season: int | None = None) -> str:
    """IPL batting stats (career + per-season) for an exact canonical player name.

    Args:
        player: Exact name as stored, e.g. "V Kohli". Resolve via search_player first.
        season: Optional year (2008-2025) to filter to one season.
    """
    conn = _db()
    where, params = "b.player = ?", [player]
    if season:
        where += " AND b.season = ?"
        params.append(season)
    rows = conn.execute(f"""
        SELECT b.season, b.matches, b.runs, b.balls, b.fours, b.sixes,
               COALESCE(o.outs, 0) AS dismissals,
               ROUND(100.0 * b.runs / NULLIF(b.balls, 0), 2) AS strike_rate,
               ROUND(1.0 * b.runs / NULLIF(o.outs, 0), 2)   AS average
        FROM batting_by_season b
        LEFT JOIN dismissals_by_season o ON o.player = b.player AND o.season = b.season
        WHERE {where} ORDER BY b.season""", params).fetchall()
    conn.close()
    if not rows:
        return json.dumps({"error": f"no batting data for '{player}' — resolve the name via search_player"})
    data = [dict(r) for r in rows]
    career = {
        "runs": sum(r["runs"] for r in data),
        "matches": sum(r["matches"] for r in data),
        "dismissals": sum(r["dismissals"] for r in data),
        "fours": sum(r["fours"] for r in data),
        "sixes": sum(r["sixes"] for r in data),
    }
    balls = sum(r["balls"] for r in data)
    career["strike_rate"] = round(100.0 * career["runs"] / balls, 2) if balls else None
    career["average"] = round(career["runs"] / career["dismissals"], 2) if career["dismissals"] else None
    return json.dumps({"player": player, "career": career, "by_season": data})


def player_bowling(player: str, season: int | None = None) -> str:
    """IPL bowling stats (career + per-season) for an exact canonical player name.

    Args:
        player: Exact name as stored, e.g. "JJ Bumrah". Resolve via search_player first.
        season: Optional year (2008-2025) to filter to one season.
    """
    conn = _db()
    where, params = "player = ?", [player]
    if season:
        where += " AND season = ?"
        params.append(season)
    rows = conn.execute(f"""
        SELECT season, matches, balls, runs_conceded, wickets,
               ROUND(6.0 * runs_conceded / NULLIF(balls, 0), 2) AS economy
        FROM bowling_by_season WHERE {where} ORDER BY season""", params).fetchall()
    conn.close()
    if not rows:
        return json.dumps({"error": f"no bowling data for '{player}' — resolve the name via search_player"})
    data = [dict(r) for r in rows]
    balls = sum(r["balls"] for r in data)
    career = {
        "wickets": sum(r["wickets"] for r in data),
        "matches": sum(r["matches"] for r in data),
        "runs_conceded": sum(r["runs_conceded"] for r in data),
        "economy": round(6.0 * sum(r["runs_conceded"] for r in data) / balls, 2) if balls else None,
    }
    return json.dumps({"player": player, "career": career, "by_season": data})


def head_to_head(team_a: str, team_b: str) -> str:
    """Win counts between two IPL teams across all seasons.

    Args:
        team_a: First team name or fragment, e.g. "Mumbai Indians".
        team_b: Second team name or fragment, e.g. "Chennai Super Kings".
    """
    conn = _db()
    rows = conn.execute("""
        SELECT winner, COUNT(*) AS wins FROM matches
        WHERE (team1 LIKE '%'||?||'%' AND team2 LIKE '%'||?||'%')
           OR (team1 LIKE '%'||?||'%' AND team2 LIKE '%'||?||'%')
        GROUP BY winner ORDER BY wins DESC""",
        (team_a, team_b, team_b, team_a)).fetchall()
    conn.close()
    if not rows:
        return json.dumps({"error": f"no matches found between '{team_a}' and '{team_b}'"})
    return json.dumps({"head_to_head": [dict(r) for r in rows],
                       "note": "NULL winner = tie / no result"})


def top_run_scorers(season: int | None = None, limit: int = 10) -> str:
    """Leading run scorers — one season (Orange Cap race) or all-time when season omitted.

    Args:
        season: Optional year (2008-2025).
        limit: Number of players to return (default 10).
    """
    conn = _db()
    where = f"WHERE season = {int(season)}" if season else ""
    rows = conn.execute(f"""
        SELECT player, SUM(runs) AS runs, SUM(matches) AS matches,
               ROUND(100.0 * SUM(runs) / NULLIF(SUM(balls), 0), 2) AS strike_rate
        FROM batting_by_season {where}
        GROUP BY player ORDER BY runs DESC LIMIT {int(limit)}""").fetchall()
    conn.close()
    return json.dumps([dict(r) for r in rows])


def top_wicket_takers(season: int | None = None, limit: int = 10) -> str:
    """Leading wicket takers — one season (Purple Cap race) or all-time when season omitted.

    Args:
        season: Optional year (2008-2025).
        limit: Number of players to return (default 10).
    """
    conn = _db()
    where = f"WHERE season = {int(season)}" if season else ""
    rows = conn.execute(f"""
        SELECT player, SUM(wickets) AS wickets, SUM(matches) AS matches,
               ROUND(6.0 * SUM(runs_conceded) / NULLIF(SUM(balls), 0), 2) AS economy
        FROM bowling_by_season {where}
        GROUP BY player ORDER BY wickets DESC LIMIT {int(limit)}""").fetchall()
    conn.close()
    return json.dumps([dict(r) for r in rows])


def season_summary(season: int) -> str:
    """Season overview: final result, champion, most runs and most wickets.

    Args:
        season: Year, 2008-2025.
    """
    conn = _db()
    final = conn.execute(
        "SELECT * FROM matches WHERE season=? ORDER BY match_date DESC LIMIT 1", (season,)).fetchone()
    n = conn.execute("SELECT COUNT(*) c FROM matches WHERE season=?", (season,)).fetchone()["c"]
    conn.close()
    if not final:
        return json.dumps({"error": f"no data for IPL {season}"})
    orange = json.loads(top_run_scorers(season, 1))
    purple = json.loads(top_wicket_takers(season, 1))
    return json.dumps({
        "season": season, "matches_played": n,
        "final": {k: final[k] for k in
                  ("match_date", "venue", "team1", "team2", "winner",
                   "win_by_type", "win_by_margin", "player_of_match")},
        "orange_cap_most_runs": orange[0] if orange else None,
        "purple_cap_most_wickets": purple[0] if purple else None,
        "note": "leaders computed from ball-by-ball data; final = last match of season",
    })


def find_matches(query: str, top_k: int = 5) -> str:
    """Semantic search over per-match summary documents (Qdrant)."""
    vec = _embed([query])[0].tolist()
    client = _qdrant()
    res = client.query_points(COLLECTION, query=vec, limit=int(top_k))
    return json.dumps([
        {"score": round(p.score, 4), **p.payload} for p in res.points
    ])


def player_bio(player: str) -> str:
    """Bio for a well-known IPL player: full name, country, role, styles, notes.
    Only the ~30 most-searched players have bios; stats tools cover everyone.

    Args:
        player: Exact name as stored, e.g. "V Kohli". Resolve via search_player first.
    """
    conn = _db()
    row = conn.execute("SELECT * FROM player_bios WHERE player_name = ?", (player,)).fetchone()
    conn.close()
    if not row:
        return json.dumps({"error": f"no bio on file for '{player}' — bios exist only for "
                                    "the most famous players; career stats are still available "
                                    "via player_batting / player_bowling"})
    return json.dumps(dict(row))


def batter_vs_bowler(batter: str, bowler: str) -> str:
    """Ball-by-ball matchup between one batter and one bowler across all IPL
    seasons: balls faced, runs, dismissals, strike rate.

    Args:
        batter: Exact batter name, e.g. "V Kohli". Resolve via search_player first.
        bowler: Exact bowler name, e.g. "JJ Bumrah". Resolve via search_player first.
    """
    conn = _db()
    row = conn.execute("""
        SELECT COUNT(*)                                            AS balls_total,
               SUM(CASE WHEN wides = 0 THEN 1 ELSE 0 END)          AS balls_faced,
               SUM(runs_batter)                                    AS runs,
               SUM(CASE WHEN runs_batter = 4 THEN 1 ELSE 0 END)    AS fours,
               SUM(CASE WHEN runs_batter = 6 THEN 1 ELSE 0 END)    AS sixes,
               SUM(CASE WHEN player_out = batter AND dismissal_kind IN
                   ('bowled','caught','caught and bowled','lbw','stumped','hit wicket')
                   THEN 1 ELSE 0 END)                              AS dismissals
        FROM deliveries WHERE batter = ? AND bowler = ?""", (batter, bowler)).fetchone()
    conn.close()
    if not row or not row["balls_total"]:
        return json.dumps({"error": f"no deliveries found for '{batter}' vs '{bowler}' — "
                                    "check both names via search_player"})
    d = dict(row)
    d["strike_rate"] = round(100.0 * d["runs"] / d["balls_faced"], 2) if d["balls_faced"] else None
    return json.dumps({"batter": batter, "bowler": bowler, **d})


def match_results(season: int | None = None, team: str | None = None,
                  venue: str | None = None, limit: int = 20) -> str:
    """List match results filtered by any combination of season, team, venue.

    Args:
        season: Optional year (2008-2025).
        team: Optional team name or fragment, e.g. "Chennai".
        venue: Optional venue/city fragment, e.g. "Wankhede" or "Chinnaswamy".
        limit: Max matches returned, newest first (default 20).
    """
    where, params = [], []
    if season:
        where.append("season = ?")
        params.append(season)
    if team:
        where.append("(team1 LIKE '%'||?||'%' OR team2 LIKE '%'||?||'%')")
        params += [team, team]
    if venue:
        where.append("(venue LIKE '%'||?||'%' OR city LIKE '%'||?||'%')")
        params += [venue, venue]
    clause = ("WHERE " + " AND ".join(where)) if where else ""
    conn = _db()
    rows = conn.execute(f"""
        SELECT match_date, season, event_stage, team1, team2, venue, city,
               winner, win_by_type, win_by_margin, player_of_match
        FROM matches {clause} ORDER BY match_date DESC LIMIT {int(limit)}""",
        params).fetchall()
    conn.close()
    if not rows:
        return json.dumps({"error": "no matches found for those filters"})
    return json.dumps([dict(r) for r in rows])


# Single registry consumed by all three surfaces (Claude agent, Gemini agent,
# MCP server) so the tool names the LLM sees always match the system prompt.
TOOL_FUNCS = [search_player, player_batting, player_bowling, player_bio,
              batter_vs_bowler, head_to_head, match_results,
              top_run_scorers, top_wicket_takers, season_summary,
              get_schema, run_sql]
if ENABLE_RAG:
    TOOL_FUNCS.insert(-2, find_matches)


# ---------------------------------------------------------------------------
# Section 4 — Agent: Gemini (free tier) or Claude Opus 5, switchable
# ---------------------------------------------------------------------------

def _agent_tools():
    """Anthropic beta tools — schemas auto-generated from the shared registry's
    signatures and docstrings."""
    from anthropic import beta_tool

    return [beta_tool(f) for f in TOOL_FUNCS]


def _answer_gemini(question: str, history: list[dict] | None = None) -> str:
    """One agent turn on Gemini. The google-genai SDK's automatic function
    calling runs the tool loop for us — it builds schemas from the tool
    functions' type hints and docstrings."""
    from google import genai
    from google.genai import types

    client = genai.Client()  # reads GEMINI_API_KEY (or GOOGLE_API_KEY) from env
    gem_history = [
        types.Content(role="user" if m["role"] == "user" else "model",
                      parts=[types.Part(text=m["content"])])
        for m in (history or [])
    ]
    chat = client.chats.create(
        model=GEMINI_MODEL,
        history=gem_history,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.2,
            tools=TOOL_FUNCS,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                maximum_remote_calls=20),
        ),
    )
    # 429 = free-tier rate limit (a few requests/min; one agent turn = several
    # requests). 503 = Google-side overload. Both are transient — back off, retry.
    import time

    from google.genai import errors

    for attempt in range(4):
        try:
            resp = chat.send_message(question)
            return resp.text or "Sorry — I could not produce an answer."
        except (errors.ClientError, errors.ServerError) as e:
            if e.code in (429, 503) and attempt < 3:
                wait = 20 * (attempt + 1)
                reason = "rate limit" if e.code == 429 else "model overloaded"
                print(f"[gemini] {reason}, retrying in {wait}s ...", file=sys.stderr)
                time.sleep(wait)
                continue
            raise


def _answer_gemini_via_mcp(question: str, history: list[dict] | None = None) -> str:
    """Full Agent -> MCP -> LLM chain: the agent connects to the MCP server as a
    real MCP client (streamable-http), discovers its tools over the protocol,
    and Gemini's automatic function calling invokes them remotely."""
    import asyncio
    import time

    async def _run() -> str:
        from google import genai
        from google.genai import types
        from mcp import ClientSession
        try:  # MCP SDK >= 2.0
            from mcp.client.streamable_http import streamable_http_client
        except ImportError:  # MCP SDK 1.x name
            from mcp.client.streamable_http import (
                streamablehttp_client as streamable_http_client)

        # SDK 2.x yields (read, write); 1.x yielded (read, write, session_id)
        async with streamable_http_client(MCP_URL) as streams:
            read, write = streams[0], streams[1]
            async with ClientSession(read, write) as session:
                await session.initialize()

                # 1. Discover tools over the protocol and hand Gemini their schemas.
                listed = await session.list_tools()
                declarations = types.Tool(function_declarations=[
                    types.FunctionDeclaration(
                        name=t.name,
                        description=t.description or "",
                        parameters_json_schema=t.inputSchema,
                    ) for t in listed.tools
                ])

                client = genai.Client()
                contents = [
                    types.Content(role="user" if m["role"] == "user" else "model",
                                  parts=[types.Part(text=m["content"])])
                    for m in (history or [])
                ] + [types.Content(role="user", parts=[types.Part(text=question)])]

                # 2. Manual function-calling loop: every tool call the model
                #    makes is executed remotely via MCP session.call_tool().
                for _ in range(20):
                    resp = await client.aio.models.generate_content(
                        model=GEMINI_MODEL,
                        contents=contents,
                        config=types.GenerateContentConfig(
                            system_instruction=SYSTEM_PROMPT,
                            temperature=0.2,
                            tools=[declarations],
                        ),
                    )
                    calls = resp.function_calls or []
                    if not calls:
                        return resp.text or "Sorry — I could not produce an answer."
                    contents.append(resp.candidates[0].content)
                    result_parts = []
                    for fc in calls:
                        mcp_result = await session.call_tool(fc.name, dict(fc.args or {}))
                        text = "".join(
                            c.text for c in mcp_result.content if getattr(c, "text", None))
                        result_parts.append(types.Part.from_function_response(
                            name=fc.name, response={"result": text}))
                    contents.append(types.Content(role="user", parts=result_parts))
                return "Sorry — tool-call limit reached without a final answer."

    from google.genai import errors

    for attempt in range(4):
        try:
            return asyncio.run(_run())
        except (errors.ClientError, errors.ServerError) as e:
            if e.code in (429, 503) and attempt < 3:
                wait = 20 * (attempt + 1)
                reason = "rate limit" if e.code == 429 else "model overloaded"
                print(f"[gemini] {reason}, retrying in {wait}s ...", file=sys.stderr)
                time.sleep(wait)
                continue
            raise
    return "Sorry — Gemini is busy right now, please try again in a minute."


def _answer_claude(question: str, history: list[dict] | None = None) -> str:
    """One agent turn on Claude: tool-use loop via the SDK tool runner."""
    import anthropic

    client = anthropic.Anthropic()
    messages = list(history or []) + [{"role": "user", "content": question}]

    runner = client.beta.messages.tool_runner(
        model=ANTHROPIC_MODEL,
        max_tokens=8192,
        system=[{"type": "text", "text": SYSTEM_PROMPT,
                 "cache_control": {"type": "ephemeral"}}],
        tools=_agent_tools(),
        messages=messages,
        # Server-side safety fallback: if Opus 5's classifiers ever decline a
        # request, the API transparently retries it on the recommended model.
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )
    last = None
    for message in runner:
        last = message
    if last is None:
        return "Sorry — the model returned no response. Please try again."
    if last.stop_reason == "refusal":
        return "Sorry — I can't help with that request."
    return "".join(b.text for b in last.content if b.type == "text") or \
        "Sorry — I could not produce an answer."


def answer(question: str, history: list[dict] | None = None) -> str:
    """Route to the configured LLM provider. Explicit LLM_PROVIDER wins;
    otherwise auto-detect from which API key is present (Gemini first — free tier)."""
    provider = LLM_PROVIDER or (
        "gemini" if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        else "claude"
    )
    if provider == "gemini":
        if MCP_URL:
            return _answer_gemini_via_mcp(question, history)
        return _answer_gemini(question, history)
    return _answer_claude(question, history)


# Golden QA set: facts verified against the database. Eval checks the agent's
# answer contains every expected fragment — grounding + regression test in one.
GOLDEN_QA = [
    ("Who won the IPL 2016 final?", ["sunrisers"]),
    ("Which team won IPL 2008, the first season?", ["rajasthan"]),
    ("Who won the Orange Cap in 2016 and with how many runs?", ["kohli", "973"]),
    ("Who took the most wickets in IPL 2025?", ["prasidh"]),
    ("What is Virat Kohli's total IPL career run count?", ["8671"]),
    ("Who is the all-time leading IPL run scorer?", ["kohli"]),
]


def run_eval() -> None:
    """Correctness + latency benchmark over the golden QA set."""
    import time

    results = []
    print(f"provider/model: {'gemini/' + GEMINI_MODEL if os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY') else 'claude/' + ANTHROPIC_MODEL}"
          f" | via MCP: {'yes' if MCP_URL else 'no'} | RAG: {'on' if ENABLE_RAG else 'off'}\n")
    for question, expected in GOLDEN_QA:
        t0 = time.perf_counter()
        try:
            text = answer(question)
        except Exception as e:
            text = f"ERROR: {e}"
        latency = time.perf_counter() - t0
        normalized = text.lower().replace(",", "")
        passed = all(frag in normalized for frag in expected)
        results.append((passed, latency))
        print(f"{'PASS' if passed else 'FAIL':4} {latency:6.1f}s  {question}")
        if not passed:
            print(f"     expected fragments {expected}; got: {text[:160]}")
    n_pass = sum(1 for p, _ in results if p)
    lats = sorted(latency for _, latency in results)
    print(f"\naccuracy: {n_pass}/{len(results)} "
          f"| latency avg {sum(lats)/len(lats):.1f}s "
          f"| min {lats[0]:.1f}s | max {lats[-1]:.1f}s")


def run_chat() -> None:
    print("IPL AI Assistant — 2008 to Dec 2025. Type 'exit' to quit.\n")
    history: list[dict] = []
    while True:
        try:
            q = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q or q.lower() in ("exit", "quit"):
            break
        text = answer(q, history)
        print(f"\nassistant> {text}\n")
        # Text-only history keeps multi-turn context without replaying tool traffic.
        history += [{"role": "user", "content": q}, {"role": "assistant", "content": text}]
        history = history[-12:]


# ---------------------------------------------------------------------------
# Section 5 — MCP server (FastMCP): same tools for any MCP client
# ---------------------------------------------------------------------------

def run_mcp() -> None:
    transport = os.environ.get("MCP_TRANSPORT", "stdio")
    host, port = "0.0.0.0", int(os.environ.get("MCP_PORT", "8765"))

    try:  # MCP SDK >= 2.0
        from mcp.server.mcpserver import MCPServer
        server = MCPServer("ipl-cricket-assistant")
        run_kwargs = {} if transport == "stdio" else {"host": host, "port": port}
    except ImportError:  # MCP SDK 1.x (FastMCP takes host/port at construction)
        from mcp.server.fastmcp import FastMCP
        server = FastMCP("ipl-cricket-assistant", host=host, port=port)
        run_kwargs = {}

    for f in TOOL_FUNCS:
        server.tool()(f)

    @server.resource("ipl://schema")
    def schema() -> str:
        """SQLite schema for the IPL database."""
        return get_schema()

    print(f"[mcp] serving via {transport}", file=sys.stderr)
    server.run(transport=transport, **run_kwargs)


# ---------------------------------------------------------------------------
# Section 6 — FastAPI HTTP interface
# ---------------------------------------------------------------------------

def run_api() -> None:
    import time

    import uvicorn
    from fastapi import FastAPI
    from pydantic import BaseModel

    app = FastAPI(title="IPL AI Assistant", version="1.0.0")
    latencies: list[float] = []  # rolling per-request latency, powers /metrics

    class ChatRequest(BaseModel):
        question: str
        history: list[dict] = []

    @app.get("/health")
    def health():
        ok = DB_PATH.exists()
        provider = LLM_PROVIDER or (
            "gemini" if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            else "claude")
        model = GEMINI_MODEL if provider == "gemini" else ANTHROPIC_MODEL
        return {"status": "ok" if ok else "db_missing", "provider": provider, "model": model}

    @app.post("/chat")
    def chat(req: ChatRequest):
        t0 = time.perf_counter()
        text = answer(req.question, req.history)
        latency_ms = round((time.perf_counter() - t0) * 1000)
        latencies.append(latency_ms)
        del latencies[:-500]  # keep the last 500 samples
        return {"answer": text, "latency_ms": latency_ms}

    @app.get("/metrics")
    def metrics():
        if not latencies:
            return {"requests": 0}
        s = sorted(latencies)
        return {
            "requests": len(s),
            "latency_ms_avg": round(sum(s) / len(s)),
            "latency_ms_p50": s[len(s) // 2],
            "latency_ms_p95": s[int(len(s) * 0.95) - 1] if len(s) >= 2 else s[-1],
            "latency_ms_max": s[-1],
        }

    uvicorn.run(app, host=API_HOST, port=API_PORT)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    global MCP_URL
    parser = argparse.ArgumentParser(description="IPL AI Assistant (2008 - Dec 2025)")
    parser.add_argument("mode",
                        choices=["etl", "index", "all", "chat", "chat-mcp",
                                 "api", "mcp", "eval"])
    mode = parser.parse_args().mode

    if mode == "chat-mcp":
        # Agent as MCP client: needs `python app.py mcp` running in another
        # terminal (MCP_TRANSPORT=streamable-http).
        MCP_URL = MCP_URL or "http://localhost:8765/mcp"
        mode = "chat"

    if mode == "etl":
        run_etl()
    elif mode == "index":
        run_index()
    elif mode == "all":
        run_etl()
        run_index()
    elif mode == "chat":
        run_chat()
    elif mode == "eval":
        run_eval()
    elif mode == "api":
        run_api()
    elif mode == "mcp":
        run_mcp()


if __name__ == "__main__":
    main()
