-- IPL Intelligence Agent — PostgreSQL schema
-- Data cutoff: 2025-12-31 (enforced at ingestion).
-- Tables marked DATA UNAVAILABLE exist per spec but hold no rows: no free,
-- licensed source provides that data (spec rule 29: never fabricate).

CREATE EXTENSION IF NOT EXISTS vector;

-- ---------------------------------------------------------------- identity
CREATE TABLE IF NOT EXISTS teams (
    id         SERIAL PRIMARY KEY,
    name       TEXT UNIQUE NOT NULL,
    short_name TEXT,
    city       TEXT
);

CREATE TABLE IF NOT EXISTS players (
    id             SERIAL PRIMARY KEY,
    cricsheet_name TEXT UNIQUE NOT NULL,   -- canonical id across seasons
    full_name      TEXT,
    nationality    TEXT,
    role           TEXT,
    batting_style  TEXT,
    bowling_style  TEXT
);

-- alias -> canonical player (identity resolution)
CREATE TABLE IF NOT EXISTS player_aliases (
    alias     TEXT PRIMARY KEY,
    player_id INT NOT NULL REFERENCES players(id)
);

CREATE TABLE IF NOT EXISTS venues (
    id   SERIAL PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    city TEXT
);

CREATE TABLE IF NOT EXISTS seasons (
    year             INT PRIMARY KEY,
    champion_team_id INT REFERENCES teams(id),
    runner_up_id     INT REFERENCES teams(id),
    matches          INT
);

-- ---------------------------------------------------------------- matches
CREATE TABLE IF NOT EXISTS matches (
    id                 TEXT PRIMARY KEY,          -- cricsheet match id
    season             INT NOT NULL REFERENCES seasons(year),
    match_date         DATE NOT NULL,
    venue_id           INT REFERENCES venues(id),
    team1_id           INT REFERENCES teams(id),
    team2_id           INT REFERENCES teams(id),
    toss_winner_id     INT REFERENCES teams(id),
    toss_decision      TEXT,
    winner_id          INT REFERENCES teams(id),
    win_by_type        TEXT,                      -- 'runs' | 'wickets'
    win_by_margin      INT,
    result_type        TEXT,
    player_of_match_id INT REFERENCES players(id),
    event_stage        TEXT
);

CREATE TABLE IF NOT EXISTS deliveries (
    id              BIGSERIAL PRIMARY KEY,
    match_id        TEXT NOT NULL REFERENCES matches(id),
    innings         SMALLINT NOT NULL,
    over_no         SMALLINT NOT NULL,
    ball_no         SMALLINT NOT NULL,
    batting_team_id INT REFERENCES teams(id),
    bowling_team_id INT REFERENCES teams(id),
    batter_id       INT NOT NULL REFERENCES players(id),
    bowler_id       INT NOT NULL REFERENCES players(id),
    non_striker_id  INT REFERENCES players(id),
    runs_batter     SMALLINT NOT NULL DEFAULT 0,
    runs_extras     SMALLINT NOT NULL DEFAULT 0,
    runs_total      SMALLINT NOT NULL DEFAULT 0,
    wides           SMALLINT NOT NULL DEFAULT 0,
    noballs         SMALLINT NOT NULL DEFAULT 0,
    byes            SMALLINT NOT NULL DEFAULT 0,
    legbyes         SMALLINT NOT NULL DEFAULT 0,
    is_wicket       BOOLEAN NOT NULL DEFAULT FALSE,
    player_out_id   INT REFERENCES players(id),
    dismissal_kind  TEXT,
    fielder_id      INT REFERENCES players(id)    -- catch / run out / stumping credit
);

CREATE INDEX IF NOT EXISTS idx_del_match  ON deliveries(match_id);
CREATE INDEX IF NOT EXISTS idx_del_batter ON deliveries(batter_id);
CREATE INDEX IF NOT EXISTS idx_del_bowler ON deliveries(bowler_id);
CREATE INDEX IF NOT EXISTS idx_del_field  ON deliveries(fielder_id);
CREATE INDEX IF NOT EXISTS idx_m_season   ON matches(season);

-- ------------------------------------------------- derived (refreshed post-ingest)
CREATE MATERIALIZED VIEW IF NOT EXISTS batting_innings AS
SELECT d.match_id, m.season, d.batter_id,
       SUM(d.runs_batter)                                   AS runs,
       SUM(CASE WHEN d.wides = 0 THEN 1 ELSE 0 END)         AS balls,
       SUM(CASE WHEN d.runs_batter = 4 THEN 1 ELSE 0 END)   AS fours,
       SUM(CASE WHEN d.runs_batter = 6 THEN 1 ELSE 0 END)   AS sixes,
       BOOL_OR(d.player_out_id = d.batter_id)               AS dismissed,
       MIN(d.bowling_team_id)                               AS opponent_id
FROM deliveries d JOIN matches m ON m.id = d.match_id
GROUP BY d.match_id, m.season, d.batter_id;

CREATE MATERIALIZED VIEW IF NOT EXISTS bowling_innings AS
SELECT d.match_id, m.season, d.bowler_id,
       SUM(CASE WHEN d.wides = 0 AND d.noballs = 0 THEN 1 ELSE 0 END) AS balls,
       SUM(d.runs_total - d.byes - d.legbyes)                         AS runs_conceded,
       SUM(CASE WHEN d.dismissal_kind IN
           ('bowled','caught','caught and bowled','lbw','stumped','hit wicket')
           THEN 1 ELSE 0 END)                                         AS wickets,
       SUM(CASE WHEN d.runs_total = 0 THEN 1 ELSE 0 END)              AS dot_balls,
       MIN(d.batting_team_id)                                         AS opponent_id
FROM deliveries d JOIN matches m ON m.id = d.match_id
GROUP BY d.match_id, m.season, d.bowler_id;

CREATE INDEX IF NOT EXISTS idx_bi_batter ON batting_innings(batter_id);
CREATE INDEX IF NOT EXISTS idx_bo_bowler ON bowling_innings(bowler_id);

-- match officials (umpires, TV umpires, referees) — from Cricsheet info.officials
CREATE TABLE IF NOT EXISTS officials (
    match_id TEXT NOT NULL REFERENCES matches(id),
    person   TEXT NOT NULL,
    role     TEXT NOT NULL,   -- umpire | tv_umpire | reserve_umpire | match_referee
    UNIQUE (match_id, person, role)
);
CREATE INDEX IF NOT EXISTS idx_off_person ON officials(person);

-- playing XI per match per team — from Cricsheet info.players
CREATE TABLE IF NOT EXISTS match_players (
    match_id  TEXT NOT NULL REFERENCES matches(id),
    team_id   INT NOT NULL REFERENCES teams(id),
    player_id INT NOT NULL REFERENCES players(id),
    UNIQUE (match_id, player_id)
);
CREATE INDEX IF NOT EXISTS idx_mp_team ON match_players(team_id);

-- ---------------------------------------------------------------- RAG (pgvector)
CREATE TABLE IF NOT EXISTS documents (
    id        SERIAL PRIMARY KEY,
    title     TEXT NOT NULL,
    doc_type  TEXT NOT NULL,        -- history | rules | team | bio | context
    season    INT,
    team      TEXT,
    player    TEXT,
    source    TEXT NOT NULL,
    content   TEXT NOT NULL,
    embedding vector(384)           -- BAAI/bge-small-en-v1.5
);
CREATE INDEX IF NOT EXISTS idx_doc_embedding ON documents
    USING hnsw (embedding vector_cosine_ops);

-- ---------------------------------------------------------------- provenance
CREATE TABLE IF NOT EXISTS data_sources (
    id           SERIAL PRIMARY KEY,
    name         TEXT NOT NULL,
    url          TEXT,
    retrieved_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    version      TEXT,
    status       TEXT NOT NULL DEFAULT 'validated',
    notes        TEXT
);

-- ------------------------------------ spec-required, DATA UNAVAILABLE (kept empty)
-- No free source provides coach/staff/auction/award data; rule 29 forbids
-- fabrication. Ingestion interfaces exist; tables stay empty until a licensed
-- source is added. Tools report this honestly.
CREATE TABLE IF NOT EXISTS team_staff (
    id        SERIAL PRIMARY KEY,
    team_id   INT REFERENCES teams(id),
    season    INT,
    person    TEXT,
    role      TEXT,                  -- head coach | batting coach | physio | ...
    source_id INT REFERENCES data_sources(id)
);
CREATE TABLE IF NOT EXISTS auctions (
    id           SERIAL PRIMARY KEY,
    season       INT,
    player_id    INT REFERENCES players(id),
    team_id      INT REFERENCES teams(id),
    price_inr    BIGINT,
    auction_type TEXT,
    source_id    INT REFERENCES data_sources(id)
);
CREATE TABLE IF NOT EXISTS awards (
    id         SERIAL PRIMARY KEY,
    season     INT,
    player_id  INT REFERENCES players(id),
    award_type TEXT,                 -- computed orange/purple cap live instead
    source_id  INT REFERENCES data_sources(id)
);
