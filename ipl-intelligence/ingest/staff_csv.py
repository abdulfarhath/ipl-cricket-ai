"""Load team staff (coaches, physios, analysts...) from a hand-maintained CSV.

No free licensed dataset carries IPL staff data, and this system never
fabricates. This importer is the honest path: YOU research the rows (e.g. from
iplt20.com team pages), fill db/staff.csv, and load them with full provenance.

CSV columns: team,season,person,role
Example row: Royal Challengers Bengaluru,2025,Andy Flower,head coach

Run: python -m ingest.staff_csv [path/to/staff.csv]
"""
import csv
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.db import execute, q1

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("staff_csv")

VALID_ROLES = {"head coach", "batting coach", "bowling coach", "fielding coach",
               "physio", "physiotherapist", "analyst", "assistant coach",
               "mentor", "director of cricket", "team manager", "captain"}


def main(csv_path: str) -> None:
    path = Path(csv_path)
    if not path.exists():
        log.error("file not found: %s — copy db/staff_template.csv, fill rows, retry", path)
        sys.exit(1)

    src = q1("""INSERT INTO data_sources (name, url, version, status, notes)
                VALUES ('manual_staff_csv', %s, 'v1', 'manually_verified',
                        'hand-researched team staff; verify against iplt20.com')
                RETURNING id""", (str(path),))

    n_ok = n_bad = 0
    with path.open(newline="", encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f), start=2):
            team = (row.get("team") or "").strip()
            person = (row.get("person") or "").strip()
            role = (row.get("role") or "").strip().lower()
            try:
                season = int(row.get("season", ""))
            except ValueError:
                season = 0
            if not (team and person and 2008 <= season <= 2025):
                log.warning("line %d skipped (need team, person, season 2008-2025)", i)
                n_bad += 1
                continue
            if role not in VALID_ROLES:
                log.warning("line %d skipped: unknown role '%s' (valid: %s)",
                            i, role, sorted(VALID_ROLES))
                n_bad += 1
                continue
            team_row = q1("SELECT id FROM teams WHERE name ILIKE '%%'||%s||'%%' LIMIT 1",
                          (team,))
            if not team_row:
                log.warning("line %d skipped: no team matching '%s'", i, team)
                n_bad += 1
                continue
            execute("""INSERT INTO team_staff (team_id, season, person, role, source_id)
                       VALUES (%s, %s, %s, %s, %s)""",
                    (team_row["id"], season, person, role, src["id"]))
            n_ok += 1
    log.info("loaded %d staff rows (%d skipped). team_staff tool now answers them.",
             n_ok, n_bad)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "db/staff.csv")
