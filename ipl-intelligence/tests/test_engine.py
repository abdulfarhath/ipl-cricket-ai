"""Deterministic DB tests (spec §24) — no LLM involved, run in CI on every change.
Run: python -m tests.test_engine
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stats import engine

FAILURES = []


def check(name, cond, got=None):
    status = "PASS" if cond else "FAIL"
    print(f"{status} {name}" + ("" if cond else f"  (got: {got})"))
    if not cond:
        FAILURES.append(name)


def main():
    b = engine.batting_stats("V Kohli")
    check("kohli career runs = 8671", b["career"]["runs"] == 8671, b["career"]["runs"])
    check("kohli hundreds >= 8", b["career"]["hundreds"] >= 8, b["career"]["hundreds"])

    h = engine.highest_score("V Kohli")["top_innings"][0]
    check("kohli highest = 113", h["runs"] == 113, h)
    check("kohli highest venue = Chinnaswamy", "Chinnaswamy" in h["venue"], h["venue"])

    s = engine.season_summary(2025)
    check("2025 champion = RCB", "Bengaluru" in s["final"]["winner"], s["final"])
    check("2025 purple cap = Prasidh Krishna",
          "Prasidh" in s["most_wickets_purple_cap"]["player"], s["most_wickets_purple_cap"])
    check("2025 orange cap = Sai Sudharsan (759)",
          s["most_runs_orange_cap"]["runs"] == 759, s["most_runs_orange_cap"])

    r = engine.records()
    check("record most runs = Kohli", r["most_career_runs"][0]["player"] == "V Kohli")
    check("record highest score = 175 (Gayle)",
          r["highest_individual_scores"][0]["runs"] == 175,
          r["highest_individual_scores"][0])

    d = engine.batter_vs_bowler("V Kohli", "JJ Bumrah")
    check("kohli vs bumrah balls = 103", d["balls"] == 103, d)

    e2016 = engine.season_summary(2016)
    check("2016 winner = SRH", "Sunrisers" in e2016["final"]["winner"], e2016["final"])

    staff = engine.team_staff("Royal Challengers")
    check("staff honestly unavailable",
          "never fabricates" in staff.get("error", "") or "staff" in staff)

    oos = engine.season_summary(2026)
    check("2026 refused (cutoff)", "error" in oos, oos)

    print(f"\n{len(FAILURES)} failures out of 12 checks")
    sys.exit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
