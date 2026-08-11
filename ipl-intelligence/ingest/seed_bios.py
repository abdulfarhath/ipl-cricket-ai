"""Seed curated profile fields (full name, nationality, role, styles) for the
~33 most-searched players. Cricsheet has no bio data; these are hand-curated
public facts. Everyone else still gets full stats — just no bio fields.
Run: python -m ingest.seed_bios
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.db import execute

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

# cricsheet_name: (full_name, nationality, role, batting_style, bowling_style)
BIOS = {
    "V Kohli": ("Virat Kohli", "India", "batter", "right-hand bat", None),
    "RG Sharma": ("Rohit Sharma", "India", "batter", "right-hand bat", None),
    "MS Dhoni": ("Mahendra Singh Dhoni", "India", "wicketkeeper-batter", "right-hand bat", None),
    "JJ Bumrah": ("Jasprit Bumrah", "India", "bowler", "right-hand bat", "right-arm fast"),
    "SK Raina": ("Suresh Raina", "India", "batter", "left-hand bat", "off-spin"),
    "DA Warner": ("David Warner", "Australia", "batter", "left-hand bat", None),
    "CH Gayle": ("Chris Gayle", "West Indies", "batter", "left-hand bat", "off-spin"),
    "AB de Villiers": ("Abraham Benjamin de Villiers", "South Africa", "batter", "right-hand bat", None),
    "R Ashwin": ("Ravichandran Ashwin", "India", "bowler", "right-hand bat", "off-spin"),
    "YS Chahal": ("Yuzvendra Chahal", "India", "bowler", "right-hand bat", "leg-spin"),
    "RA Jadeja": ("Ravindra Jadeja", "India", "all-rounder", "left-hand bat", "left-arm orthodox"),
    "HH Pandya": ("Hardik Pandya", "India", "all-rounder", "right-hand bat", "right-arm medium-fast"),
    "KL Rahul": ("Kannaur Lokesh Rahul", "India", "wicketkeeper-batter", "right-hand bat", None),
    "S Dhawan": ("Shikhar Dhawan", "India", "batter", "left-hand bat", None),
    "RR Pant": ("Rishabh Pant", "India", "wicketkeeper-batter", "left-hand bat", None),
    "SV Samson": ("Sanju Samson", "India", "wicketkeeper-batter", "right-hand bat", None),
    "SA Yadav": ("Suryakumar Yadav", "India", "batter", "right-hand bat", None),
    "Shubman Gill": ("Shubman Gill", "India", "batter", "right-hand bat", None),
    "B Sai Sudharsan": ("Bharathidasan Sai Sudharsan", "India", "batter", "left-hand bat", None),
    "M Prasidh Krishna": ("Prasidh Krishna", "India", "bowler", "right-hand bat", "right-arm fast"),
    "JC Buttler": ("Jos Buttler", "England", "wicketkeeper-batter", "right-hand bat", None),
    "SP Narine": ("Sunil Narine", "West Indies", "all-rounder", "left-hand bat", "mystery spin"),
    "AD Russell": ("Andre Russell", "West Indies", "all-rounder", "right-hand bat", "right-arm fast"),
    "Rashid Khan": ("Rashid Khan", "Afghanistan", "bowler", "right-hand bat", "leg-spin"),
    "F du Plessis": ("Faf du Plessis", "South Africa", "batter", "right-hand bat", None),
    "Q de Kock": ("Quinton de Kock", "South Africa", "wicketkeeper-batter", "left-hand bat", None),
    "SL Malinga": ("Lasith Malinga", "Sri Lanka", "bowler", "right-hand bat", "right-arm fast"),
    "DJ Bravo": ("Dwayne Bravo", "West Indies", "all-rounder", "right-hand bat", "right-arm medium-fast"),
    "SR Tendulkar": ("Sachin Tendulkar", "India", "batter", "right-hand bat", None),
    "G Gambhir": ("Gautam Gambhir", "India", "batter", "left-hand bat", None),
    "V Sehwag": ("Virender Sehwag", "India", "batter", "right-hand bat", None),
    "TA Boult": ("Trent Boult", "New Zealand", "bowler", "right-hand bat", "left-arm fast"),
    "Arshdeep Singh": ("Arshdeep Singh", "India", "bowler", "left-hand bat", "left-arm fast-medium"),
}


def main() -> None:
    n = 0
    for name, (full, nat, role, bat, bowl) in BIOS.items():
        execute("""UPDATE players SET full_name=%s, nationality=%s, role=%s,
                   batting_style=%s, bowling_style=%s WHERE cricsheet_name=%s""",
                (full, nat, role, bat, bowl, name))
        n += 1
    logging.info("seeded %d player bios", n)


if __name__ == "__main__":
    main()
