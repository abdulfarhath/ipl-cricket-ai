"""End-to-end agent evaluation (spec §24): golden questions with known answers,
checks correctness fragments + latency, including out-of-scope behavior.
Run: python -m tests.eval_agent   (needs LLM API key; costs a few requests)
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.graph import answer

GOLDEN = [
    ("Who won the IPL 2016 final?", ["sunrisers"], "stats"),
    ("What is Virat Kohli's total IPL career run count?", ["8671"], "stats"),
    ("Who took the most wickets in IPL 2025?", ["prasidh"], "stats"),
    ("What is Chris Gayle's highest IPL score?", ["175"], "stats"),
    ("When did the IPL start?", ["2008"], "knowledge"),
    ("What is quantum computing?", ["ipl"], "out_of_scope"),
    ("Who is in RCB's current 2026 squad?", ["2025"], "current_data"),
]


def main():
    n_pass = 0
    lats = []
    for question, expected, want_intent in GOLDEN:
        t0 = time.perf_counter()
        r = answer(question)
        dt = time.perf_counter() - t0
        lats.append(dt)
        text = r["answer"].lower().replace(",", "")
        ok = all(f in text for f in expected) and r["intent"] == want_intent
        n_pass += ok
        print(f"{'PASS' if ok else 'FAIL':4} {dt:5.1f}s [{r['intent']:12}] {question}")
        if not ok:
            print(f"     wanted {expected}/{want_intent}; got: {r['answer'][:120]}")
    lats.sort()
    print(f"\naccuracy {n_pass}/{len(GOLDEN)} | "
          f"latency p50 {lats[len(lats)//2]:.1f}s max {lats[-1]:.1f}s")
    sys.exit(0 if n_pass == len(GOLDEN) else 1)


if __name__ == "__main__":
    main()
