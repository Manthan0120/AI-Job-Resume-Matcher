"""
Aggregates src/career_agent.py's tool_calls.jsonl into an actual fallback rate.

Answers: "out of all CareerAgent.run() calls logged so far, what fraction used
extract_skills_llm / verify_claim at least once" -- not just raw call counts,
which would overcount a single run that calls a tool five times.

Usage:
    python tests/tool_call_stats.py [path-to-tool_calls.jsonl]
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

DEFAULT_LOG_PATH = Path(__file__).resolve().parent.parent / "tool_calls.jsonl"


def main():
    log_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_LOG_PATH
    if not log_path.exists():
        print(f"No log at {log_path} yet -- run some CareerAgent.run() calls first.")
        return

    runs_using_tool = defaultdict(set)
    call_counts = defaultdict(int)
    all_run_ids = set()

    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            all_run_ids.add(record["run_id"])
            runs_using_tool[record["tool"]].add(record["run_id"])
            call_counts[record["tool"]] += 1

    total_runs = len(all_run_ids)
    if total_runs == 0:
        print("Log exists but has no entries.")
        return

    print(f"Total distinct runs logged: {total_runs}\n")
    print(f"{'tool':<28}{'runs using it':<16}{'% of runs':<12}{'total calls':<12}")
    for tool_name in sorted(call_counts, key=lambda t: -len(runs_using_tool[t])):
        n_runs = len(runs_using_tool[tool_name])
        pct = 100 * n_runs / total_runs
        print(f"{tool_name:<28}{n_runs:<16}{pct:<12.1f}{call_counts[tool_name]:<12}")


if __name__ == "__main__":
    main()
