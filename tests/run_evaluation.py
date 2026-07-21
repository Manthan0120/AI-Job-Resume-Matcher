"""
Runs the uncurated-ish corpus (tests/evaluation_corpus.py) through both
ResumeJobMatcher and CareerAgent and reports mechanical metrics -- declared
here, before running, so there's no post-hoc criteria-shopping once results
are in.

Metrics (all computed from structural facts, not eyeballed judgment):
  - recall@k: does each resume's "paired" job appear in that resume's top-k
    matches? (a resume being a plausible fit for its paired job is itself an
    assumption, not verified -- but the check itself is mechanical)
  - escape_hatch_rate: fraction of skill extractions (resume + job side,
    across all runs) that fell back to the LLM
  - retry_rate: fraction of CareerAgent runs where the deterministic search
    retry fired
  - soft_match_trigger_rate / soft_match_confirm_rate: how often the
    soft-match check queued a candidate, and how often it actually confirmed
    an implied skill when it did
  - confidence distribution across all top_matches produced
  - fallback_count: how many runs produced raw_output/validation_error
    instead of a structured result (should be 0; if not, that's a real bug)

Usage:
    python tests/run_evaluation.py

Requires OPENAI_API_KEY with usable quota in .env. Uses real LLM/embedding
calls -- not free, not instant (32 texts seeded, 16 resumes each searched
against the full job pool with a real CareerAgent.run() and a real
ResumeJobMatcher.find_best_matches()).
"""
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config
from src.vector_store import VectorStore
from src.resume_matcher import ResumeJobMatcher
from src.career_agent import CareerAgent
from evaluation_corpus import JOBS, RESUMES

TOP_K = 5


def main():
    config = Config()
    if not config.OPENAI_API_KEY:
        print("No OPENAI_API_KEY configured in .env -- cannot run. Set it and retry.")
        return

    vs = VectorStore(config)
    vs.add_documents(JOBS)
    matcher = ResumeJobMatcher(config, vs)
    agent = CareerAgent(config, vs)

    pipeline_recall_hits = 0
    agent_recall_hits = 0
    escape_hatch_events = 0
    retry_fired = 0
    soft_match_triggered = 0
    soft_match_confirmed = 0
    confidence_counts = Counter()
    fallback_count = 0

    raw_results = []

    for resume in RESUMES:
        label = resume["label"]
        paired_job = resume["paired_job_id"]
        print(f"\n{'=' * 90}\nRESUME: {label} (paired job: {paired_job})\n{'=' * 90}")

        t0 = time.time()
        pipeline_matches = matcher.find_best_matches(resume["content"], top_k=TOP_K)
        pipeline_job_ids = [m.get("job_id") for m in pipeline_matches]
        pipeline_hit = paired_job in pipeline_job_ids
        pipeline_recall_hits += int(pipeline_hit)
        print(f"  ResumeJobMatcher ({time.time() - t0:.1f}s): "
              f"{[m['title'] for m in pipeline_matches]} | paired job recalled: {pipeline_hit}")

        t0 = time.time()
        agent_result = agent.run(resume["content"], top_k=TOP_K)
        elapsed = time.time() - t0

        if "raw_output" in agent_result or "validation_error" in agent_result:
            fallback_count += 1
            print(f"  CareerAgent ({elapsed:.1f}s): FELL BACK -- {agent_result}")
            raw_results.append({"label": label, "pipeline": pipeline_matches, "agent": agent_result})
            continue

        agent_job_ids = [m.get("job_id") for m in agent_result["top_matches"]]
        agent_hit = paired_job in agent_job_ids
        agent_recall_hits += int(agent_hit)
        print(f"  CareerAgent ({elapsed:.1f}s): "
              f"{[m['title'] for m in agent_result['top_matches']]} | paired job recalled: {agent_hit}")

        for m in agent_result["top_matches"]:
            if m.get("confidence"):
                confidence_counts[m["confidence"]["level"]] += 1

        trace = agent_result.get("_reasoning_trace", [])
        escape_hatch_events += sum(1 for l in trace if "invoking LLM extraction fallback" in l)
        if any("retrying with extracted skills" in l for l in trace):
            retry_fired += 1
        soft_match_candidates_this_run = sum(1 for l in trace if "checking whether missing skills are just phrased differently" in l)
        soft_match_confirms_this_run = sum(1 for l in trace if "LLM confirmed" in l and "are implied by resume wording" in l)
        if soft_match_candidates_this_run:
            soft_match_triggered += soft_match_candidates_this_run
            soft_match_confirmed += soft_match_confirms_this_run

        raw_results.append({"label": label, "pipeline": pipeline_matches, "agent": agent_result})

    n = len(RESUMES)
    print(f"\n\n{'#' * 90}\nSUMMARY ({n} resumes, top_k={TOP_K})\n{'#' * 90}")
    print(f"Recall@{TOP_K} -- paired job appears in top matches:")
    print(f"  ResumeJobMatcher: {pipeline_recall_hits}/{n} ({100*pipeline_recall_hits/n:.0f}%)")
    print(f"  CareerAgent:      {agent_recall_hits}/{n} ({100*agent_recall_hits/n:.0f}%)")
    print(f"\nEscape hatches fired: {escape_hatch_events} (across {n} resume runs)")
    print(f"Search retry fired: {retry_fired}/{n} runs ({100*retry_fired/n:.0f}%)")
    print(f"Soft-match candidates queued: {soft_match_triggered}, confirmed: {soft_match_confirmed} "
          f"({100*soft_match_confirmed/soft_match_triggered:.0f}% confirm rate)" if soft_match_triggered else
          "Soft-match candidates queued: 0")
    print(f"\nConfidence distribution across all top_matches: {dict(confidence_counts)}")
    print(f"\nFallback count (raw_output/validation_error instead of structured result): {fallback_count}/{n}")

    out_path = Path(__file__).resolve().parent / "evaluation_raw_results.json"
    out_path.write_text(json.dumps(raw_results, indent=2))
    print(f"\nRaw per-resume results written to {out_path}")


if __name__ == "__main__":
    main()
