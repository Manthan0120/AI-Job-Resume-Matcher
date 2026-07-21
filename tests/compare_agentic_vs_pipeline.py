"""
Empirical comparison: ResumeJobMatcher (fixed pipeline) vs CareerAgent (agentic).

This exists because "the agent should do better on certain resume/job pairs"
is a hypothesis until it's actually run and diffed. This script constructs a
handful of pairs, each targeting one specific claimed advantage of the agent
over the fixed pipeline, runs both, and prints a side-by-side so you can see
whether the claim holds -- not just whether the two outputs differ.

Usage:
    python tests/compare_agentic_vs_pipeline.py

Requires a real OPENAI_API_KEY with usable quota in .env (this makes real
LLM/embedding calls -- it is not free and not instant: each pair runs one
ResumeJobMatcher.find_best_matches call and one full CareerAgent.run() loop,
which can be several LLM calls on its own).
"""
import sys
import time
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config
from src.vector_store import VectorStore
from src.resume_matcher import ResumeJobMatcher
from src.career_agent import CareerAgent


# Each pair names the specific mechanism it's meant to exercise, so a result
# can be read as "did that mechanism actually fire and help" rather than a
# vague vibe check.

JOBS = [
    {
        "id": "job_devops", "title": "DevOps Engineer", "company": "CloudScale Technologies",
        "content": "Title: DevOps Engineer\n\nDescription: Own CI/CD and cloud infra.\n\n"
                   "Requirements: Docker, Kubernetes, CI/CD, Terraform, AWS, 3+ years",
        "type": "job",
    },
    {
        "id": "job_backend", "title": "Backend Software Engineer", "company": "DataWorks Inc.",
        "content": "Title: Backend Software Engineer\n\nDescription: Build and scale REST APIs.\n\n"
                   "Requirements: Python, FastAPI or Django, PostgreSQL, Docker, AWS, 3+ years",
        "type": "job",
    },
    {
        "id": "job_ml", "title": "Machine Learning Engineer", "company": "NeuralEdge AI",
        "content": "Title: Machine Learning Engineer\n\nDescription: Deploy ML models for recommendations.\n\n"
                   "Requirements: Python, scikit-learn, TensorFlow or PyTorch, model deployment, AWS or GCP",
        "type": "job",
    },
]

PAIRS = [
    {
        "name": "phrasing_mismatch_recovery",
        "targets": "Resume describes container/orchestration work in dense prose with no literal "
                   "'Docker'/'Kubernetes' tokens, against a terse keyword-list job. Claim: pipeline's "
                   "one-shot embedding search may score this below SIMILARITY_THRESHOLD and drop the "
                   "job; agent can extract skills, notice the gap, and retry search_jobs with reformulated "
                   "terms to recover it.",
        "resume": (
            "Jordan Reyes\n\nSummary: Infrastructure engineer who architected a multi-tenant SaaS "
            "platform on orchestrated, containerized microservices with automated, repeatable rollout "
            "pipelines and fleet-wide configuration management across cloud nodes.\n\n"
            "Experience: Led the move from monolith to a fleet of isolated, orchestrated service "
            "instances; built the automated build-test-deploy pipeline that cut release time from days "
            "to minutes; managed infrastructure-as-code for all cloud compute fleets.\n\n"
            "Education: B.S. Computer Science."
        ),
    },
    {
        "name": "false_gap_suppression",
        "targets": "Resume clearly demonstrates Docker/AWS/Python but truncation or a single-shot "
                   "prompt might cause a fixed pipeline to still list them as gaps. Claim: agent's "
                   "compare_skills (deterministic set logic) + verify_claim should suppress a false gap "
                   "that a bare LLM judgment call might report.",
        "resume": (
            "Priya Nair\n\nSummary: Backend engineer, 5 years, deep AWS and containerization background.\n\n"
            "Skills: Python, FastAPI, Docker, AWS (EC2, ECS, Lambda), PostgreSQL, Git\n\n"
            "Experience: Built and deployed containerized Python services on AWS ECS; wrote FastAPI "
            "services backed by PostgreSQL; owns CI/CD via Docker images.\n\n"
            "Education: B.S. Computer Science."
        ),
    },
    {
        "name": "genuine_gap_case",
        "targets": "Control case: a resume with real, unambiguous gaps against an ML role. Both systems "
                   "should agree these are real gaps -- if the agent 'discovers' gaps the resume text "
                   "doesn't support, or the pipeline misses obvious ones, that's a finding either way.",
        "resume": (
            "Sam Okafor\n\nSummary: Frontend developer, 3 years, React and TypeScript.\n\n"
            "Skills: JavaScript, React, TypeScript, CSS, REST API integration\n\n"
            "Experience: Built responsive UIs consuming REST APIs; no backend, ML, or cloud "
            "infrastructure experience listed.\n\nEducation: B.S. Computer Science."
        ),
    },
]


def seed_vector_store(vs: VectorStore):
    vs.add_documents(JOBS)


def run_pipeline(matcher: ResumeJobMatcher, resume_content: str):
    t0 = time.time()
    matches = matcher.find_best_matches(resume_content, top_k=3)
    elapsed = time.time() - t0
    return {
        "elapsed_sec": round(elapsed, 1),
        "jobs_surfaced": [m["title"] for m in matches],
        "missing_by_job": {m["title"]: m["analysis"].get("missing_qualifications", []) for m in matches},
    }


def run_agent(agent: CareerAgent, resume_content: str):
    t0 = time.time()
    result = agent.run(resume_content, top_k=3)
    elapsed = time.time() - t0
    top_matches = result.get("top_matches", [])
    return {
        "elapsed_sec": round(elapsed, 1),
        "jobs_surfaced": [m.get("title") for m in top_matches],
        "missing_by_job": {m.get("title"): m.get("missing_skills", []) for m in top_matches},
        "verified_gaps_by_job": {m.get("title"): m.get("verified_gaps", []) for m in top_matches},
        "verification": result.get("_verification"),
        "raw_output": result.get("raw_output"),
    }


def main():
    config = Config()
    if not config.OPENAI_API_KEY:
        print("No OPENAI_API_KEY configured in .env -- cannot run. Set it and retry.")
        return

    vs = VectorStore(config)
    seed_vector_store(vs)
    matcher = ResumeJobMatcher(config, vs)
    agent = CareerAgent(config, vs)

    for pair in PAIRS:
        print("=" * 100)
        print(f"PAIR: {pair['name']}")
        print(f"TARGETS: {pair['targets']}")
        print("-" * 100)

        pipeline_result = run_pipeline(matcher, pair["resume"])
        print("PIPELINE (ResumeJobMatcher):")
        print(json.dumps(pipeline_result, indent=2))

        agent_result = run_agent(agent, pair["resume"])
        print("\nAGENT (CareerAgent):")
        print(json.dumps(agent_result, indent=2))
        print()


if __name__ == "__main__":
    main()
