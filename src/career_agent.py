# career_agent.py
"""
Deterministic-first resume-job matcher with narrow LLM escape hatches.

This is a deliberate departure from an earlier version of this module, which
gave an LLM a set of tools and let it plan its own call sequence (a LangGraph
agent/tools loop). That version made the same LLM decide things a cheap
deterministic check can decide instead -- whether to re-extract skills with
an LLM, whether to retry a search, how many verification calls to make -- at
real cost (a full tool-planning loop per resume) and with no guarantee the
"good" behavior (e.g. retrying a weak search) happened on any given run.

The rule now: run the deterministic path first (skill extraction via the
rule-based catalog, exact skill-set comparison, embedding similarity). Only
call the LLM when a cheap signal says the deterministic path is probably
insufficient:
  - rule-based skill extraction came back too thin           -> extract_skills_llm escape hatch
  - keyword overlap is low but embedding similarity is high   -> one batched soft-match/paraphrase check
  - a claimed gap came from LLM-extracted (not exact) skills  -> one batched gap-verification call
None of these are the LLM's call to make; Python decides when each fires
based on a fixed threshold, and every firing is logged (see TOOL_CALL_LOG_PATH)
so the actual escape-hatch rate is a measurable number, not a guess. Every
firing is also appended to a human-readable `trace` list threaded through the
whole run and returned as result["_reasoning_trace"] -- the same underlying
data serves both explainability (why did this call happen) and, when an LLM
call fails, transparency about the degraded fallback that was taken instead
of a crash.
"""

from __future__ import annotations
import contextvars
import json
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from langchain_openai import ChatOpenAI

from src.normalizer import canonicalize_skills
from src.schemas import AgentMatchResult

try:
    import fcntl
    _HAS_FCNTL = True
except ImportError:
    # fcntl is POSIX-only. On Windows, concurrent multi-worker appends to
    # TOOL_CALL_LOG_PATH aren't locked -- short JSON lines rarely interleave
    # mid-write in practice, but this is a known, undefended gap there.
    _HAS_FCNTL = False


# --- Thresholds -------------------------------------------------------------
# None of these are calibrated against observed data (there isn't any logged
# yet -- see TOOL_CALL_LOG_PATH). They're reasonable-guess placeholders, not
# tuned values. Revisit once real runs are logged.

# Fewer than this many rule-based skills found -> treat extraction as too
# thin to trust, escalate to one LLM extraction call for that text.
SPARSE_SKILLS_THRESHOLD = 3

# Embedding cosine similarity at/above this, combined with low keyword
# overlap (below SOFT_MATCH_OVERLAP_THRESHOLD), is the specific signal that a
# resume might describe a required skill in different words than the job
# posting -- the one case with a demonstrated reason to spend an LLM call.
SOFT_MATCH_SIMILARITY_THRESHOLD = 0.75
SOFT_MATCH_OVERLAP_THRESHOLD = 0.3

# Confidence banding on the gap between match_percentage and its purely
# deterministic base (see _confidence_for). <=HIGH -> no LLM adjustment at
# all; <=MODERATE -> an LLM soft-match contribution of a normal size;
# above that -> unusually large relative to the deterministic signal, flag it.
CONFIDENCE_HIGH_MAX_DISCREPANCY_PCT = 5
CONFIDENCE_MODERATE_MAX_DISCREPANCY_PCT = 25

# Composite match_percentage weights: keyword overlap vs. embedding
# similarity. A judgment call, not a fitted value.
SKILL_OVERLAP_WEIGHT = 0.6
SIMILARITY_WEIGHT = 0.4
SOFT_MATCH_BONUS_WEIGHT = 0.15


# --- Durable, cross-process step log ---------------------------------------
# Records every deterministic step and every LLM escape hatch fired, so
# "how often does the LLM extraction / soft-match / gap-verification escape
# hatch actually fire" is answerable from tests/tool_call_stats.py instead of
# being an unlogged guess.
TOOL_CALL_LOG_PATH = Path(__file__).resolve().parent.parent / "tool_calls.jsonl"

# contextvars, not a plain self.attribute: CareerAgent is a singleton cached
# and reused across requests in backend/main.py's get_career_agent(). FastAPI
# runs sync path-operation functions in a threadpool, and Starlette copies the
# context at the moment of each dispatch (verified against the installed
# anyio's run_sync_in_worker_thread) -- so each concurrent request keeps its
# own run_id even though they share one CareerAgent instance.
_current_run_id = contextvars.ContextVar("career_agent_run_id", default="unknown")


def _record_step(step_name: str, meta: Optional[Dict] = None) -> None:
    record = {
        "run_id": _current_run_id.get(),
        "tool": step_name,
        "meta": meta or {},
        "ts": time.time(),
    }
    line = json.dumps(record) + "\n"
    with open(TOOL_CALL_LOG_PATH, "a") as f:
        if _HAS_FCNTL:
            fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.write(line)
        finally:
            if _HAS_FCNTL:
                fcntl.flock(f, fcntl.LOCK_UN)


def _extract_json(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    return text.strip()


def _compare_skills(resume_skills: List[str], job_skills: List[str]) -> Dict:
    """Deterministic set comparison -- no hallucination risk, always the
    first source of truth for matching/missing skills."""
    r, j = set(resume_skills), set(job_skills)
    matching = sorted(r & j)
    missing = sorted(j - r)
    ratio = len(matching) / len(j) if j else 0.0
    return {"matching_skills": matching, "missing_skills": missing, "skill_match_ratio": round(ratio, 4)}


class CareerAgent:
    """Deterministic-first resume-to-job matcher. Not a LangGraph agent
    loop -- see module docstring for why."""

    def __init__(self, config, vector_store):
        self.config = config
        self.vector_store = vector_store
        self.llm = ChatOpenAI(
            model=config.LLM_MODEL,
            openai_api_key=config.OPENAI_API_KEY,
            temperature=0,
        )

    # --- skill extraction: rule-based first, LLM only if sparse -----------

    def _extract_skills(self, text: str, label: str, trace: List[str]) -> Tuple[List[str], bool]:
        """Returns (skills, used_llm_escape_hatch). Degrades to rule-based-only
        (used_llm_escape_hatch=False) if the LLM call itself fails, rather
        than raising -- an LLM outage should make matching slightly less
        accurate, not make the endpoint 500."""
        skills = canonicalize_skills(text)
        _record_step("extract_skills_rule_based", {"label": label, "count": len(skills)})
        if len(skills) >= SPARSE_SKILLS_THRESHOLD:
            return skills, False

        trace.append(
            f"{label}: rule-based extraction found only {len(skills)} skill(s) "
            f"(below threshold of {SPARSE_SKILLS_THRESHOLD}) -- invoking LLM extraction fallback."
        )
        try:
            llm_skills = self._extract_skills_llm(text)
        except Exception as e:
            _record_step("extract_skills_llm_failed", {"label": label, "error": str(e)[:200]})
            trace.append(
                f"{label}: LLM extraction fallback failed ({type(e).__name__}) -- "
                f"continuing with the {len(skills)} rule-based skill(s) found only."
            )
            return skills, False

        _record_step("extract_skills_llm_escape_hatch", {"label": label, "rule_based_count": len(skills)})
        merged = sorted(set(skills) | set(llm_skills))
        trace.append(f"{label}: LLM fallback added {len(merged) - len(skills)} additional skill(s).")
        # Union, not replace: the rule-based catalog is exact where it hits;
        # the LLM pass fills gaps, it doesn't get to override a confirmed hit.
        return merged, True

    def _extract_skills_llm(self, text: str) -> List[str]:
        prompt = (
            "Extract a JSON array of concrete technical/professional skills "
            "mentioned in this text. Lowercase, no duplicates, no explanations, "
            "just a JSON array of strings.\n\nTEXT:\n" + text[:3000]
        )
        response = self.llm.invoke(prompt).content  # may raise -- caller degrades on failure
        try:
            skills = json.loads(_extract_json(response))
            return [s.lower() for s in skills] if isinstance(skills, list) else []
        except json.JSONDecodeError:
            return []

    # --- search: one deterministic retry, no LLM rephrasing ----------------

    def _search_jobs_with_retry(self, resume_content: str, resume_skills: List[str], k: int, trace: List[str]):
        results = self.vector_store.similarity_search(resume_content, doc_type="job", k=k)
        _record_step("search_jobs")
        best_score = max((1 - dist for _, dist in results), default=0.0)

        if best_score >= self.config.SIMILARITY_THRESHOLD or not resume_skills:
            return results

        # Exactly one retry, using the literal extracted skills as the query
        # instead of an LLM-improvised rephrase -- deterministic, and it's
        # the mechanism demonstrated to recover phrasing-mismatch dropouts.
        trace.append(
            f"Initial job search's best result scored {best_score:.2f} "
            f"(below the {self.config.SIMILARITY_THRESHOLD} threshold) -- "
            f"retrying with extracted skills {resume_skills} as the query."
        )
        _record_step("search_jobs_retry_literal_skills")
        retry_query = " ".join(resume_skills)
        retry_results = self.vector_store.similarity_search(retry_query, doc_type="job", k=k)
        retry_best = max((1 - dist for _, dist in retry_results), default=0.0)
        if retry_best > best_score:
            trace.append(f"Retry improved the best score to {retry_best:.2f} -- using retry results.")
            return retry_results
        trace.append("Retry did not improve on the initial results -- keeping the original search.")
        return results

    # --- soft-match batch: one call for every low-overlap/high-similarity candidate ---

    def _batch_check_soft_matches(self, resume_content: str, candidates: List[Dict], trace: List[str]) -> None:
        """Mutates each candidate in place: moves skills the LLM confirms are
        genuinely implied (not invented) from missing_skills to
        matching_skills, sets soft_match_bonus, and collects bullet rewrites.
        Degrades to a no-op (candidates left as pure keyword-overlap scores)
        if the LLM call fails."""
        for c in candidates:
            trace.append(
                f"{c['title']}: keyword overlap was {c['skill_match_ratio'] * 100:.0f}% but embedding "
                f"similarity was {c['similarity']:.2f} (>= {SOFT_MATCH_SIMILARITY_THRESHOLD}) -- "
                "checking whether missing skills are just phrased differently."
            )
        _record_step("soft_match_batch_llm", {"candidate_count": len(candidates)})

        job_lines = [
            f"{i}. Title: {c['title']} | Flagged missing skills: {c['missing_skills']}"
            for i, c in enumerate(candidates)
        ]
        prompt = (
            "A resume and several jobs were compared by exact keyword matching, which "
            "flagged some job requirements as \"missing\" from the resume. But each of these "
            "jobs is still semantically similar to the resume overall -- meaning the resume "
            "MIGHT describe a flagged skill in different words than the job posting uses.\n\n"
            f"Resume:\n{resume_content[:3000]}\n\n"
            "Jobs:\n" + "\n".join(job_lines) + "\n\n"
            "For each job in order, decide which flagged skills are genuinely implied by the "
            "resume's actual wording (never invent evidence that isn't there), and if any are, "
            "suggest ONE improved resume bullet per job that states it explicitly -- base it on "
            "an actual sentence from the resume, don't fabricate a new claim.\n\n"
            "Respond with ONLY a JSON array, one object per job in the same order:\n"
            '[{"implied_skills": ["..."], "bullet_rewrite": {"original": "...", "rewritten": "..."} or null}]'
        )
        try:
            response = self.llm.invoke(prompt).content
        except Exception as e:
            _record_step("soft_match_batch_failed", {"error": str(e)[:200]})
            trace.append(
                f"Soft-match check failed ({type(e).__name__}) -- skipping paraphrase detection for "
                f"{len(candidates)} candidate(s); their scores reflect exact keyword overlap only."
            )
            return

        try:
            parsed = json.loads(_extract_json(response))
        except json.JSONDecodeError:
            parsed = []

        for i, c in enumerate(candidates):
            entry = parsed[i] if i < len(parsed) and isinstance(parsed[i], dict) else {}
            implied = [s for s in entry.get("implied_skills", []) if s in c["missing_skills"]]
            if implied:
                c["missing_skills"] = [s for s in c["missing_skills"] if s not in implied]
                c["matching_skills"] = sorted(set(c["matching_skills"]) | set(implied))
                c["soft_match_bonus"] = SOFT_MATCH_BONUS_WEIGHT
                trace.append(f"{c['title']}: LLM confirmed {implied} are implied by resume wording.")
            else:
                trace.append(f"{c['title']}: LLM found no additional implied skills.")
            rewrite = entry.get("bullet_rewrite")
            if rewrite and rewrite.get("original") and rewrite.get("rewritten"):
                c["_bullet_rewrite"] = {"original": rewrite["original"], "rewritten": rewrite["rewritten"]}

    # --- gap verification batch: only for gaps touched by the LLM escape hatch ---

    def _batch_verify_gaps(self, resume_content: str, candidates: List[Dict], trace: List[str]) -> None:
        """Only called for candidates whose skill extraction used the LLM
        escape hatch on either side -- purely rule-based comparisons are
        exact regex matches and don't need an LLM to double-check them.
        Degrades to treating all queued gaps as genuinely missing (the
        conservative assumption) if the LLM call fails."""
        claims = [(c, skill) for c in candidates for skill in c["missing_skills"]]
        if not claims:
            return
        trace.append(
            f"{len(candidates)} candidate(s) had skill extraction fall back to the LLM, so their "
            f"{len(claims)} claimed gap(s) are being double-checked against the resume."
        )
        _record_step("verify_gaps_batch_llm", {"claim_count": len(claims)})

        claim_lines = [
            f"{i}. Job \"{c['title']}\" claims resume is missing: \"{skill}\""
            for i, (c, skill) in enumerate(claims)
        ]
        prompt = (
            f"Resume:\n{resume_content[:3000]}\n\n"
            "For each claim below, decide if the resume genuinely lacks that skill "
            "(true = really missing) or if this is a false negative and the resume "
            "actually demonstrates it (false = not really missing).\n\n"
            "Claims:\n" + "\n".join(claim_lines) + "\n\n"
            'Respond with ONLY a JSON array in the same order: [{"really_missing": true|false}]'
        )
        try:
            response = self.llm.invoke(prompt).content
        except Exception as e:
            _record_step("verify_gaps_batch_failed", {"error": str(e)[:200]})
            trace.append(
                f"Gap verification failed ({type(e).__name__}) -- unverified missing skills are "
                "reported as-is, treated conservatively as still missing."
            )
            for c, skill in claims:
                c.setdefault("verified_gaps", []).append(skill)
            return

        try:
            parsed = json.loads(_extract_json(response))
        except json.JSONDecodeError:
            parsed = []

        for i, (c, skill) in enumerate(claims):
            entry = parsed[i] if i < len(parsed) and isinstance(parsed[i], dict) else {}
            really_missing = entry.get("really_missing", True)
            c.setdefault("verified_gaps", [])
            if really_missing:
                c["verified_gaps"].append(skill)
            else:
                c["missing_skills"] = [s for s in c["missing_skills"] if s != skill]
                c["matching_skills"] = sorted(set(c["matching_skills"]) | {skill})
                trace.append(f"{c['title']}: verification found \"{skill}\" is a false gap, resume does show it.")

    # --- deterministic, templated summary: no LLM call -------------------

    @staticmethod
    def _overall_assessment(top: List[Dict]) -> str:
        if not top:
            return "No qualifying matches found above the configured similarity threshold."
        best = top[0]
        parts = [f"Best match: {best['title']} at {best['company']} ({best['match_percentage']}% overlap)."]
        gaps = best["missing_skills"]
        parts.append(f"Primary gaps: {', '.join(gaps[:5])}." if gaps else "No significant skill gaps detected for the top match.")
        if len(top) > 1:
            others = ", ".join(f"{c['title']} ({c['match_percentage']}%)" for c in top[1:])
            parts.append(f"Other candidates: {others}.")
        return " ".join(parts)

    # --- per-candidate confidence: continuous signal, not a binary flag ----

    @staticmethod
    def _confidence_for(c: Dict) -> Dict:
        discrepancy = c["match_percentage"] - c["_deterministic_base_pct"]
        if discrepancy <= CONFIDENCE_HIGH_MAX_DISCREPANCY_PCT:
            return {
                "level": "high",
                "reason": "Computed entirely from exact skill-keyword overlap and embedding "
                          "similarity -- no LLM judgment involved in this score.",
            }
        if discrepancy <= CONFIDENCE_MODERATE_MAX_DISCREPANCY_PCT:
            return {
                "level": "moderate",
                "reason": f"LLM confirmed skills implied by resume wording beyond exact keyword "
                          f"match, contributing {discrepancy} of the {c['match_percentage']}% score.",
            }
        return {
            "level": "low",
            "reason": f"LLM-attributed adjustment (+{discrepancy} points over the "
                      f"{c['_deterministic_base_pct']}% deterministic overlap) is unusually large "
                      "-- verify this match manually.",
        }

    # --- main entry point ---------------------------------------------------

    def run(self, resume_content: str, top_k: int = 5) -> Dict:
        run_id = uuid.uuid4().hex[:8]
        token = _current_run_id.set(run_id)
        trace: List[str] = []
        try:
            resume_skills, resume_used_llm = self._extract_skills(resume_content, "resume", trace)
            search_k = max(top_k * 2, top_k + 3)
            job_results = self._search_jobs_with_retry(resume_content, resume_skills, k=search_k, trace=trace)

            candidates = []
            soft_match_queue = []
            seen_job_ids = set()
            for doc, _vector_score in job_results:
                job_id = doc.metadata.get("id")
                if job_id in seen_job_ids:
                    continue
                seen_job_ids.add(job_id)

                title = doc.metadata.get("title", "Unknown")
                job_skills, job_used_llm = self._extract_skills(doc.page_content, f"job:{title}", trace)
                cmp = _compare_skills(resume_skills, job_skills)
                similarity = self.vector_store.calculate_cosine_similarity(resume_content, doc.page_content)
                _record_step("compare_skills")
                _record_step("score_similarity")

                candidate = {
                    "job_id": job_id,
                    "title": title,
                    "company": doc.metadata.get("company", "Unknown"),
                    "matching_skills": cmp["matching_skills"],
                    "missing_skills": cmp["missing_skills"],
                    "skill_match_ratio": cmp["skill_match_ratio"],
                    "similarity": similarity,
                    "needs_gap_verification": resume_used_llm or job_used_llm,
                }
                if cmp["skill_match_ratio"] < SOFT_MATCH_OVERLAP_THRESHOLD and similarity >= SOFT_MATCH_SIMILARITY_THRESHOLD:
                    soft_match_queue.append(candidate)
                candidates.append(candidate)

            if soft_match_queue:
                self._batch_check_soft_matches(resume_content, soft_match_queue, trace)

            for c in candidates:
                base = SKILL_OVERLAP_WEIGHT * c["skill_match_ratio"] + SIMILARITY_WEIGHT * c["similarity"]
                bonus = c.get("soft_match_bonus", 0.0)
                c["_deterministic_base_pct"] = round(base * 100)
                c["match_percentage"] = round(min(1.0, base + bonus) * 100)

            candidates.sort(key=lambda c: c["match_percentage"], reverse=True)
            top = candidates[:top_k]

            gap_verification_queue = [c for c in top if c["needs_gap_verification"] and c["missing_skills"]]
            if gap_verification_queue:
                self._batch_verify_gaps(resume_content, gap_verification_queue, trace)
            for c in top:
                c.setdefault("verified_gaps", list(c["missing_skills"]) if not c["needs_gap_verification"] else [])

            rewrites = [c.pop("_bullet_rewrite") for c in top if "_bullet_rewrite" in c]

            for c in top:
                c["confidence"] = self._confidence_for(c)

            result = {
                "top_matches": [
                    {
                        "job_id": c["job_id"],
                        "title": c["title"],
                        "company": c["company"],
                        "match_percentage": c["match_percentage"],
                        "matching_skills": c["matching_skills"],
                        "missing_skills": c["missing_skills"],
                        "verified_gaps": c["verified_gaps"],
                        "confidence": c["confidence"],
                        "recommendations": (
                            [f"Address gap: {s}" for s in c["missing_skills"][:3]]
                            if c["missing_skills"] else ["Strong match -- no significant gaps detected."]
                        ),
                    }
                    for c in top
                ],
                "suggested_bullet_rewrites": rewrites,
                "overall_assessment": self._overall_assessment(top),
            }
            result["_verification"] = {
                "checked": True,
                "max_deterministic_skill_overlap_pct": max((c["_deterministic_base_pct"] for c in top), default=0),
                "flagged_matches": [c["title"] for c in top if c["confidence"]["level"] == "low"],
            }
            result["_reasoning_trace"] = trace
            return self._validate(result)
        finally:
            _current_run_id.reset(token)

    @staticmethod
    def _validate(result: Dict) -> Dict:
        side_channel_keys = ("_verification", "_reasoning_trace")
        try:
            validated = AgentMatchResult.model_validate(
                {k: v for k, v in result.items() if k not in side_channel_keys}
            )
        except Exception as e:
            return {"raw_output": json.dumps({k: v for k, v in result.items() if k not in side_channel_keys}),
                    "validation_error": str(e), "_reasoning_trace": result.get("_reasoning_trace", [])}
        validated_dict = validated.model_dump()
        for key in side_channel_keys:
            validated_dict[key] = result[key]
        return validated_dict
