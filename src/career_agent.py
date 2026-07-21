# career_agent.py
"""
Agentic resume-job matching system.

Unlike resume_matcher.py (a fixed pipeline: search -> filter -> analyze),
this module gives an LLM a set of tools and a goal, and lets the model
itself decide which tool to call, in what order, and when to stop.
That's what makes it "agentic" rather than a conditional workflow.

Architecture: LangGraph state machine with two nodes
    agent  -> LLM decides: call a tool, or respond with final answer
    tools  -> executes whatever tool(s) the LLM asked for, returns results
The graph loops agent -> tools -> agent -> ... until the LLM stops
requesting tools, or a step limit is hit (recursion_limit) so a
confused agent can't loop forever and burn API credits.
"""

from __future__ import annotations
import json
from typing import List, Dict, Optional

from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, END
from langgraph.prebuilt import ToolNode, tools_condition

from src.normalizer import canonicalize_skills


def build_tools(vector_store, llm):
    """
    Build the toolset the agent can choose from. Tools are closures over
    vector_store/llm so they can be plain @tool functions (what LangGraph
    tool-calling expects) while still reaching your existing infrastructure.
    """

    @tool
    def search_jobs(query: str, k: int = 5) -> str:
        """Search the job database for postings semantically similar to `query`.
        Use this to find candidate jobs for a resume, or to re-search with
        different phrasing if earlier results looked like weak matches."""
        results = vector_store.similarity_search(query, doc_type="job", k=k)
        if not results:
            return "No jobs found."
        out = []
        seen = set()
        for doc, score in results:
            jid = doc.metadata.get("id")
            if jid in seen:
                continue
            seen.add(jid)
            out.append({
                "job_id": jid,
                "title": doc.metadata.get("title", "Unknown"),
                "company": doc.metadata.get("company", "Unknown"),
                "content": doc.page_content[:800],
            })
        return json.dumps(out)

    @tool
    def extract_skills_rule_based(text: str) -> str:
        """Extract a canonical skills list from text using the fast rule-based
        normalizer. Cheap and exact, but only catches skills in the curated
        catalog -- may miss skills phrased unusually. Call this first."""
        return json.dumps(canonicalize_skills(text))

    @tool
    def extract_skills_llm(text: str) -> str:
        """Extract skills from text using the LLM instead of the rule-based
        catalog. Use this ONLY when extract_skills_rule_based looks incomplete
        (e.g. returned very few skills for a long, detailed text) -- it's
        slower and costs a call, so don't use it by default."""
        prompt = (
            "Extract a flat JSON list of concrete technical/professional "
            "skills mentioned in this text. Lowercase, no duplicates, no "
            "explanations, just a JSON array of strings.\n\nTEXT:\n" + text[:3000]
        )
        response = llm.invoke(prompt)
        try:
            return json.dumps(json.loads(response.content))
        except json.JSONDecodeError:
            return json.dumps([])

    @tool
    def compare_skills(resume_skills: List[str], job_skills: List[str]) -> str:
        """Deterministically compare two skill lists. Returns matching skills,
        missing skills (in job but not resume), and a match ratio. This is
        plain set logic -- no hallucination risk -- always prefer this over
        asking the LLM to eyeball the gap itself."""
        r, j = set(resume_skills), set(job_skills)
        matching = sorted(r & j)
        missing = sorted(j - r)
        ratio = len(matching) / len(j) if j else 0.0
        return json.dumps({
            "matching_skills": matching,
            "missing_skills": missing,
            "skill_match_ratio": round(ratio, 2),
        })

    @tool
    def score_similarity(text1: str, text2: str) -> str:
        """Compute cosine similarity between two texts using OpenAI embeddings.
        Use this for a precise numeric match score between resume and job."""
        score = vector_store.calculate_cosine_similarity(text1, text2)
        return json.dumps({"cosine_similarity": round(score, 3)})

    @tool
    def rewrite_bullet(bullet: str, target_skills: List[str]) -> str:
        """Rewrite a single resume bullet point so it more clearly surfaces
        skills the candidate already has but has phrased weakly. Do NOT
        invent skills the candidate doesn't have -- only rephrase for clarity
        and keyword visibility (helps with ATS keyword matching too)."""
        prompt = (
            "Rewrite this resume bullet to be more impactful and to clearly "
            "surface these relevant skills, ONLY if the bullet already implies "
            f"them (never invent experience): {target_skills}\n\n"
            f"Original bullet: {bullet}\n\n"
            "Return only the rewritten bullet, nothing else."
        )
        response = llm.invoke(prompt)
        return response.content.strip()

    @tool
    def verify_claim(claim: str, source_text: str) -> str:
        """Self-check tool: verify whether `claim` (e.g. 'candidate is missing
        Docker experience') is actually true given `source_text`. Use this
        before reporting a gap or missing qualification you're not fully sure
        about, to avoid reporting a false gap."""
        prompt = (
            f"Claim: {claim}\n\nSource text:\n{source_text[:2500]}\n\n"
            "Is this claim accurate based on the source text? Reply with a "
            "JSON object: {\"accurate\": true/false, \"reason\": \"...\"}"
        )
        response = llm.invoke(prompt)
        try:
            return json.dumps(json.loads(response.content))
        except json.JSONDecodeError:
            return json.dumps({"accurate": None, "reason": response.content[:300]})

    return [
        search_jobs,
        extract_skills_rule_based,
        extract_skills_llm,
        compare_skills,
        score_similarity,
        rewrite_bullet,
        verify_claim,
    ]


SYSTEM_PROMPT = """You are a career agent helping a candidate get hired.

Goal: given the candidate's resume, find the best-fit job(s), determine
their REAL skill gaps (verified, not guessed), and produce concrete,
honest recommendations -- including rewritten resume bullets where the
candidate already has a skill but phrased it weakly.

Rules:
- Prefer rule-based / deterministic tools (extract_skills_rule_based,
  compare_skills, score_similarity) before reaching for an LLM tool.
- Only use extract_skills_llm if the rule-based extraction looks sparse.
- Before listing a missing qualification in your final answer, if you are
  not fully confident, call verify_claim to check it against the resume text.
- If initial job search results look like weak matches, try search_jobs
  again with different query phrasing before giving up.
- When you are done gathering information, respond with a final answer
  (no more tool calls) as a JSON object with this exact structure:
  {
    "top_matches": [
      {"job_id": "...", "title": "...", "company": "...",
       "match_percentage": <0-100 int>,
       "matching_skills": [...], "missing_skills": [...],
       "verified_gaps": [...],
       "recommendations": [...]}
    ],
    "suggested_bullet_rewrites": [
      {"original": "...", "rewritten": "..."}
    ],
    "overall_assessment": "..."
  }
Do not fabricate matches, skills, or gaps that your tools did not surface."""


class CareerAgent:
    """Agentic resume-to-job matcher built on LangGraph."""

    def __init__(self, config, vector_store, max_steps: int = 12):
        self.config = config
        self.vector_store = vector_store
        self.max_steps = max_steps
        self.llm = ChatOpenAI(
            model=config.LLM_MODEL,
            openai_api_key=config.OPENAI_API_KEY,
            temperature=0,
        )
        self.tools = build_tools(vector_store, self.llm)
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        self.graph = self._build_graph()

    def _agent_node(self, state: MessagesState):
        response = self.llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    def _build_graph(self):
        graph = StateGraph(MessagesState)
        graph.add_node("agent", self._agent_node)
        graph.add_node("tools", ToolNode(self.tools))
        graph.set_entry_point("agent")
        # tools_condition inspects the last message: if the LLM requested
        # tool calls, route to "tools"; otherwise route to END (agent is done)
        graph.add_conditional_edges("agent", tools_condition, {
            "tools": "tools",
            END: END,
        })
        graph.add_edge("tools", "agent")  # loop back after tool results
        return graph.compile()

    def run(self, resume_content: str, top_k: int = 5) -> Dict:
        """Run the agent loop end-to-end for a given resume."""
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=(
                f"Candidate resume:\n{resume_content[:4000]}\n\n"
                f"Find the top {top_k} job matches and produce the analysis "
                "described in your instructions."
            )),
        ]
        # recursion_limit caps agent<->tools round trips so a confused
        # agent can't loop indefinitely and burn API credits
        final_state = self.graph.invoke(
            {"messages": messages},
            config={"recursion_limit": self.max_steps * 2},
        )
        last_message = final_state["messages"][-1]
        try:
            return json.loads(last_message.content)
        except json.JSONDecodeError:
            return {"raw_output": last_message.content}