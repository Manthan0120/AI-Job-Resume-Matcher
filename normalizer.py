# normalizer.py
# Minimal skeleton; expand with spaCy or regex rules, plus a curated skills map.

import re
from typing import Dict, List
from collections import Counter


# Example curated skills catalog and synonyms
CANON_SKILLS = {
    "python": ["python3", "py"],
    "pandas": [],
    "numpy": [],
    "sql": ["postgresql", "mysql", "sqlite"],
    "aws": ["amazon web services"],
    "gcp": ["google cloud"],
    "azure": ["microsoft azure"],
    "java": [],
    "javascript": ["js", "nodejs", "node.js"],
    "docker": [],
    "kubernetes": ["k8s"],
    "langchain": [],
    "chromadb": ["chroma db", "chroma-db"],
}

def _lemmatize(token: str) -> str:
    return token.lower().strip()

def canonicalize_skills(raw_text: str) -> List[str]:
    text = raw_text.lower()
    hits = []
    for canon, syns in CANON_SKILLS.items():
        candidates = [canon] + syns
        for c in candidates:
            # word-boundary match to avoid substring noise
            if re.search(rf"(?<![A-Za-z0-9]){re.escape(c)}(?![A-Za-z0-9])", text):
                hits.append(canon)
                break
    # dedupe by frequency
    return list(Counter(hits).keys())

def normalize_candidate_record(record: Dict) -> Dict:
    sections = record.get("sections", {})
    text = record.get("content", "")

    skills = canonicalize_skills((sections.get("skills") or "") + "\n" + text)
    # naive year extraction; replace with robust date parsing
    years = 0
    yrs = re.findall(r"(\d+)\s+(?:\+?\s*)?(?:years?|yrs)\b", text.lower())
    if yrs:
        years = max(int(y) for y in yrs if y.isdigit())

    return {
        "skills": skills,
        "years_experience_est": years,
        "has_education": bool(sections.get("education")),
        "has_experience": bool(sections.get("experience")),
    }

def normalize_job_record(record: Dict) -> Dict:
    fields = record.get("fields", {})
    req_text = "\n".join([fields.get("requirements", ""), fields.get("description", ""), fields.get("skills", "")])
    skills = canonicalize_skills(req_text)

    # simple must-haves: items explicitly in 'requirements' line-bullets
    must_haves = []
    for line in fields.get("requirements", "").splitlines():
        l = line.strip("-• ").lower()
        if not l:
            continue
        # heuristic: lines with 'must' or years often denote requirements
        if "must" in l or re.search(r"\b\d+\s+(?:years?|yrs)\b", l):
            must_haves.append(l)

    return {
        "skills": skills,
        "must_haves_raw": must_haves,
    }
