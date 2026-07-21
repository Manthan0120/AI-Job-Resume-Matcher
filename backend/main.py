# backend/main.py
"""
FastAPI backend for the AI Resume Job Matcher.

This replaces app.py (Streamlit) as the interface layer. It does not
change any matching logic -- it wraps the existing src/ modules
(VectorStore, ResumeJobMatcher, CareerAgent, DataProcessor) behind
HTTP endpoints so a separate React frontend can call them.
"""
import os
import tempfile
from typing import List, Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.config import Config
from src.data_processor import DataProcessor
from src.vector_store import VectorStore
from src.resume_matcher import ResumeJobMatcher
from src.career_agent import CareerAgent

app = FastAPI(title="AI Resume Job Matcher API")

# The React dev server runs on a different port than the API, so the
# browser blocks requests without explicit CORS permission. Restrict
# origins to your actual dev/prod frontend URLs, not "*", once deployed.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite dev server default
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

config = Config()
_state = {"vector_store": None, "matcher": None, "career_agent": None}


def get_vector_store() -> VectorStore:
    if _state["vector_store"] is None:
        if not config.OPENAI_API_KEY:
            raise HTTPException(status_code=400, detail="OPENAI_API_KEY is not set on the server (.env).")
        _state["vector_store"] = VectorStore(config)
    return _state["vector_store"]


def get_matcher() -> ResumeJobMatcher:
    if _state["matcher"] is None:
        _state["matcher"] = ResumeJobMatcher(config, get_vector_store())
    return _state["matcher"]


def get_career_agent() -> CareerAgent:
    if _state["career_agent"] is None:
        _state["career_agent"] = CareerAgent(config, get_vector_store())
    return _state["career_agent"]


# ---------- schemas ----------

class MatchRequest(BaseModel):
    content: str
    top_k: int = 5


class AgentRequest(BaseModel):
    resume_content: str
    top_k: int = 3


class ChatRequest(BaseModel):
    question: str
    chat_history: List[dict] = []


# ---------- health ----------

@app.get("/api/health")
def health():
    return {"status": "ok", "openai_key_configured": bool(config.OPENAI_API_KEY)}


# ---------- data management ----------

@app.post("/api/documents/resumes")
async def upload_resumes(files: List[UploadFile] = File(...)):
    processor = DataProcessor()
    vector_store = get_vector_store()
    resumes = []

    for file in files:
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        try:
            if file.filename.endswith(".pdf"):
                text = processor.extract_text_from_pdf(tmp_path)
            elif file.filename.endswith(".docx"):
                text = processor.extract_text_from_docx(tmp_path)
            else:
                text = content.decode("utf-8")

            if text:
                resumes.append({
                    "id": file.filename,
                    "filename": file.filename,
                    "content": processor.clean_text(text),
                    "type": "resume",
                })
        finally:
            os.unlink(tmp_path)

    if resumes:
        vector_store.add_documents(resumes)

    return {"processed": len(resumes), "filenames": [r["filename"] for r in resumes]}


@app.post("/api/documents/jobs")
async def upload_jobs(file: UploadFile = File(...)):
    import pandas as pd
    import io

    raw = await file.read()
    df = pd.read_csv(io.BytesIO(raw))

    vector_store = get_vector_store()
    jobs = []
    skipped = 0

    for index, row in df.iterrows():
        title = (row.get("title") or row.get("job_title") or row.get("position") or "Unknown Position")
        description = (row.get("description") or row.get("job_description") or row.get("details") or "")
        requirements = (row.get("requirements") or row.get("qualifications") or row.get("skills") or "")
        company = (row.get("company") or row.get("employer") or "Unknown Company")

        job_desc = f"Title: {title}\n\nDescription: {description}\n\nRequirements: {requirements}"

        if job_desc.strip() and len(job_desc.strip()) > 20:
            jobs.append({
                "id": f"job_{index}",
                "title": str(title),
                "company": str(company),
                "content": job_desc.strip(),
                "type": "job",
            })
        else:
            skipped += 1

    if jobs:
        vector_store.add_documents(jobs)

    return {"processed": len(jobs), "skipped": skipped, "columns": df.columns.tolist()}


# ---------- matching (fixed pipeline) ----------

@app.post("/api/match/resume-to-jobs")
def match_resume_to_jobs(req: MatchRequest):
    matcher = get_matcher()
    return {"matches": matcher.find_best_matches(req.content, top_k=req.top_k)}


@app.post("/api/match/job-to-resumes")
def match_job_to_resumes(req: MatchRequest):
    matcher = get_matcher()
    return {"matches": matcher.find_best_resumes(req.content, top_k=req.top_k)}


# ---------- agentic matching ----------

@app.post("/api/agent/career-match")
def agent_career_match(req: AgentRequest):
    agent = get_career_agent()
    try:
        result = agent.run(req.resume_content, top_k=req.top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent run failed: {e}")
    return result


# ---------- chat ----------

@app.post("/api/chat")
def chat(req: ChatRequest):
    matcher = get_matcher()
    return matcher.get_chat_response(req.question, chat_history=req.chat_history)


# ---------- analytics ----------

@app.get("/api/analytics/metrics")
def analytics_metrics():
    vector_store = get_vector_store()
    metrics = {"total_resumes": 0, "total_jobs": 0, "total_documents": 0}

    if not vector_store.vectorstore:
        return metrics

    all_docs = vector_store.vectorstore.get(include=["metadatas"])
    if not all_docs or "metadatas" not in all_docs:
        return metrics

    metadatas = all_docs["metadatas"]
    metrics["total_documents"] = len(metadatas)

    unique_resume_ids, unique_job_ids = set(), set()
    for metadata in metadatas:
        doc_type = metadata.get("type", "")
        doc_id = metadata.get("id", "")
        if doc_id:
            if doc_type == "resume":
                unique_resume_ids.add(doc_id)
            elif doc_type == "job":
                unique_job_ids.add(doc_id)

    metrics["total_resumes"] = len(unique_resume_ids)
    metrics["total_jobs"] = len(unique_job_ids)
    return metrics


# ---------- config (set API key at runtime, mirrors old sidebar input) ----------

@app.post("/api/config/api-key")
def set_api_key(api_key: str = Body(..., embed=True)):
    config.OPENAI_API_KEY = api_key
    os.environ["OPENAI_API_KEY"] = api_key
    # Reset cached instances so they pick up the new key
    _state["vector_store"] = None
    _state["matcher"] = None
    _state["career_agent"] = None
    return {"status": "ok"}
