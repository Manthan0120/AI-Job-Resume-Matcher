import os
from dotenv import load_dotenv

# Load environment variables from F drive
load_dotenv(dotenv_path="F:/ai-resume-job-matcher/.env")

class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    CHROMA_PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIRECTORY", "F:/ai-resume-job-matcher/chroma_db")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
    SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))
    TOP_K_MATCHES = int(os.getenv("TOP_K_MATCHES", "5"))
