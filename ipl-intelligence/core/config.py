"""Central configuration. Everything overridable via environment / .env."""
import os
from pathlib import Path


def _load_dotenv() -> None:
    env = Path(__file__).resolve().parent.parent / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://ipl:ipl@localhost:5433/ipl")
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6380/0")

DATA_CUTOFF = os.environ.get("DATA_CUTOFF", "2025-12-31")
RAW_DIR = os.environ.get(
    "RAW_DIR", str(Path(__file__).resolve().parent.parent / "data" / "raw"))
CRICSHEET_URL = os.environ.get(
    "CRICSHEET_URL", "https://cricsheet.org/downloads/ipl_json.zip")

# LLM provider: "gemini" | "openai" (openai-compatible: OpenAI, vLLM, Ollama...)
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "gemini")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-flash-lite-latest")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

EMBED_MODEL = os.environ.get("EMBED_MODEL", "BAAI/bge-small-en-v1.5")  # 384-dim

API_HOST = os.environ.get("API_HOST", "0.0.0.0")
API_PORT = int(os.environ.get("API_PORT", "8000"))
MCP_PORT = int(os.environ.get("MCP_PORT", "8765"))
CACHE_TTL_S = int(os.environ.get("CACHE_TTL_S", "3600"))
