"""RAG over pgvector (spec §11). Used ONLY for unstructured IPL knowledge
(history, rules, context) — never for numerical statistics.

Ingest: python -m rag.store   (embeds rag/knowledge/*.md into documents table)
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import EMBED_MODEL
from core.db import execute, q

log = logging.getLogger("rag")
_model = None


def _embed(texts: list[str]) -> list[list[float]]:
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        log.info("loading embedding model %s", EMBED_MODEL)
        _model = SentenceTransformer(EMBED_MODEL)
    return _model.encode(texts, normalize_embeddings=True).tolist()


def ingest_knowledge() -> int:
    """Load every rag/knowledge/*.md file. Frontmatter-free: first line = title,
    filename encodes doc_type (history-*.md, rules-*.md, team-*.md...)."""
    knowledge_dir = Path(__file__).resolve().parent / "knowledge"
    files = sorted(knowledge_dir.glob("*.md"))
    execute("DELETE FROM documents")
    for fp in files:
        text = fp.read_text(encoding="utf-8").strip()
        title, _, body = text.partition("\n")
        doc_type = fp.stem.split("-")[0]
        vec = _embed([body])[0]
        execute("""INSERT INTO documents (title, doc_type, source, content, embedding)
                   VALUES (%s, %s, %s, %s, %s)""",
                (title.lstrip("# ").strip(), doc_type, "curated summary", body, str(vec)))
    log.info("ingested %d knowledge documents", len(files))
    return len(files)


def search(query: str, top_k: int = 4) -> list[dict]:
    vec = _embed([query])[0]
    return q("""SELECT title, doc_type, source, content,
                       1 - (embedding <=> %s::vector) AS score
                FROM documents ORDER BY embedding <=> %s::vector LIMIT %s""",
             (str(vec), str(vec), top_k))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ingest_knowledge()
