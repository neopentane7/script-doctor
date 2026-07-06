import shutil
import hashlib
import logging
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
import os

logger = logging.getLogger(__name__)

# ── Absolute paths anchored to this file's location ──────────────────────────
_HERE = Path(__file__).parent.parent
DB_DIR = str(_HERE / "db")
DATA_FILE = str(_HERE / "data" / "dialogues.txt")

# ── Chunking parameters (part of the cache fingerprint) ──────────────────────
_CHUNK_SIZE = 450
_CHUNK_OVERLAP = 50
_SEPARATORS = ["\n\n---\n\n", "\n---\n", "\n\n", "\n", " "]

# Marker file storing a fingerprint of the corpus + chunking config. The DB is
# rebuilt whenever this fingerprint changes (e.g. the corpus is edited or the
# chunking parameters are tuned), so retrieval never silently serves a stale index.
_FINGERPRINT_MARKER = _HERE / "db" / ".corpus_fingerprint"


def _corpus_fingerprint() -> str:
    """Hash the corpus content and chunking config into a short cache key."""
    h = hashlib.sha256()
    h.update(f"{_CHUNK_SIZE}:{_CHUNK_OVERLAP}:{_SEPARATORS}".encode("utf-8"))
    try:
        with open(DATA_FILE, "rb") as f:
            h.update(f.read())
    except OSError as exc:
        logger.warning("Could not read corpus for fingerprint: %s", exc)
    return h.hexdigest()

# ── Cached embedding and vectorstore singletons — loaded once per process ──────
_embeddings = None
_vectorstore = None

def get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        logger.info("Loading embedding model (first time only)...")
        _embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    return _embeddings


def create_vectorstore() -> None:
    """Build the ChromaDB vector store from the dialogue corpus.

    The DB is rebuilt only when the corpus content or chunking config changes
    (tracked via a fingerprint marker), so edits to ``data/dialogues.txt`` are
    picked up automatically while unchanged runs reuse the existing index.
    """
    fingerprint = _corpus_fingerprint()

    # Reuse the existing index only if it exists AND its fingerprint matches the
    # current corpus + chunking config.
    if os.path.exists(DB_DIR) and _FINGERPRINT_MARKER.exists():
        try:
            if _FINGERPRINT_MARKER.read_text(encoding="utf-8").strip() == fingerprint:
                logger.info("Vector DB is up to date — skipping rebuild.")
                return
        except OSError:
            pass  # unreadable marker → fall through and rebuild
        logger.info("Corpus or chunking config changed — rebuilding Vector DB...")

    # Rebuild from scratch: remove any stale index first. If the directory is
    # locked (WinError 32) we cannot safely append on top of an old-format index,
    # so surface a clear error rather than silently corrupting the store.
    if os.path.exists(DB_DIR):
        try:
            shutil.rmtree(DB_DIR)
        except OSError as exc:
            raise RuntimeError(
                f"Cannot rebuild the vector DB because {DB_DIR!r} is locked "
                f"({exc}). Close any process holding the ChromaDB files and retry."
            ) from exc

    loader = TextLoader(DATA_FILE, encoding="utf-8")
    documents = loader.load()

    # Prioritize splitting at the scene boundary (---) first,
    # then fall back to paragraph and line breaks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=_CHUNK_SIZE,
        chunk_overlap=_CHUNK_OVERLAP,
        separators=_SEPARATORS,
    )
    docs = splitter.split_documents(documents)

    db = Chroma.from_documents(
        docs,
        embedding=get_embeddings(),
        persist_directory=DB_DIR,
    )

    # Record the fingerprint so the next run can decide whether to reuse this index.
    try:
        os.makedirs(DB_DIR, exist_ok=True)
        _FINGERPRINT_MARKER.write_text(fingerprint, encoding="utf-8")
    except OSError as exc:
        logger.warning("Failed to write corpus fingerprint marker: %s", exc)

    global _vectorstore
    _vectorstore = db
    logger.info("Vector DB created — %d screenplay chunks indexed.", len(docs))


def get_vectorstore() -> Chroma:
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = Chroma(
            persist_directory=DB_DIR,
            embedding_function=get_embeddings(),
        )
    return _vectorstore


def get_similar_dialogues(query: str, k: int = 5) -> list[str]:
    """Return the top-k most relevant dialogue excerpts for the given query."""
    db = get_vectorstore()
    results = db.similarity_search(query, k=k)
    return [doc.page_content.strip() for doc in results]