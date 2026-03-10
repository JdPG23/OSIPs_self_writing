"""
RAG module — LlamaIndex + Supabase vector store.

Provides semantic search over the OSIP corpus:
- Indexes all corpus documents (priorities, campaigns, references, themes)
- Retrieves the most relevant context for a given OSIP topic
- Used by the pipeline to inject targeted context into agent prompts
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

from config import CORPUS_DIR, GENERATOR_MODEL

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SUPABASE_CONNECTION_STRING = os.getenv("SUPABASE_DB_URL", "")
COLLECTION_NAME = os.getenv("OSIP_COLLECTION_NAME", "osip_corpus")

# Embedding config: "openai" or "huggingface"
EMBEDDING_PROVIDER = os.getenv("OSIP_EMBEDDING_PROVIDER", "openai")

# Embedding dimensions by provider
EMBEDDING_DIMS = {
    "openai": 1536,         # text-embedding-3-small
    "huggingface": 384,     # BAAI/bge-small-en-v1.5
}

# Number of chunks to retrieve per query
TOP_K = int(os.getenv("OSIP_RAG_TOP_K", "8"))


# ---------------------------------------------------------------------------
# LlamaIndex Setup (lazy init)
# ---------------------------------------------------------------------------

_index = None
_retriever = None


def _get_embed_model():
    """Get the embedding model based on config."""
    if EMBEDDING_PROVIDER == "openai":
        from llama_index.embeddings.openai import OpenAIEmbedding
        return OpenAIEmbedding(model="text-embedding-3-small")
    else:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        return HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")


def _get_vector_store():
    """Create SupabaseVectorStore connection."""
    from llama_index.vector_stores.supabase import SupabaseVectorStore

    if not SUPABASE_CONNECTION_STRING:
        raise ValueError(
            "SUPABASE_DB_URL not set. Add it to .env:\n"
            "SUPABASE_DB_URL=postgresql://postgres:<password>@<host>:5432/postgres"
        )

    # Ensure postgresql:// prefix (Supabase sometimes gives postgres://)
    conn_str = SUPABASE_CONNECTION_STRING
    if conn_str.startswith("postgres://"):
        conn_str = conn_str.replace("postgres://", "postgresql://", 1)

    dimension = EMBEDDING_DIMS.get(EMBEDDING_PROVIDER, 1536)

    return SupabaseVectorStore(
        postgres_connection_string=conn_str,
        collection_name=COLLECTION_NAME,
        dimension=dimension,
    )


def _configure_settings():
    """Configure LlamaIndex global settings."""
    from llama_index.core import Settings
    from llama_index.llms.anthropic import Anthropic

    Settings.embed_model = _get_embed_model()
    Settings.llm = Anthropic(model=GENERATOR_MODEL)
    Settings.chunk_size = 512
    Settings.chunk_overlap = 64


# ---------------------------------------------------------------------------
# Document Loading
# ---------------------------------------------------------------------------

def _load_corpus_documents():
    """Load all corpus files as LlamaIndex Documents with rich metadata."""
    from llama_index.core.schema import Document

    documents = []

    # --- ESA Priorities ---
    priorities_path = CORPUS_DIR / "esa_priorities.md"
    if priorities_path.exists():
        text = priorities_path.read_text(encoding="utf-8")
        # Split by sections for better chunking
        sections = text.split("\n## ")
        for i, section in enumerate(sections):
            if not section.strip():
                continue
            section_text = section if i == 0 else f"## {section}"
            documents.append(Document(
                text=section_text,
                metadata={
                    "source": "esa_priorities",
                    "type": "priorities",
                    "section_index": i,
                },
            ))

    # --- Campaigns ---
    campaigns_path = CORPUS_DIR / "osip_campaigns.md"
    if campaigns_path.exists():
        text = campaigns_path.read_text(encoding="utf-8")
        sections = text.split("\n## ")
        for i, section in enumerate(sections):
            if not section.strip():
                continue
            section_text = section if i == 0 else f"## {section}"
            documents.append(Document(
                text=section_text,
                metadata={
                    "source": "osip_campaigns",
                    "type": "campaigns",
                    "section_index": i,
                },
            ))

    # --- Domain Themes ---
    themes_path = CORPUS_DIR / "domain_themes.md"
    if themes_path.exists():
        text = themes_path.read_text(encoding="utf-8")
        sections = text.split("\n### ")
        for i, section in enumerate(sections):
            if not section.strip():
                continue
            section_text = section if i == 0 else f"### {section}"
            documents.append(Document(
                text=section_text,
                metadata={
                    "source": "domain_themes",
                    "type": "themes",
                    "section_index": i,
                },
            ))

    # --- Rejection Patterns ---
    rejections_path = CORPUS_DIR / "rejection_patterns.md"
    if rejections_path.exists():
        text = rejections_path.read_text(encoding="utf-8")
        documents.append(Document(
            text=text,
            metadata={"source": "rejection_patterns", "type": "patterns"},
        ))

    # --- Reference OSIPs (JSON files — individual entries) ---
    refs_dir = CORPUS_DIR / "references"
    if refs_dir.exists():
        for filepath in sorted(refs_dir.iterdir()):
            if filepath.suffix == ".json":
                data = json.loads(filepath.read_text(encoding="utf-8"))
                entries = _extract_entries_from_json(data)
                for entry in entries:
                    documents.append(Document(
                        text=entry["text"],
                        metadata={
                            "source": filepath.name,
                            "type": "reference_osip",
                            "title": entry.get("title", ""),
                            "institution": entry.get("institution", ""),
                            "country": entry.get("country", ""),
                            "domain": entry.get("domain", ""),
                            "year": entry.get("year", ""),
                        },
                    ))
            elif filepath.suffix == ".md":
                text = filepath.read_text(encoding="utf-8")
                # Split MD tables into individual entries
                entries = _extract_entries_from_md_tables(text, filepath.name)
                for entry in entries:
                    documents.append(Document(
                        text=entry["text"],
                        metadata={
                            "source": filepath.name,
                            "type": "reference_osip",
                            "title": entry.get("title", ""),
                            "institution": entry.get("institution", ""),
                            "country": entry.get("country", ""),
                            "domain": entry.get("domain", ""),
                        },
                    ))

    logger.info(f"Loaded {len(documents)} documents from corpus")
    return documents


def _extract_entries_from_json(data: dict) -> list[dict]:
    """Extract individual OSIP entries from a JSON corpus file."""
    entries = []

    # Handle nested structure: {campaigns: [{entries: [...]}]} or {sections: [{entries: [...]}]}
    containers = data.get("campaigns", data.get("sections", []))
    if isinstance(containers, list):
        for container in containers:
            channel = container.get("name", container.get("channel", ""))
            for entry in container.get("entries", []):
                title = entry.get("title", "")
                institution = entry.get("institution", "")
                country = entry.get("country", entry.get("country_name", ""))
                domain = entry.get("domain", "")
                description = entry.get("description", "")
                month = entry.get("month", "")

                text = f"OSIP: {title}\nInstitution: {institution}\nCountry: {country}\nDomain: {domain}\nChannel: {channel}"
                if month:
                    text += f"\nMonth: {month}"
                if description:
                    text += f"\nDescription: {description}"

                entries.append({
                    "text": text,
                    "title": title,
                    "institution": institution,
                    "country": country,
                    "domain": domain,
                    "year": str(data.get("fetched", data.get("fetched_date", ""))),
                })

    return entries


def _extract_entries_from_md_tables(text: str, filename: str) -> list[dict]:
    """Extract individual OSIP entries from markdown table rows."""
    entries = []
    current_section = ""

    for line in text.split("\n"):
        line = line.strip()

        # Track section headers
        if line.startswith("## ") or line.startswith("### "):
            current_section = line.lstrip("#").strip()
            continue

        # Parse table rows (skip headers and separators)
        if line.startswith("|") and not line.startswith("|---") and not line.startswith("| #"):
            parts = [p.strip() for p in line.split("|")[1:-1]]  # strip empty first/last
            if len(parts) >= 4:
                # Expected: | # | Title | Institution | Country | Domain |
                try:
                    idx = parts[0]
                    # Skip if first col is a header word
                    if idx.lower() in ("", "#", "rank", "country"):
                        continue
                    title = parts[1] if len(parts) > 1 else ""
                    institution = parts[2] if len(parts) > 2 else ""
                    country = parts[3] if len(parts) > 3 else ""
                    domain = parts[4] if len(parts) > 4 else ""

                    entry_text = f"OSIP: {title}\nInstitution: {institution}\nCountry: {country}\nDomain: {domain}\nSection: {current_section}"

                    entries.append({
                        "text": entry_text,
                        "title": title,
                        "institution": institution,
                        "country": country,
                        "domain": domain,
                    })
                except (IndexError, ValueError):
                    continue

    return entries


# ---------------------------------------------------------------------------
# Indexing (Ingestion)
# ---------------------------------------------------------------------------

def ingest_corpus(force: bool = False) -> int:
    """Ingest corpus documents into Supabase vector store.

    Args:
        force: If True, re-ingest even if collection exists.

    Returns:
        Number of documents indexed.
    """
    from llama_index.core import VectorStoreIndex, StorageContext

    _configure_settings()

    documents = _load_corpus_documents()
    if not documents:
        logger.warning("No documents found in corpus")
        return 0

    vector_store = _get_vector_store()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    print(f"Indexing {len(documents)} documents into Supabase ({COLLECTION_NAME})...")
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True,
    )
    print(f"Indexing complete: {len(documents)} documents")

    return len(documents)


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def get_retriever(top_k: int = TOP_K):
    """Get a retriever connected to the existing Supabase index."""
    from llama_index.core import VectorStoreIndex

    global _index, _retriever

    if _retriever is not None:
        return _retriever

    _configure_settings()
    vector_store = _get_vector_store()

    _index = VectorStoreIndex.from_vector_store(vector_store)
    _retriever = _index.as_retriever(similarity_top_k=top_k)

    return _retriever


def retrieve_context(query: str, top_k: int = TOP_K) -> str:
    """Retrieve relevant corpus context for a given query/topic.

    Returns a formatted string with the most relevant chunks,
    ready to inject into an agent prompt.
    """
    retriever = get_retriever(top_k=top_k)
    nodes = retriever.retrieve(query)

    if not nodes:
        return "No relevant context found in corpus."

    parts = []
    seen_titles = set()

    for i, node in enumerate(nodes, 1):
        meta = node.metadata
        source = meta.get("source", "unknown")
        doc_type = meta.get("type", "")
        title = meta.get("title", "")

        # Deduplicate by title for reference OSIPs
        if title and title in seen_titles:
            continue
        if title:
            seen_titles.add(title)

        header = f"[{source} | {doc_type}]"
        if title:
            header += f" {title}"

        parts.append(f"### Context {i}: {header}\n{node.text}")

    return "\n\n---\n\n".join(parts)


def retrieve_similar_osips(topic: str, top_k: int = 5) -> list[dict]:
    """Retrieve similar implemented OSIPs for a given topic.

    Returns list of dicts with title, institution, country, domain, text.
    """
    from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter

    retriever = get_retriever(top_k=top_k)

    # We could filter by type="reference_osip" but simple retrieval works well
    nodes = retriever.retrieve(f"OSIP proposal about: {topic}")

    results = []
    seen = set()
    for node in nodes:
        meta = node.metadata
        if meta.get("type") != "reference_osip":
            continue
        title = meta.get("title", "")
        if title in seen:
            continue
        seen.add(title)
        results.append({
            "title": title,
            "institution": meta.get("institution", ""),
            "country": meta.get("country", ""),
            "domain": meta.get("domain", ""),
            "text": node.text,
            "score": node.score,
        })

    return results


# ---------------------------------------------------------------------------
# Query Engine (for direct Q&A over corpus)
# ---------------------------------------------------------------------------

def get_query_engine():
    """Get a LlamaIndex query engine for Q&A over the corpus.

    Uses Claude to synthesize answers from retrieved context.
    """
    from llama_index.core import VectorStoreIndex

    _configure_settings()
    vector_store = _get_vector_store()
    index = VectorStoreIndex.from_vector_store(vector_store)

    return index.as_query_engine(
        similarity_top_k=TOP_K,
        response_mode="tree_summarize",
    )
