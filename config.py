"""
Configuration constants for OSIP Self-Writing system.
Analogous to the fixed constants in autoresearch's prepare.py.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# --- Paths ---
PROJECT_ROOT = Path(__file__).parent
CORPUS_DIR = PROJECT_ROOT / "corpus"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
TEMPLATES_DIR = PROJECT_ROOT / "templates"
RESULTS_FILE = PROJECT_ROOT / "results.tsv"
CHROMA_DIR = PROJECT_ROOT / ".chromadb"  # local vector store

# --- LLM Settings (OpenRouter) ---
# OpenRouter base URL (OpenAI-compatible)
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

# Generator model (proposal writing — needs to be smart)
GENERATOR_MODEL = os.getenv("OSIP_GENERATOR_MODEL", "anthropic/claude-sonnet-4.6")
# Scorer model (LLM-as-judge — needs consistency, can be cheaper)
SCORER_MODEL = os.getenv("OSIP_SCORER_MODEL", "deepseek/deepseek-v3.2")

# --- Embedding Settings ---
# "huggingface" (free, local) or "openai" (paid, needs OPENAI_API_KEY)
EMBEDDING_PROVIDER = os.getenv("OSIP_EMBEDDING_PROVIDER", "huggingface")
EMBEDDING_MODEL = os.getenv("OSIP_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
EMBEDDING_DIM = int(os.getenv("OSIP_EMBEDDING_DIM", "384"))  # 384 for bge-small, 1536 for OpenAI

# --- RAG Settings ---
# Vector store: "chroma" (local, default) or "supabase" (cloud)
VECTOR_STORE = os.getenv("OSIP_VECTOR_STORE", "chroma")
SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL", "")
CHROMA_COLLECTION = os.getenv("OSIP_CHROMA_COLLECTION", "osip_corpus")
RAG_TOP_K = int(os.getenv("OSIP_RAG_TOP_K", "8"))
# Whether RAG is available
USE_RAG = VECTOR_STORE == "chroma" or (VECTOR_STORE == "supabase" and SUPABASE_DB_URL != "")

# --- Generation Constraints ---
MAX_TOKENS_PER_CALL = 4096
MAX_TOKENS_PER_EXPERIMENT = 100_000
DEFAULT_TEMPERATURE = 0.7
SCORER_TEMPERATURE = 0.1

# --- Scoring Weights ---
SCORE_WEIGHTS = {
    "alignment": 25,
    "structure": 25,
    "quality": 25,
    "novelty": 25,
}

# --- OSIP Proposal Constraints ---
MIN_WORDS = 800
MAX_WORDS = 2000
REQUIRED_SECTIONS = [
    "Title",
    "Problem Statement",
    "Proposed Approach",
    "Innovation & Novelty",
    "ESA Relevance",
    "Technical Readiness",
    "Expected Outcomes",
    "Budget Estimate",
]

# --- Experiment Settings ---
EXPERIMENT_TIMEOUT = 300
PROPOSALS_PER_EXPERIMENT = 1
