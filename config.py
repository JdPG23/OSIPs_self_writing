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

# Generator model (proposal writing)
# Gemini 3.1 Flash Lite: #34 LM Arena (ELO 1380), $0.25/$1.50 per M tokens
# 3.4s per call, 1M context. Best speed/quality/cost for autonomous loop.
GENERATOR_MODEL = os.getenv("OSIP_GENERATOR_MODEL", "google/gemini-3.1-flash-lite-preview")

# Scorer model (LLM-as-judge — use different model to avoid self-reinforcement)
# DeepSeek V3.2: strong reasoning, $0.25/$0.40 per M tokens
SCORER_MODEL = os.getenv("OSIP_SCORER_MODEL", "deepseek/deepseek-v3.2")

# Premium model (for final polished proposals after pipeline is optimized)
# Gemini 3 Flash: #8 LM Arena (ELO 1473), $0.50/$3.00 per M tokens
PREMIUM_MODEL = os.getenv("OSIP_PREMIUM_MODEL", "google/gemini-3-flash-preview")

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

# --- OSIP Phase 2 Proposal Constraints ---
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

# --- OSIP Phase 1 Pitch Constraints ---
PITCH_MIN_WORDS = 100
PITCH_MAX_WORDS = 350

# --- Experiment Settings ---
EXPERIMENT_TIMEOUT = 300
PROPOSALS_PER_EXPERIMENT = 1
