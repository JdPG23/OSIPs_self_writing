"""
Configuration constants for OSIP Self-Writing system.
Analogous to the fixed constants in autoresearch's prepare.py.
"""

import os
from pathlib import Path

# --- Paths ---
PROJECT_ROOT = Path(__file__).parent
CORPUS_DIR = PROJECT_ROOT / "corpus"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
TEMPLATES_DIR = PROJECT_ROOT / "templates"
RESULTS_FILE = PROJECT_ROOT / "results.tsv"

# --- LLM Settings ---
# Default model for generation agents
GENERATOR_MODEL = os.getenv("OSIP_GENERATOR_MODEL", "claude-sonnet-4-20250514")
# Model for scoring/evaluation (cheaper, faster)
SCORER_MODEL = os.getenv("OSIP_SCORER_MODEL", "claude-haiku-4-5-20251001")
# API provider: "anthropic" or "openai"
API_PROVIDER = os.getenv("OSIP_API_PROVIDER", "anthropic")

# --- Generation Constraints ---
# Max tokens per agent call
MAX_TOKENS_PER_CALL = 4096
# Max total tokens per experiment (cost control)
MAX_TOKENS_PER_EXPERIMENT = 100_000
# Default temperature for generation
DEFAULT_TEMPERATURE = 0.7
# Default temperature for scoring (low = deterministic)
SCORER_TEMPERATURE = 0.1

# --- Scoring Weights ---
# These define the maximum points per dimension (total = 100)
SCORE_WEIGHTS = {
    "alignment": 25,
    "structure": 25,
    "quality": 25,
    "novelty": 25,
}

# --- OSIP Proposal Constraints ---
# Target word count range for proposals
MIN_WORDS = 800
MAX_WORDS = 2000
# Required sections in a valid proposal
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
# Timeout per experiment in seconds
EXPERIMENT_TIMEOUT = 300  # 5 minutes, matching autoresearch
# Number of proposals to generate per experiment for averaging
PROPOSALS_PER_EXPERIMENT = 1  # increase for more stable scores (costs more)
