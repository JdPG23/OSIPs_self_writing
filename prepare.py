"""
Fixed utilities for OSIP Self-Writing system.
Analogous to autoresearch's prepare.py — DO NOT MODIFY.

Handles:
- Corpus loading (ESA priorities, reference OSIPs)
- Proposal validation (structural checks)
- Results logging
"""

import os
import json
import time
from pathlib import Path
from typing import Optional

from config import (
    CORPUS_DIR,
    OUTPUTS_DIR,
    RESULTS_FILE,
    REQUIRED_SECTIONS,
    MIN_WORDS,
    MAX_WORDS,
)


# ---------------------------------------------------------------------------
# Corpus Loading
# ---------------------------------------------------------------------------

def load_corpus() -> dict:
    """Load all reference materials from the corpus directory.

    Returns dict with keys:
        - 'priorities': str — ESA priorities document
        - 'reference_osips': list[dict] — example OSIPs with metadata
        - 'rejected_patterns': list[str] — common rejection reasons
        - 'domain_glossary': dict — domain-specific terms
    """
    corpus = {
        "priorities": "",
        "reference_osips": [],
        "rejected_patterns": [],
        "domain_glossary": {},
    }

    # Load ESA priorities
    priorities_path = CORPUS_DIR / "esa_priorities.md"
    if priorities_path.exists():
        corpus["priorities"] = priorities_path.read_text(encoding="utf-8")

    # Load reference OSIPs (each as a .md or .json file in corpus/references/)
    refs_dir = CORPUS_DIR / "references"
    if refs_dir.exists():
        for f in sorted(refs_dir.iterdir()):
            if f.suffix == ".json":
                ref = json.loads(f.read_text(encoding="utf-8"))
                corpus["reference_osips"].append(ref)
            elif f.suffix == ".md":
                corpus["reference_osips"].append({
                    "filename": f.name,
                    "content": f.read_text(encoding="utf-8"),
                    "status": "approved",  # assumed if in references/
                })

    # Load rejection patterns
    rejections_path = CORPUS_DIR / "rejection_patterns.md"
    if rejections_path.exists():
        lines = rejections_path.read_text(encoding="utf-8").strip().split("\n")
        corpus["rejected_patterns"] = [l.strip("- ") for l in lines if l.strip()]

    # Load domain glossary
    glossary_path = CORPUS_DIR / "glossary.json"
    if glossary_path.exists():
        corpus["domain_glossary"] = json.loads(
            glossary_path.read_text(encoding="utf-8")
        )

    return corpus


def get_corpus_summary() -> str:
    """Return a short summary of available corpus data."""
    corpus = load_corpus()
    parts = []
    if corpus["priorities"]:
        parts.append(f"ESA priorities: {len(corpus['priorities'])} chars")
    parts.append(f"Reference OSIPs: {len(corpus['reference_osips'])}")
    parts.append(f"Rejection patterns: {len(corpus['rejected_patterns'])}")
    parts.append(f"Glossary terms: {len(corpus['domain_glossary'])}")
    return " | ".join(parts)


# ---------------------------------------------------------------------------
# Proposal Validation (structural, not quality)
# ---------------------------------------------------------------------------

def validate_proposal(proposal_text: str) -> dict:
    """Validate that a proposal meets structural requirements.

    Returns dict:
        - 'valid': bool
        - 'word_count': int
        - 'sections_found': list[str]
        - 'sections_missing': list[str]
        - 'errors': list[str]
    """
    result = {
        "valid": True,
        "word_count": 0,
        "sections_found": [],
        "sections_missing": [],
        "errors": [],
    }

    # Word count check
    words = proposal_text.split()
    result["word_count"] = len(words)
    if len(words) < MIN_WORDS:
        result["errors"].append(
            f"Too short: {len(words)} words (min {MIN_WORDS})"
        )
        result["valid"] = False
    if len(words) > MAX_WORDS * 1.5:  # soft upper limit with margin
        result["errors"].append(
            f"Too long: {len(words)} words (target max {MAX_WORDS})"
        )

    # Section presence check
    text_lower = proposal_text.lower()
    for section in REQUIRED_SECTIONS:
        # Check for section header (with ## or **Section** or Section:)
        variants = [
            f"## {section.lower()}",
            f"**{section.lower()}**",
            f"{section.lower()}:",
            f"# {section.lower()}",
        ]
        found = any(v in text_lower for v in variants)
        if found:
            result["sections_found"].append(section)
        else:
            result["sections_missing"].append(section)

    if result["sections_missing"]:
        result["errors"].append(
            f"Missing sections: {', '.join(result['sections_missing'])}"
        )
        result["valid"] = False

    return result


# ---------------------------------------------------------------------------
# Results Logging
# ---------------------------------------------------------------------------

def init_results_file():
    """Create results.tsv with header if it doesn't exist."""
    if not RESULTS_FILE.exists():
        header = "commit\tscore\talignment\tstructure\tquality\tnovelty\ttokens\tstatus\tdescription\n"
        RESULTS_FILE.write_text(header, encoding="utf-8")


def log_result(
    commit: str,
    score: float,
    alignment: float,
    structure: float,
    quality: float,
    novelty: float,
    tokens: int,
    status: str,
    description: str,
):
    """Append a result row to results.tsv."""
    init_results_file()
    row = f"{commit}\t{score:.1f}\t{alignment:.1f}\t{structure:.1f}\t{quality:.1f}\t{novelty:.1f}\t{tokens}\t{status}\t{description}\n"
    with open(RESULTS_FILE, "a", encoding="utf-8") as f:
        f.write(row)


# ---------------------------------------------------------------------------
# Output Saving
# ---------------------------------------------------------------------------

def save_proposal(
    proposal_text: str,
    topic: str,
    experiment_id: str,
    scores: Optional[dict] = None,
) -> Path:
    """Save a generated proposal to the outputs directory."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"{experiment_id}_{timestamp}.md"
    filepath = OUTPUTS_DIR / filename

    # Build header with metadata
    header = f"---\ntopic: {topic}\nexperiment: {experiment_id}\n"
    if scores:
        for k, v in scores.items():
            header += f"{k}: {v}\n"
    header += f"generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n---\n\n"

    filepath.write_text(header + proposal_text, encoding="utf-8")
    return filepath


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def get_best_score() -> Optional[float]:
    """Read results.tsv and return the best (highest) score so far."""
    if not RESULTS_FILE.exists():
        return None
    lines = RESULTS_FILE.read_text(encoding="utf-8").strip().split("\n")
    if len(lines) <= 1:  # only header
        return None
    best = 0.0
    for line in lines[1:]:
        parts = line.split("\t")
        if len(parts) >= 2 and parts[7] != "error":
            try:
                score = float(parts[1])
                best = max(best, score)
            except (ValueError, IndexError):
                continue
    return best if best > 0 else None
