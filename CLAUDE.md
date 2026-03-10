# CLAUDE.md — Project Intelligence

## What is this project?
Autonomous OSIP (Open Space Innovation Platform) proposal quality optimizer.
Inspired by karpathy/autoresearch: a meta-agent iterates on `pipeline.py`
to find the writing pipeline that produces the highest-scoring ESA proposals.

## Architecture Rules
- **pipeline.py** is the ONLY file the experiment agent modifies
- **prepare.py** and **scorer.py** are READ-ONLY (fixed evaluation harness)
- **program.md** is edited by the human to steer research direction
- Scoring metric: 0-100 across 4 dimensions (alignment, structure, quality, novelty)
- Each experiment = one proposal generation + scoring cycle

## Commands
```bash
# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env   # then fill in API keys + Supabase URL

# Ingest corpus into Supabase (first time / after corpus changes)
python scripts/ingest.py
python scripts/ingest.py --test-query "debris removal"  # ingest + verify
python scripts/ingest.py --stats  # check document counts without ingesting

# Run single experiment
python run.py --topic "Your OSIP topic here"

# Run with quick scoring (faster, less accurate)
python run.py --topic "Topic" --quick

# Check best score so far
grep -v "^commit" results.tsv | sort -t$'\t' -k2 -rn | head -5

# Analyze all results
python scripts/analyze_results.py --table
```

## RAG Architecture
- **LlamaIndex** orchestrates document loading, chunking, and retrieval
- **Supabase** (pgvector) stores embeddings for semantic search
- **OpenAI text-embedding-3-small** generates 1536-dim vectors (or HuggingFace bge-small for free)
- **Claude** (via Anthropic API) generates and scores proposals
- Pipeline auto-detects: if SUPABASE_DB_URL is set → RAG mode, otherwise → plain text fallback

## Branching Convention
- `main` — stable baseline
- `osip-research/<tag>` — experiment branches (e.g. `osip-research/mar10`)
- Never commit `results.tsv` — it's the experiment log, kept local

## Code Style
- Python 3.10+, no type: ignore hacks
- Minimal dependencies (only anthropic + openai SDKs)
- f-strings, pathlib over os.path

## Key Directories
- `corpus/` — reference data (ESA priorities, example OSIPs, rejection patterns)
- `corpus/references/` — approved OSIP examples as .md or .json
- `outputs/` — generated proposals with metadata headers
- `templates/` — structural references

## When Experimenting on pipeline.py
1. Always establish baseline first (run unmodified pipeline.py)
2. Change ONE thing per experiment
3. If score improves → keep commit, advance branch
4. If score regresses → `git reset --hard HEAD~1`
5. Log every result in results.tsv (tab-separated)
6. Track token cost — don't 2x cost for +1 point

## Scoring Calibration
- 0-40: Poor (missing sections, vague, no ESA alignment)
- 40-60: Below average (some structure but weak on specifics)
- 60-75: Good (solid structure, decent alignment, some novelty)
- 75-85: Very good (specific, well-aligned, clear innovation)
- 85-100: Exceptional (rare — publication-quality proposal)
