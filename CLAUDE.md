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

## Setup (from scratch on a new machine)
```bash
# 1. Clone and checkout experiment branch
git clone https://github.com/JdPG23/OSIPs_self_writing.git
cd OSIPs_self_writing
git checkout osip-research/mar10

# 2. Create Python environment
python -m venv .venv
source .venv/Scripts/activate   # Windows (Git Bash)
# source .venv/bin/activate     # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env → add your OPENROUTER_API_KEY

# 5. Ingest corpus into local ChromaDB (first time only, ~2 min)
python scripts/ingest.py
python scripts/ingest.py --test-query "debris removal"  # verify it works

# 6. Run a single experiment
python run.py --topic "Your OSIP topic here"
python run.py --topic "Topic" --quick    # quick scoring (faster)
python run.py --topic "Topic" --premium  # Gemini 3 Flash (higher quality)
```

## Current Models (via OpenRouter)
- **Generator**: `google/gemini-3.1-flash-lite-preview` — 3.4s/call, $0.25/$1.50/M tokens
- **Scorer**: `deepseek/deepseek-v3.2` — strict, well-calibrated for scoring
- **Premium**: `google/gemini-3-flash-preview` — for final polished proposals
- **Embeddings**: HuggingFace `BAAI/bge-small-en-v1.5` (free, local, 384d)
- All LLM calls go through OpenRouter (single API key). See `llm_client.py`.
- Avoid Seed 2.0 Mini — 110s/call due to mandatory reasoning tokens.

## RAG Architecture
- **LlamaIndex** orchestrates document loading, chunking, and retrieval
- **ChromaDB** (local, `.chromadb/`) stores embeddings — no cloud needed
- **HuggingFace bge-small-en-v1.5** generates 384-dim vectors (free, CPU)
- 365 documents indexed from 336 implemented ESA OSIPs + priorities + campaigns
- Pipeline auto-detects: if ChromaDB is available → RAG mode, otherwise → plain text fallback

## User's OSIP Topic
"Integration of MCP (Model Context Protocol) into IENAI Space 360 mission
analysis software for AI-driven automated mission planning, manoeuvre
optimization, and end-to-end mission design automation from Phase 0 to Phase D"

## Experiment Progress (branch: osip-research/mar10)
- Baseline (DeepSeek scorer): 77/100
- **Best so far: 82/100** (alignment:19, structure:21, quality:20, novelty:22)
- Key findings:
  - Reinforcing Critic on novelty → +4pts novelty (18→22)
  - 2 revision rounds = same score + more tokens (not worth it)
  - 5 ideation angles = worse than 3 (too dispersed)
  - Writer temp 0.7 > 0.5 (+3pts)
- Weakest dimensions: **alignment (19/25)** and **quality (20/25)**
- Next experiments to try: few-shot examples, ESA Alignment Checker agent,
  structured XML output, chain-of-thought, domain glossary injection

## Branching Convention
- `master` — stable baseline
- `osip-research/<tag>` — experiment branches (e.g. `osip-research/mar10`)
- `results.tsv` tracks all experiment results (tab-separated)

## Code Style
- Python 3.10+, no type: ignore hacks
- f-strings, pathlib over os.path

## Key Directories
- `corpus/` — reference data (ESA priorities, example OSIPs, rejection patterns)
- `corpus/references/` — 336 approved OSIP examples as .md or .json
- `outputs/` — generated proposals with YAML frontmatter metadata
- `.claude/MEMORY.md` — Claude Code session memory for continuity
- `.chromadb/` — local vector store (gitignored, rebuild with `scripts/ingest.py`)

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
