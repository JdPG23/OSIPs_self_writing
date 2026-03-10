# OSIP Self-Writing

> *The year is 2027. Proposal writing used to be done by researchers hunched over
> laptops at 2am before deadline, fueled by coffee and existential dread. That era
> is fading. The proposals now write themselves — iterating through the night while
> you sleep, each draft sharper than the last. You wake up to a scored log of 50
> experiments and a proposal that reads better than anything you'd have written
> by hand. This repo is how it started.* — March 2026

An autonomous system that iteratively improves ESA OSIP proposal quality using
the [autoresearch](https://github.com/karpathy/autoresearch) pattern: a
meta-agent modifies the writing pipeline, generates a proposal, scores it,
keeps improvements, discards regressions, and repeats — indefinitely.

## How It Works

```
┌─────────────────────────────────────────────────────┐
│  YOU (human)                                        │
│  Edit program.md — define the research org          │
│  Populate corpus/ — feed it domain knowledge        │
│  Sleep.                                             │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│  META-AGENT (autonomous loop)                       │
│                                                     │
│  1. Modify pipeline.py (prompts, agents, flow)      │
│  2. Run pipeline → generate OSIP proposal           │
│  3. Score proposal (0-100, 4 dimensions)            │
│  4. Score improved? → git commit (keep)             │
│     Score worse?   → git reset  (discard)           │
│  5. GOTO 1                                          │
└─────────────────────────────────────────────────────┘
```

### The Three Files That Matter

| File | Who edits | What it does |
|---|---|---|
| `program.md` | **You** | Instructions for the meta-agent. The "org code." |
| `pipeline.py` | **The agent** | Multi-agent writing pipeline. Prompts, topology, flow. |
| `scorer.py` | **Nobody** | Fixed evaluation harness. The ground truth metric. |

### Scoring (the metric)

Every proposal is scored on 4 dimensions (0-25 each, total 0-100):

| Dimension | What it measures |
|---|---|
| **Alignment** | Connection to specific ESA programs, missions, priorities |
| **Structure** | Completeness: TRL, budget, milestones, validation plan |
| **Quality** | Technical precision, no filler, quantified claims |
| **Novelty** | Clear innovation delta over existing funded work |

### The Pipeline (baseline)

```
Context Agent    → Reads ESA priorities + reference OSIPs
     ↓
Ideation Agent   → Generates 3 distinct proposal angles
     ↓
Writer Agent     → Writes full proposal from best angle
     ↓
Critic Agent     → Finds weaknesses and gaps
     ↓
Reviser Agent    → Addresses all critique points
     ↓
Output           → Scored, saved, logged
```

The meta-agent experiments with: prompts, agent count, revision depth,
temperature, context selection, output format, few-shot examples, etc.

## Quick Start

```bash
# 1. Clone and checkout experiment branch
git clone https://github.com/JdPG23/OSIPs_self_writing.git
cd OSIPs_self_writing
git checkout osip-research/mar10    # latest experiment branch

# 2. Create Python virtual environment
python -m venv .venv
source .venv/Scripts/activate       # Windows (Git Bash)
# source .venv/bin/activate         # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure API key (OpenRouter — single key for all models)
cp .env.example .env
# Edit .env → add your OPENROUTER_API_KEY (get one at https://openrouter.ai)

# 5. Build local vector store (ChromaDB, first time only, ~2 min)
python scripts/ingest.py
python scripts/ingest.py --test-query "debris removal"   # verify RAG works
python scripts/ingest.py --stats                          # check doc counts

# 6. Run one experiment
python run.py --topic "Autonomous collision avoidance for debris removal"
python run.py --topic "Topic" --quick      # faster scoring (single LLM call)
python run.py --topic "Topic" --premium    # Gemini 3 Flash (higher quality)

# 7. Start autonomous research loop (via Claude Code)
# Point Claude Code at this repo, then:
# "Read program.md and kick off a new experiment run"
```

## Continuing on Another Machine

If you cloned this repo on a new PC:

```bash
git checkout osip-research/mar10
python -m venv .venv && source .venv/Scripts/activate
pip install -r requirements.txt
cp .env.example .env           # add your OPENROUTER_API_KEY
python scripts/ingest.py       # rebuild ChromaDB (~2 min, corpus is in git)
```

Everything else is already in git: `results.tsv` (experiment log), `outputs/` (proposals),
`.claude/MEMORY.md` (session memory), `CLAUDE.md` (project intelligence).
The only things to rebuild locally are `.venv/` and `.chromadb/`.

## Project Structure

```
.
├── program.md              # Meta-agent instructions (you edit this)
├── pipeline.py             # Writing pipeline (agent edits this)
├── run.py                  # Experiment runner
├── scorer.py               # Fixed scoring harness (don't touch)
├── prepare.py              # Fixed utilities (don't touch)
├── llm_client.py           # OpenRouter client (OpenAI-compatible)
├── rag.py                  # LlamaIndex + ChromaDB RAG module
├── config.py               # Constants and model selection
├── .claude/MEMORY.md       # Claude Code session memory
├── corpus/
│   ├── esa_priorities.md   # ESA strategic priorities + GSTP 2026-2028
│   ├── osip_campaigns.md   # Active ESA campaigns and channels
│   ├── domain_themes.md    # Thematic analysis of 336 OSIPs
│   ├── rejection_patterns.md
│   └── references/         # 336 approved OSIPs (.json + .md)
├── scripts/
│   ├── ingest.py           # Corpus → ChromaDB ingestion
│   ├── analyze_results.py  # Results analysis and reporting
│   └── scrape_osips.py     # ESA OSIP scraper
├── outputs/                # Generated proposals (YAML frontmatter)
└── results.tsv             # Experiment log (tab-separated)
```

## Populating the Corpus

The system improves faster with better reference data:

1. **Reference OSIPs**: Add approved OSIP summaries to `corpus/references/`
   - Source: [ESA Implemented OSIP Ideas](https://www.esa.int/Enabling_Support/Space_Engineering_Technology/Shaping_the_Future/Implemented_OSIP_ideas)
   - Format: `.md` files with title, abstract, institution, outcome
2. **ESA Priorities**: Keep `corpus/esa_priorities.md` updated from ESA publications
3. **Rejection Patterns**: Add failure modes to `corpus/rejection_patterns.md`

## Design Philosophy

- **Fixed time, variable quality**: Like autoresearch's 5-min training budget,
  each experiment has bounded cost. The pipeline improves within that budget.
- **Single file to modify**: The agent only touches `pipeline.py`. Clean diffs,
  easy rollback, reviewable experiments.
- **LLM-as-judge**: The scorer uses a different model (DeepSeek V3.2) from
  the generator (Gemini Flash Lite) to avoid self-reinforcing score inflation.
- **Sleep-compatible**: Start it, go to bed, review results in the morning.
- **Cost-efficient**: Uses OpenRouter with cheap models (~$0.25/M tokens) for
  the autonomous loop. Premium models available via `--premium` flag.

## Adapting to Other Domains

The pattern generalizes beyond ESA OSIPs. Fork and change:
1. `corpus/` → your domain's reference material
2. `scorer.py` → your evaluation rubric
3. `pipeline.py` → your document generation pipeline
4. `program.md` → your research instructions

Examples: grant proposals, research papers, technical reports, patent claims.

## Inspired By

- [karpathy/autoresearch](https://github.com/karpathy/autoresearch) — the original pattern
- [ESA OSIP](https://ideas.esa.int) — the target platform

## License

MIT
