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
# 1. Clone and install
git clone <this-repo>
cd OSIPs_self_writing
pip install -r requirements.txt

# 2. Configure API key
cp .env.example .env
# Edit .env → add your ANTHROPIC_API_KEY

# 3. Run one experiment
python run.py --topic "Autonomous collision avoidance for debris removal"

# 4. Start autonomous research loop
# Point Claude Code at this repo, then:
# "Read program.md and kick off a new experiment run"
```

## Project Structure

```
.
├── program.md              # Meta-agent instructions (you edit this)
├── pipeline.py             # Writing pipeline (agent edits this)
├── run.py                  # Experiment runner
├── scorer.py               # Fixed scoring harness (don't touch)
├── prepare.py              # Fixed utilities (don't touch)
├── llm_client.py           # Anthropic/OpenAI client
├── config.py               # Constants
├── corpus/
│   ├── esa_priorities.md   # ESA strategic priorities 2025-2030
│   ├── rejection_patterns.md
│   └── references/         # Example approved OSIPs (.md/.json)
├── templates/
│   └── osip_template.md    # Reference structure
├── outputs/                # Generated proposals
└── results.tsv             # Experiment log (untracked)
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
- **LLM-as-judge**: The scorer uses a separate (cheaper) model to evaluate,
  avoiding self-reinforcing loops.
- **Sleep-compatible**: Start it, go to bed, review results in the morning.

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
