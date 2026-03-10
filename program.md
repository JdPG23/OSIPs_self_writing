# OSIP Self-Writing: Autonomous Research Program

This is an experiment to have an AI agent autonomously improve the quality of
ESA OSIP (Open Space Innovation Platform) proposals through iterative
experimentation on the writing pipeline.

Inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch):
instead of optimizing a neural network's val_bpb, we optimize a multi-agent
writing pipeline's proposal quality score.

## Setup

To set up a new experiment run, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar10`).
   The branch `osip-research/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b osip-research/<tag>` from main.
3. **Read the in-scope files**:
   - `program.md` — this file. Repository context and agent instructions.
   - `prepare.py` — fixed: corpus loading, scoring harness, evaluation. Do not modify.
   - `scorer.py` — fixed: multi-dimensional quality scoring. Do not modify.
   - `pipeline.py` — the file you modify. Agent prompts, architecture, flow.
4. **Verify corpus exists**: Check that `corpus/` contains at least
   `esa_priorities.md` and some reference OSIPs. If not, tell the human to
   populate it.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row.
6. **Pick a topic**: The human provides an OSIP topic/domain to generate
   proposals about. This stays fixed for the entire run.
7. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation loop.

## The Core Idea

```
YOU (human) edit program.md  →  defines the "research org"
AGENT edits pipeline.py      →  defines how proposals are written
prepare.py + scorer.py       →  fixed evaluation harness (DO NOT MODIFY)
```

The agent iterates on `pipeline.py` to find the combination of prompts,
agent topology, revision strategies, and context selection that produces
the highest-scoring OSIP proposals.

## What Makes a Good OSIP Proposal

ESA evaluates OSIPs on these dimensions (encoded in `scorer.py`):

1. **Alignment with ESA priorities** (0-25 pts)
   - References active ESA programs (Terrae Novae, Space Safety, etc.)
   - Addresses current technology gaps
   - Mentions specific missions or programmatic lines

2. **Technical structure & completeness** (0-25 pts)
   - Clear problem statement
   - Defined TRL entry and target
   - Quantified benefit to ESA missions
   - Validation/verification plan
   - Realistic timeline and budget estimate

3. **Writing quality & precision** (0-25 pts)
   - Dense with specific technical content (no filler/buzzwords)
   - Proper use of domain terminology
   - Concrete numbers and references
   - Clear, concise language

4. **Novelty & differentiation** (0-25 pts)
   - Not duplicating existing funded work
   - References state-of-the-art
   - Clear innovation delta over prior art
   - Identifies unique capability or approach

**Total score: 0-100. The goal is to maximize this score.**

## Experimentation

Each experiment generates a proposal using the current `pipeline.py`
configuration and scores it automatically.

**What you CAN do:**
- Modify `pipeline.py` — this is the ONLY file you edit. Everything is fair
  game: agent prompts, number of agents, revision loops, context selection
  strategy, output formatting, chain-of-thought instructions, few-shot
  examples, temperature settings, etc.

**What you CANNOT do:**
- Modify `prepare.py` or `scorer.py`. They are read-only.
- Change the scoring rubric or evaluation harness.
- Install new packages beyond what's in `requirements.txt`.
- Hardcode proposal content (the pipeline must be general-purpose).

**Simplicity criterion**: All else being equal, simpler is better. A small
score improvement that adds excessive prompt complexity is not worth it.
Removing prompt instructions while maintaining score is a win.

**Cost awareness**: Each experiment uses API tokens. Track approximate
token usage. If a pipeline change doubles token cost for +1 point, probably
not worth it. If it halves cost for -1 point, consider keeping it.

## Output Format

After each experiment, the system prints:

```
---
topic:            <the OSIP topic>
overall_score:    78.5
alignment_score:  22.0
structure_score:  19.5
quality_score:    18.0
novelty_score:    19.0
token_cost:       ~45000
generation_time:  32.4s
pipeline_agents:  4
revision_rounds:  2
```

## Logging Results

Log each experiment to `results.tsv` (tab-separated):

```
commit	score	alignment	structure	quality	novelty	tokens	status	description
```

- commit: git short hash (7 chars)
- score: overall score (e.g. 78.5)
- alignment/structure/quality/novelty: sub-scores
- tokens: approximate token usage
- status: `keep`, `discard`, or `error`
- description: short text of what this experiment tried

Example:

```
commit	score	alignment	structure	quality	novelty	tokens	status	description
a1b2c3d	62.0	15.0	18.0	14.0	15.0	32000	keep	baseline
b2c3d4e	71.5	20.0	19.5	16.0	16.0	38000	keep	added ESA priorities RAG context
c3d4e5f	68.0	18.0	17.0	15.0	18.0	45000	discard	3 revision rounds (diminishing returns)
```

## The Experiment Loop

LOOP FOREVER:

1. Check git state and current baseline score.
2. Modify `pipeline.py` with an experimental idea.
3. `git commit -m "experiment: <description>"`
4. Run: `python run.py --topic "<topic>" > run.log 2>&1`
5. Read results: `grep "^overall_score:" run.log`
6. If grep is empty, there was an error. Check `tail -n 30 run.log`.
7. Record results in `results.tsv` (do NOT commit this file).
8. If score improved → keep the commit, advance the branch.
9. If score is equal or worse → `git reset --hard HEAD~1` to revert.

**Ideas to explore** (non-exhaustive — be creative):
- Vary the system prompts for each agent role
- Add/remove agents (e.g., separate "ESA alignment checker" agent)
- Change the order of agent execution
- Vary the amount of corpus context injected
- Add few-shot examples of good OSIPs to prompts
- Experiment with structured output formats (XML, JSON, sections)
- Try chain-of-thought vs direct generation
- Vary temperature and top_p settings
- Add self-critique loops of varying depth
- Experiment with different context window management strategies

**NEVER STOP**: Once the loop begins, do NOT pause to ask the human.
The human might be asleep. You are autonomous. If you run out of ideas,
re-read the corpus, study patterns in successful OSIPs, try combining
previous near-misses, or try more radical pipeline changes. The loop
runs until the human interrupts you.
