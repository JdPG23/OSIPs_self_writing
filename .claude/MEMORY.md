# OSIP Self-Writing Project Memory

## Overview
Autonomous OSIP proposal optimizer inspired by karpathy/autoresearch.
Meta-agent iterates on pipeline.py to maximize a 4-dimension quality score.

## Architecture
- `program.md` — meta-agent instructions (HUMAN edits)
- `pipeline.py` — multi-agent writing pipeline (AGENT edits)
- `prepare.py` + `scorer.py` — fixed evaluation harness (DO NOT MODIFY)
- `rag.py` — LlamaIndex + ChromaDB local vector store (365 docs indexed)
- `llm_client.py` — OpenRouter API (OpenAI-compatible)
- `config.py`, `run.py` — configuration and experiment runner

## Models (via OpenRouter)
- **Generator**: `google/gemini-3.1-flash-lite-preview` — 3.4s/call, $0.25/$1.50/M
- **Scorer**: `deepseek/deepseek-v3.2` — stricter than Gemini, better calibrated
- **Premium**: `google/gemini-3-flash-preview` — for final polished proposals
- **Embeddings**: HuggingFace `BAAI/bge-small-en-v1.5` (free, local, 384d)
- Seed 2.0 Mini was too slow (110s/call due to reasoning tokens). Avoid.

## RAG
- ChromaDB local (`.chromadb/`), 365 documents from ESA corpus
- Corpus: 336 implemented OSIPs (2023-2025) + priorities + campaigns + themes
- Ingest: `python scripts/ingest.py --force`

## Experiment Results (branch: osip-research/mar10)
- Baseline (Gemini scorer): 86/100 — artificially high, self-reinforcing
- Baseline (DeepSeek scorer): 77/100 — stricter, better for optimization
- Best so far: **82/100** (alignment:19, structure:21, quality:20, novelty:22)
- Key finding: reinforcing Critic on novelty → +4pts novelty (18→22)
- Key finding: 2 revision rounds = same score + more tokens (not worth it)
- Key finding: 5 ideation angles = worse than 3 (too dispersed)
- Key finding: Writer temp 0.7 > 0.5 (+3pts)

## User's Real OSIP Topic
"Integration of MCP into IENAI Space 360 mission analysis SW for AI-driven
automated mission planning, manoeuvre optimization, Phase 0 to Phase D"
- IENAI Space = Spanish company, electric propulsion, has "360" SW
- MCP = Model Context Protocol (Anthropic) for AI-tool integration
- Generated proposal name: M-ADAPT

## User Preferences
- Uses OpenRouter (not direct Anthropic/OpenAI)
- No Supabase (free tier exhausted) → ChromaDB local
- Values cost efficiency — prefers cheap models for autonomous loop
- Working on Windows 11, bash shell, Python 3.12 venv

## Git Structure
- `master` — stable baseline
- `osip-research/<tag>` — experiment branches
- Repo: https://github.com/JdPG23/OSIPs_self_writing
- `results.tsv` is untracked (local experiment log)
