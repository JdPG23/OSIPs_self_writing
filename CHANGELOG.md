# Changelog

All notable changes to this project will be documented in this file.
Format based on [Keep a Changelog](https://keepachangelog.com/).

## [0.1.0] - 2026-03-10

### Added
- Initial project structure inspired by karpathy/autoresearch
- Multi-agent writing pipeline: Context → Ideation → Writer → Critic → Reviser
- 4-dimension LLM-as-judge scorer (alignment, structure, quality, novelty)
- Unified LLM client supporting Anthropic and OpenAI APIs
- ESA priorities corpus (2025-2030)
- OSIP rejection patterns reference
- Proposal structural validation
- Experiment runner with parseable output format
- `program.md` with full autonomous experiment loop instructions
