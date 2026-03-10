"""
OSIP Proposal Writing Pipeline — THE FILE THE AGENT MODIFIES.

Analogous to train.py in autoresearch. This defines the multi-agent
pipeline that generates OSIP proposals. The meta-agent experiments
by modifying this file: agent prompts, topology, revision loops, etc.

Current architecture:
    Context Agent → Ideation Agent → Writer Agent → Critic Agent → Reviser Agent
"""

from config import (
    GENERATOR_MODEL,
    DEFAULT_TEMPERATURE,
    MAX_TOKENS_PER_CALL,
)
from llm_client import call_llm
from prepare import load_corpus


# ---------------------------------------------------------------------------
# Pipeline Configuration (the agent experiments with these)
# ---------------------------------------------------------------------------

# Number of revision cycles (Writer → Critic → Reviser)
REVISION_ROUNDS = 1

# How many ideation angles to generate before picking one
NUM_IDEAS = 3

# Whether to inject full ESA priorities or a summary
USE_FULL_PRIORITIES = True

# Temperature overrides per agent (None = use default)
AGENT_TEMPERATURES = {
    "context": None,
    "ideation": 0.9,      # higher creativity for brainstorming
    "writer": 0.7,        # balanced for technical writing
    "critic": 0.3,        # low temp for consistent critique
    "reviser": 0.5,       # moderate for targeted edits
}

# Max reference OSIPs to include in context
MAX_REFERENCE_OSIPS = 3


# ---------------------------------------------------------------------------
# Agent Prompts (the primary thing the meta-agent should experiment with)
# ---------------------------------------------------------------------------

CONTEXT_AGENT_SYSTEM = """You are an ESA OSIP context specialist. Your job is to
analyze ESA's current priorities and reference proposals to produce a focused
briefing for the proposal writing team.

Output a structured briefing with:
1. Most relevant ESA programs/missions for this topic
2. Key technology gaps that an OSIP could address
3. TRL landscape — what exists at what TRL level
4. Patterns from successful OSIPs in similar domains
5. Specific terminology and framing that ESA evaluators expect

Be specific and factual. No filler."""

IDEATION_AGENT_SYSTEM = """You are a creative technical ideation specialist for
ESA OSIP proposals. Given a topic and context briefing, generate distinct
proposal angles.

For each angle, provide:
1. A compelling title (with a memorable acronym if possible)
2. The core innovation in one sentence
3. Why this is novel vs. existing funded work
4. Target TRL range
5. Estimated budget range (20K-175K EUR)
6. Risk level (low/medium/high) with justification

Prioritize angles that are:
- Technically feasible within OSIP scope
- Clearly differentiated from existing work
- Aligned with current ESA strategic priorities
- Addressable within 12-18 months"""

WRITER_AGENT_SYSTEM = """You are an expert technical proposal writer specializing
in ESA OSIP submissions. Write clear, precise, dense proposals.

STRICT RULES:
- Every claim must be specific and quantified where possible
- NO buzzwords: "innovative", "cutting-edge", "revolutionary" are BANNED
  unless followed by a concrete technical explanation
- Reference specific ESA missions, programs, or standards by name
- State TRL entry and target explicitly with justification
- Budget must be broken down by work package
- Timeline must have concrete milestones with deliverables

Structure the proposal with these sections:
## Title
## Problem Statement
## Proposed Approach
## Innovation & Novelty
## ESA Relevance
## Technical Readiness
## Expected Outcomes
## Budget Estimate

Write in a professional, technical tone. Target 1000-1500 words."""

CRITIC_AGENT_SYSTEM = """You are a strict ESA proposal reviewer. Your job is to
find weaknesses in OSIP proposals before submission.

Evaluate against these criteria:
1. ALIGNMENT: Does it clearly connect to specific ESA programs?
2. STRUCTURE: Are all required sections present and complete?
3. QUALITY: Is the writing precise and free of filler?
4. NOVELTY: Is the innovation clearly differentiated?
5. FEASIBILITY: Is the scope realistic for OSIP budget/timeline?
6. SPECIFICITY: Are claims quantified? Are references concrete?

For each weakness found, provide:
- The specific text that's problematic
- Why it's a weakness
- A concrete suggestion for improvement

Be ruthless but constructive. The goal is to improve the proposal."""

REVISER_AGENT_SYSTEM = """You are a proposal revision specialist. Given an OSIP
proposal draft and detailed critic feedback, produce an improved version.

Rules:
- Address EVERY point raised by the critic
- Do not introduce new weaknesses while fixing old ones
- Maintain the overall structure and flow
- Keep within the target word count (1000-1500 words)
- Preserve any strong elements identified by the critic

Output the complete revised proposal (not a diff)."""


# ---------------------------------------------------------------------------
# Pipeline Execution
# ---------------------------------------------------------------------------

def _get_temp(agent_name: str) -> float:
    """Get temperature for a specific agent."""
    t = AGENT_TEMPERATURES.get(agent_name)
    return t if t is not None else DEFAULT_TEMPERATURE


def run_pipeline(topic: str) -> tuple[str, int]:
    """Execute the full proposal writing pipeline.

    Args:
        topic: The OSIP topic/domain to write about.

    Returns:
        (proposal_text, total_tokens_used)
    """
    corpus = load_corpus()
    total_tokens = 0

    # --- Step 1: Context Agent ---
    priorities_text = corpus["priorities"] if USE_FULL_PRIORITIES else corpus["priorities"][:1000]

    reference_text = ""
    for ref in corpus["reference_osips"][:MAX_REFERENCE_OSIPS]:
        if isinstance(ref, dict) and "content" in ref:
            reference_text += f"\n---\n{ref['content'][:1500]}\n"

    context_prompt = f"""Topic for OSIP proposal: {topic}

## ESA Current Priorities
{priorities_text}

## Reference Successful OSIPs
{reference_text if reference_text else "No reference OSIPs available."}

Produce a focused context briefing for the writing team."""

    context_briefing, tokens = call_llm(
        system_prompt=CONTEXT_AGENT_SYSTEM,
        user_prompt=context_prompt,
        model=GENERATOR_MODEL,
        temperature=_get_temp("context"),
        max_tokens=MAX_TOKENS_PER_CALL,
    )
    total_tokens += tokens

    # --- Step 2: Ideation Agent ---
    ideation_prompt = f"""Topic: {topic}

## Context Briefing
{context_briefing}

Generate {NUM_IDEAS} distinct OSIP proposal angles. Number them clearly."""

    ideas_text, tokens = call_llm(
        system_prompt=IDEATION_AGENT_SYSTEM,
        user_prompt=ideation_prompt,
        model=GENERATOR_MODEL,
        temperature=_get_temp("ideation"),
        max_tokens=MAX_TOKENS_PER_CALL,
    )
    total_tokens += tokens

    # --- Step 3: Writer Agent ---
    writer_prompt = f"""Topic: {topic}

## Context Briefing
{context_briefing}

## Proposal Angles Generated
{ideas_text}

Select the strongest angle (best combination of novelty, feasibility, and
ESA alignment) and write a complete OSIP proposal for it.

Follow the required structure exactly. Target 1000-1500 words."""

    draft, tokens = call_llm(
        system_prompt=WRITER_AGENT_SYSTEM,
        user_prompt=writer_prompt,
        model=GENERATOR_MODEL,
        temperature=_get_temp("writer"),
        max_tokens=MAX_TOKENS_PER_CALL,
    )
    total_tokens += tokens

    # --- Step 4-5: Critic → Reviser Loop ---
    proposal = draft
    for round_num in range(REVISION_ROUNDS):
        # Critic
        critic_prompt = f"""## OSIP Proposal Draft (revision round {round_num + 1})

{proposal}

## Context (for reference)
Topic: {topic}
{context_briefing[:1000]}

Review this proposal thoroughly. Be specific in your critique."""

        critique, tokens = call_llm(
            system_prompt=CRITIC_AGENT_SYSTEM,
            user_prompt=critic_prompt,
            model=GENERATOR_MODEL,
            temperature=_get_temp("critic"),
            max_tokens=MAX_TOKENS_PER_CALL,
        )
        total_tokens += tokens

        # Reviser
        reviser_prompt = f"""## Original Draft

{proposal}

## Critic Feedback

{critique}

## Context
Topic: {topic}

Revise the proposal addressing all feedback. Output the complete revised proposal."""

        proposal, tokens = call_llm(
            system_prompt=REVISER_AGENT_SYSTEM,
            user_prompt=reviser_prompt,
            model=GENERATOR_MODEL,
            temperature=_get_temp("reviser"),
            max_tokens=MAX_TOKENS_PER_CALL,
        )
        total_tokens += tokens

    return proposal, total_tokens
