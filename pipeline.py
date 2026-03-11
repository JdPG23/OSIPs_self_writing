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
    PREMIUM_MODEL,
    DEFAULT_TEMPERATURE,
    MAX_TOKENS_PER_CALL,
    USE_RAG,
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
    "pitch": 0.3,         # low for precise distillation
}

# Max reference OSIPs to include in context (plain text fallback)
MAX_REFERENCE_OSIPS = 3

# RAG retrieval settings (used when USE_RAG=True)
RAG_CONTEXT_TOP_K = 8       # chunks for general context
RAG_SIMILAR_OSIPS_TOP_K = 5  # similar implemented OSIPs to retrieve


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
4. NOVELTY (MOST IMPORTANT): Is the innovation clearly differentiated from
   existing funded work? Does the Innovation & Novelty section name specific
   prior art and state what is NEW? Vague novelty claims are the #1 reason
   ESA rejects OSIPs.
5. FEASIBILITY: Is the scope realistic for OSIP budget/timeline?
6. SPECIFICITY: Are claims quantified? Are references concrete?

For each weakness found, provide:
- The specific text that's problematic
- Why it's a weakness
- A concrete suggestion for improvement

Pay EXTRA attention to novelty. If the proposal doesn't explicitly name
a competing approach and explain why this one is better, flag it as a
critical weakness.

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

PITCH_AGENT_SYSTEM = """You are an ESA OSIP idea submission specialist. Your job is
to distill a full technical proposal into a compelling 2-paragraph pitch for the
ideas.esa.int submission form.

CRITICAL CONTEXT: At Phase 1, ESA evaluates ideas on ONE criterion only:
INNOVATION POTENTIAL. Your pitch must make the novelty immediately obvious.

STRUCTURE (follow exactly):
- Paragraph 1 (the hook): The VERY FIRST SENTENCE must state what is new.
  Do not start with the problem — start with the innovation itself.
  Example opening: "We propose X, the first Y that enables Z." Then explain
  the technical mechanism in 2-3 concise sentences.
- Paragraph 2 (differentiation + ESA fit): You MUST name a SPECIFIC existing
  tool, project, or product (not just a standard) and explain both:
  (a) what it CANNOT do (a fundamental limitation, not just "it's slower")
  (b) what YOUR approach enables that is IMPOSSIBLE with the existing one.
  Use language like "fundamentally cannot", "architecturally unable to",
  "there is no mechanism for". End with one sentence naming a specific ESA
  programme this supports.

WRITING STYLE:
- Maximum 200 words total, exactly 2 paragraphs
- Mix short punchy sentences (8-12 words) with one or two longer technical
  ones. Vary the rhythm — do NOT make every sentence the same length.
- NO section headers, NO bullet points, NO tables — plain prose only
- NO budget, NO timeline, NO TRL numbers — save those for Phase 2
- Write to persuade, not to inform. This is a pitch, not an abstract.

Output ONLY the 2 paragraphs. Nothing else."""


# ---------------------------------------------------------------------------
# Pipeline Execution
# ---------------------------------------------------------------------------

def _get_temp(agent_name: str) -> float:
    """Get temperature for a specific agent."""
    t = AGENT_TEMPERATURES.get(agent_name)
    return t if t is not None else DEFAULT_TEMPERATURE


def _build_context_with_rag(topic: str) -> tuple[str, str, str]:
    """Build context using RAG (semantic search over Supabase).

    Returns (priorities_context, similar_osips_text, rag_context).
    """
    from rag import retrieve_context, retrieve_similar_osips

    # Retrieve semantically relevant context for the topic
    rag_context = retrieve_context(
        f"ESA OSIP proposal about: {topic}",
        top_k=RAG_CONTEXT_TOP_K,
    )

    # Retrieve similar implemented OSIPs
    similar = retrieve_similar_osips(topic, top_k=RAG_SIMILAR_OSIPS_TOP_K)
    similar_text = ""
    if similar:
        parts = []
        for osip in similar:
            score_str = f" (relevance: {osip['score']:.2f})" if osip.get('score') else ""
            parts.append(
                f"- **{osip['title']}** — {osip['institution']} ({osip['country']})"
                f" [{osip['domain']}]{score_str}"
            )
        similar_text = "\n".join(parts)

    # Load priorities separately (always useful as full text)
    corpus = load_corpus()
    priorities = corpus["priorities"] if USE_FULL_PRIORITIES else corpus["priorities"][:1000]

    return priorities, similar_text, rag_context


def _build_context_plain(topic: str) -> tuple[str, str, str]:
    """Build context using plain text corpus (fallback when no Supabase)."""
    corpus = load_corpus()
    priorities = corpus["priorities"] if USE_FULL_PRIORITIES else corpus["priorities"][:1000]

    reference_text = ""
    for ref in corpus["reference_osips"][:MAX_REFERENCE_OSIPS]:
        if isinstance(ref, dict) and "content" in ref:
            reference_text += f"\n---\n{ref['content'][:1500]}\n"

    return priorities, reference_text, ""


def run_pipeline(topic: str) -> tuple[str, str, int]:
    """Execute the full proposal writing pipeline.

    Args:
        topic: The OSIP topic/domain to write about.

    Returns:
        (proposal_text, pitch_text, total_tokens_used)
    """
    total_tokens = 0

    # --- Build context (RAG or plain text) ---
    if USE_RAG:
        priorities_text, similar_osips, rag_context = _build_context_with_rag(topic)
    else:
        priorities_text, similar_osips, rag_context = _build_context_plain(topic)

    # --- Step 1: Context Agent ---
    context_prompt = f"""Topic for OSIP proposal: {topic}

## ESA Current Priorities
{priorities_text}

## Similar Implemented OSIPs (from semantic search)
{similar_osips if similar_osips else "No similar OSIPs found."}

## Additional Relevant Context (RAG)
{rag_context if rag_context else "No additional context available."}

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

    # --- Step 6: Pitch Distiller (Phase 1 output) ---
    pitch_prompt = f"""## Full OSIP Proposal (already reviewed and refined)

{proposal}

## Context
Topic: {topic}

Distill this into a 2-paragraph pitch (max 200 words) for the ESA ideas.esa.int
submission form. Focus entirely on innovation and novelty — that is the ONLY
criterion at Phase 1 screening."""

    pitch, tokens = call_llm(
        system_prompt=PITCH_AGENT_SYSTEM,
        user_prompt=pitch_prompt,
        model=GENERATOR_MODEL,
        temperature=_get_temp("pitch"),
        max_tokens=1024,
    )
    total_tokens += tokens

    return proposal, pitch, total_tokens
