"""
Fixed scoring harness for OSIP proposals.
Analogous to the evaluate_bpb function in autoresearch — DO NOT MODIFY.

Uses LLM-as-judge for each scoring dimension, with structured rubrics
to ensure consistency and reproducibility.
"""

import json
import re
from dataclasses import dataclass, asdict
from typing import Optional

from config import SCORE_WEIGHTS, SCORER_MODEL, SCORER_TEMPERATURE, MAX_TOKENS_PER_CALL
from llm_client import call_llm


# ---------------------------------------------------------------------------
# Score Data Structure
# ---------------------------------------------------------------------------

@dataclass
class ProposalScore:
    alignment: float = 0.0      # ESA priorities alignment (0-25)
    structure: float = 0.0      # Technical structure & completeness (0-25)
    quality: float = 0.0        # Writing quality & precision (0-25)
    novelty: float = 0.0        # Novelty & differentiation (0-25)
    overall: float = 0.0        # Sum of above (0-100)
    feedback: str = ""          # Detailed feedback for pipeline improvement
    tokens_used: int = 0        # Tokens consumed by scoring

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Scoring Rubrics (injected into LLM-as-judge prompts)
# ---------------------------------------------------------------------------

ALIGNMENT_RUBRIC = """
Score the proposal's alignment with ESA priorities on a scale of 0-25.

Scoring guide:
- 0-5:   No mention of ESA programs, missions, or strategic priorities
- 6-10:  Vague references to space activities without specific ESA context
- 11-15: References some ESA programs but connection is superficial
- 16-20: Clear connection to specific ESA programs/missions with rationale
- 21-25: Deep alignment with current ESA strategic priorities, references
         specific programmatic lines (Terrae Novae, Space Safety, etc.),
         addresses known technology gaps, and names target missions

Key ESA priorities to check against:
- Space Safety Programme (SSA, space debris, planetary defense)
- Terrae Novae (Moon, Mars exploration)
- Earth observation and climate monitoring
- Telecommunications and navigation (Galileo, EGNOS)
- In-orbit servicing and manufacturing
- Green/sustainable space operations
- Digital transformation and AI in space operations
- Quantum technologies for space
"""

STRUCTURE_RUBRIC = """
Score the proposal's technical structure and completeness on a scale of 0-25.

Scoring guide:
- 0-5:   Missing most required sections, no clear structure
- 6-10:  Has some sections but key elements missing (no TRL, no budget)
- 11-15: All major sections present but thin on detail
- 16-20: Well-structured with clear problem, approach, TRL roadmap, budget
- 21-25: Exemplary structure with quantified objectives, clear milestones,
         risk assessment, concrete deliverables, and realistic budget

Required elements to check:
- [ ] Clear problem statement with quantified impact
- [ ] TRL entry level and target TRL explicitly stated
- [ ] Proposed approach with technical specifics
- [ ] Innovation/novelty clearly articulated
- [ ] ESA relevance with specific mission/program linkage
- [ ] Validation/verification plan
- [ ] Expected outcomes with measurable KPIs
- [ ] Budget estimate (should be in 20K-175K EUR range for OSIP)
- [ ] Timeline with milestones
"""

QUALITY_RUBRIC = """
Score the proposal's writing quality and technical precision on a scale of 0-25.

Scoring guide:
- 0-5:   Vague, filled with buzzwords and generic statements
- 6-10:  Some technical content but mixed with filler language
- 11-15: Reasonably precise but could be more specific in places
- 16-20: Dense with technical specifics, minimal filler, proper terminology
- 21-25: Publication-quality writing with precise technical claims, specific
         numbers/references, zero buzzword filler, clear and concise

Red flags (subtract points for):
- "innovative solution" without specifying the innovation
- "cutting-edge technology" without naming the technology
- "significant improvement" without quantifying
- Excessive acronyms without definitions
- Repetitive statements that add no information
- Marketing language instead of technical language

Green flags (add points for):
- Specific performance numbers with units
- References to published work (papers, standards)
- Clear technical trade-off analysis
- Concrete comparison with state-of-the-art
"""

NOVELTY_RUBRIC = """
Score the proposal's novelty and differentiation on a scale of 0-25.

Scoring guide:
- 0-5:   Describes well-known approaches with no innovation
- 6-10:  Minor incremental improvement over existing work
- 11-15: Clear novel element but limited in scope or impact
- 16-20: Significant innovation with clear differentiation from prior art
- 21-25: Highly novel approach that opens new possibilities, clearly
         differentiated from all existing funded work, with potential
         for high impact and follow-on research

Check for:
- Does it reference and differentiate from state-of-the-art?
- Is the core innovation clearly stated (not buried)?
- Would this be fundable vs. existing ESA projects?
- Is the novelty in the approach, application, or both?
- Does it enable capabilities that don't currently exist?
"""


# ---------------------------------------------------------------------------
# Scoring Functions
# ---------------------------------------------------------------------------

def _score_dimension(
    proposal: str,
    dimension: str,
    rubric: str,
    esa_priorities: str = "",
) -> tuple[float, str, int]:
    """Score a single dimension using LLM-as-judge.

    Returns (score, feedback, tokens_used).
    """
    system_prompt = f"""You are an expert ESA proposal evaluator. You are scoring
the '{dimension}' dimension of an OSIP proposal.

You must respond with ONLY a JSON object (no markdown, no extra text):
{{"score": <float 0-25>, "feedback": "<specific feedback for this dimension>"}}

Be strict and calibrated. A score of 25 is exceptional and rare.
Average proposals score 12-15. Good proposals score 16-20."""

    user_prompt = f"""## Scoring Rubric

{rubric}

## ESA Current Priorities Context

{esa_priorities[:2000] if esa_priorities else "Not provided — score based on general ESA knowledge."}

## Proposal to Evaluate

{proposal}

Score this proposal on the '{dimension}' dimension (0-25). Return JSON only."""

    response, tokens = call_llm(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model=SCORER_MODEL,
        temperature=SCORER_TEMPERATURE,
        max_tokens=500,
    )

    # Parse JSON response
    try:
        # Try to extract JSON from response
        json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            score = float(data.get("score", 0))
            feedback = data.get("feedback", "")
            # Clamp score to valid range
            score = max(0.0, min(25.0, score))
            return score, feedback, tokens
    except (json.JSONDecodeError, ValueError):
        pass

    return 0.0, f"Failed to parse scorer response: {response[:200]}", tokens


def score_proposal(
    proposal: str,
    esa_priorities: str = "",
) -> ProposalScore:
    """Score a proposal across all dimensions.

    This is the main evaluation function — analogous to evaluate_bpb
    in autoresearch. Returns a ProposalScore with breakdown.
    """
    result = ProposalScore()
    total_tokens = 0
    feedback_parts = []

    # Score each dimension
    dimensions = [
        ("alignment", ALIGNMENT_RUBRIC),
        ("structure", STRUCTURE_RUBRIC),
        ("quality", QUALITY_RUBRIC),
        ("novelty", NOVELTY_RUBRIC),
    ]

    for dim_name, rubric in dimensions:
        score, feedback, tokens = _score_dimension(
            proposal=proposal,
            dimension=dim_name,
            rubric=rubric,
            esa_priorities=esa_priorities,
        )
        setattr(result, dim_name, score)
        total_tokens += tokens
        feedback_parts.append(f"**{dim_name}** ({score:.1f}/25): {feedback}")

    result.overall = result.alignment + result.structure + result.quality + result.novelty
    result.feedback = "\n\n".join(feedback_parts)
    result.tokens_used = total_tokens

    return result


# ---------------------------------------------------------------------------
# Quick Score (for faster iteration — single call, less accurate)
# ---------------------------------------------------------------------------

def quick_score(proposal: str, esa_priorities: str = "") -> ProposalScore:
    """Score all dimensions in a single LLM call. Faster but less accurate."""

    system_prompt = """You are an expert ESA OSIP proposal evaluator.
Score the proposal on 4 dimensions, each 0-25 points.

Respond with ONLY a JSON object:
{
  "alignment": <0-25>,
  "structure": <0-25>,
  "quality": <0-25>,
  "novelty": <0-25>,
  "feedback": "<brief overall feedback>"
}

Be strict. Average proposals: 12-15 per dimension. Good: 16-20. Exceptional (rare): 21-25."""

    user_prompt = f"""## ESA Priorities
{esa_priorities[:2000] if esa_priorities else "Use general ESA knowledge."}

## Proposal
{proposal}

Score this OSIP proposal. Return JSON only."""

    response, tokens = call_llm(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model=SCORER_MODEL,
        temperature=SCORER_TEMPERATURE,
        max_tokens=500,
    )

    result = ProposalScore(tokens_used=tokens)

    try:
        json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            result.alignment = max(0, min(25, float(data.get("alignment", 0))))
            result.structure = max(0, min(25, float(data.get("structure", 0))))
            result.quality = max(0, min(25, float(data.get("quality", 0))))
            result.novelty = max(0, min(25, float(data.get("novelty", 0))))
            result.overall = result.alignment + result.structure + result.quality + result.novelty
            result.feedback = data.get("feedback", "")
    except (json.JSONDecodeError, ValueError):
        result.feedback = f"Parse error: {response[:200]}"

    return result
