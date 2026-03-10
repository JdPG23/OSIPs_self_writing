"""
Experiment runner — analogous to running `uv run train.py` in autoresearch.

Usage:
    python run.py --topic "Autonomous GNC for debris removal"
    python run.py --topic "On-board AI for Earth observation" --quick
"""

import argparse
import time
import sys

from config import EXPERIMENT_TIMEOUT
from pipeline import run_pipeline
from scorer import score_proposal, quick_score
from prepare import (
    load_corpus,
    validate_proposal,
    save_proposal,
    get_corpus_summary,
)


def main():
    parser = argparse.ArgumentParser(description="Run one OSIP proposal experiment")
    parser.add_argument(
        "--topic",
        type=str,
        required=True,
        help="OSIP topic/domain for the proposal",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use quick scoring (single LLM call, less accurate)",
    )
    parser.add_argument(
        "--experiment-id",
        type=str,
        default=None,
        help="Experiment ID for file naming (defaults to timestamp)",
    )
    args = parser.parse_args()

    experiment_id = args.experiment_id or time.strftime("%Y%m%d_%H%M%S")

    print(f"=== OSIP Self-Writing Experiment ===")
    print(f"topic:          {args.topic}")
    print(f"experiment_id:  {experiment_id}")
    print(f"scoring_mode:   {'quick' if args.quick else 'full'}")
    print(f"corpus:         {get_corpus_summary()}")
    print()

    # --- Generate Proposal ---
    print("Generating proposal...")
    t0 = time.time()
    proposal_text, gen_tokens = run_pipeline(args.topic)
    gen_time = time.time() - t0
    print(f"Generation complete in {gen_time:.1f}s ({gen_tokens} tokens)")

    # --- Validate Structure ---
    validation = validate_proposal(proposal_text)
    if not validation["valid"]:
        print(f"\nWARNING: Proposal failed structural validation:")
        for err in validation["errors"]:
            print(f"  - {err}")
    print(f"Word count: {validation['word_count']}")
    print(f"Sections found: {len(validation['sections_found'])}/{len(validation['sections_found']) + len(validation['sections_missing'])}")
    print()

    # --- Score Proposal ---
    print("Scoring proposal...")
    corpus = load_corpus()
    t0 = time.time()
    if args.quick:
        scores = quick_score(proposal_text, corpus["priorities"])
    else:
        scores = score_proposal(proposal_text, corpus["priorities"])
    score_time = time.time() - t0
    total_tokens = gen_tokens + scores.tokens_used
    print(f"Scoring complete in {score_time:.1f}s")
    print()

    # --- Save Proposal ---
    filepath = save_proposal(
        proposal_text=proposal_text,
        topic=args.topic,
        experiment_id=experiment_id,
        scores=scores.to_dict(),
    )

    # --- Print Results (parseable format, matching program.md spec) ---
    print("---")
    print(f"topic:            {args.topic}")
    print(f"overall_score:    {scores.overall:.1f}")
    print(f"alignment_score:  {scores.alignment:.1f}")
    print(f"structure_score:  {scores.structure:.1f}")
    print(f"quality_score:    {scores.quality:.1f}")
    print(f"novelty_score:    {scores.novelty:.1f}")
    print(f"token_cost:       ~{total_tokens}")
    print(f"generation_time:  {gen_time:.1f}s")
    print(f"scoring_time:     {score_time:.1f}s")
    print(f"total_time:       {gen_time + score_time:.1f}s")
    print(f"word_count:       {validation['word_count']}")
    print(f"proposal_saved:   {filepath}")

    # --- Print Feedback ---
    if scores.feedback:
        print(f"\n=== Scorer Feedback ===")
        print(scores.feedback)

    return 0


if __name__ == "__main__":
    sys.exit(main())
