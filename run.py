"""
Experiment runner — analogous to running `uv run train.py` in autoresearch.

Usage:
    python run.py --topic "Autonomous GNC for debris removal"
    python run.py --topic "On-board AI for Earth observation" --quick
    python run.py --topic "Topic" --phase1-only   # pitch only (fast)
    python run.py --topic "Topic" --phase2-only   # proposal only (legacy)
"""

import argparse
import time
import sys

from dotenv import load_dotenv
load_dotenv()

from config import EXPERIMENT_TIMEOUT, USE_RAG, GENERATOR_MODEL, SCORER_MODEL, PREMIUM_MODEL
from pipeline import run_pipeline
from scorer import score_proposal, quick_score, score_pitch
from prepare import (
    load_corpus,
    validate_proposal,
    validate_pitch,
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
        "--premium",
        action="store_true",
        help="Use premium model for generation (higher quality, higher cost)",
    )
    parser.add_argument(
        "--phase1-only",
        action="store_true",
        help="Generate and score only the Phase 1 pitch",
    )
    parser.add_argument(
        "--phase2-only",
        action="store_true",
        help="Generate and score only the Phase 2 proposal (legacy mode)",
    )
    parser.add_argument(
        "--experiment-id",
        type=str,
        default=None,
        help="Experiment ID for file naming (defaults to timestamp)",
    )
    args = parser.parse_args()

    # Override generator model if --premium flag
    if args.premium:
        import config
        config.GENERATOR_MODEL = PREMIUM_MODEL

    experiment_id = args.experiment_id or time.strftime("%Y%m%d_%H%M%S")

    gen_model = PREMIUM_MODEL if args.premium else GENERATOR_MODEL
    print(f"=== OSIP Self-Writing Experiment ===")
    print(f"topic:          {args.topic}")
    print(f"experiment_id:  {experiment_id}")
    print(f"generator:      {gen_model}")
    print(f"scorer:         {SCORER_MODEL}")
    print(f"scoring_mode:   {'quick' if args.quick else 'full'}")
    mode = "phase1-only" if args.phase1_only else ("phase2-only" if args.phase2_only else "dual")
    print(f"output_mode:    {mode}")
    print(f"rag_mode:       {'chroma (local)' if USE_RAG else 'plain text'}")
    print(f"corpus:         {get_corpus_summary()}")
    print()

    # --- Generate Proposal + Pitch ---
    print("Generating proposal...")
    t0 = time.time()
    proposal_text, pitch_text, gen_tokens = run_pipeline(args.topic)
    gen_time = time.time() - t0
    print(f"Generation complete in {gen_time:.1f}s ({gen_tokens} tokens)")

    # --- Validate Phase 2 ---
    if not args.phase1_only:
        validation = validate_proposal(proposal_text)
        if not validation["valid"]:
            print(f"\nWARNING: Proposal failed structural validation:")
            for err in validation["errors"]:
                print(f"  - {err}")
        print(f"Proposal word count: {validation['word_count']}")
        print(f"Sections found: {len(validation['sections_found'])}/{len(validation['sections_found']) + len(validation['sections_missing'])}")

    # --- Validate Phase 1 ---
    pitch_validation = validate_pitch(pitch_text)
    if not pitch_validation["valid"]:
        print(f"\nWARNING: Pitch failed structural validation:")
        for err in pitch_validation["errors"]:
            print(f"  - {err}")
    print(f"Pitch word count: {pitch_validation['word_count']} (paragraphs: {pitch_validation['paragraph_count']})")
    print()

    # --- Score ---
    corpus = load_corpus()
    t0 = time.time()

    # Phase 2 scoring
    proposal_scores = None
    if not args.phase1_only:
        print("Scoring Phase 2 proposal...")
        if args.quick:
            proposal_scores = quick_score(proposal_text, corpus["priorities"])
        else:
            proposal_scores = score_proposal(proposal_text, corpus["priorities"])

    # Phase 1 scoring
    pitch_scores = None
    if not args.phase2_only:
        print("Scoring Phase 1 pitch...")
        pitch_scores = score_pitch(pitch_text, corpus["priorities"])

    score_time = time.time() - t0

    total_tokens = gen_tokens
    if proposal_scores:
        total_tokens += proposal_scores.tokens_used
    if pitch_scores:
        total_tokens += pitch_scores.tokens_used

    print(f"Scoring complete in {score_time:.1f}s")
    print()

    # --- Save Output ---
    filepath = save_proposal(
        proposal_text=proposal_text,
        topic=args.topic,
        experiment_id=experiment_id,
        scores=proposal_scores.to_dict() if proposal_scores else None,
        pitch_text=pitch_text,
        pitch_scores=pitch_scores.to_dict() if pitch_scores else None,
    )

    # --- Print Results ---
    print("---")
    print(f"topic:            {args.topic}")

    if proposal_scores:
        print(f"phase2_score:     {proposal_scores.overall:.1f}/100")
        print(f"  alignment:      {proposal_scores.alignment:.1f}")
        print(f"  structure:      {proposal_scores.structure:.1f}")
        print(f"  quality:        {proposal_scores.quality:.1f}")
        print(f"  novelty:        {proposal_scores.novelty:.1f}")

    if pitch_scores:
        print(f"phase1_score:     {pitch_scores.overall:.1f}/50")
        print(f"  pitch_novelty:  {pitch_scores.novelty:.1f}")
        print(f"  pitch_concise:  {pitch_scores.conciseness:.1f}")
        print(f"  pitch_words:    {pitch_validation['word_count']}")

    print(f"token_cost:       ~{total_tokens}")
    print(f"generation_time:  {gen_time:.1f}s")
    print(f"scoring_time:     {score_time:.1f}s")
    print(f"total_time:       {gen_time + score_time:.1f}s")
    print(f"proposal_saved:   {filepath}")

    # --- Print Feedback ---
    if proposal_scores and proposal_scores.feedback:
        print(f"\n=== Phase 2 Scorer Feedback ===")
        print(proposal_scores.feedback)

    if pitch_scores and pitch_scores.feedback:
        print(f"\n=== Phase 1 Scorer Feedback ===")
        print(pitch_scores.feedback)

    return 0


if __name__ == "__main__":
    sys.exit(main())
