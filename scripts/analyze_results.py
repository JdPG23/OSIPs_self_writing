"""
Analyze experiment results from results.tsv.
Generates summary statistics and identifies patterns.

Usage:
    python scripts/analyze_results.py
    python scripts/analyze_results.py --plot
"""

import argparse
import sys
from pathlib import Path

RESULTS_FILE = Path(__file__).parent.parent / "results.tsv"


def load_results() -> list[dict]:
    """Load results.tsv into list of dicts."""
    if not RESULTS_FILE.exists():
        return []

    lines = RESULTS_FILE.read_text(encoding="utf-8").strip().split("\n")
    if len(lines) <= 1:
        return []

    headers = lines[0].split("\t")
    results = []
    for line in lines[1:]:
        parts = line.split("\t")
        if len(parts) >= len(headers):
            row = dict(zip(headers, parts))
            # Parse numeric fields
            for key in ("score", "alignment", "structure", "quality", "novelty", "tokens"):
                try:
                    row[key] = float(row.get(key, 0))
                except ValueError:
                    row[key] = 0.0
            results.append(row)

    return results


def print_summary(results: list[dict]):
    """Print experiment summary."""
    if not results:
        print("No results found.")
        return

    kept = [r for r in results if r.get("status") == "keep"]
    discarded = [r for r in results if r.get("status") == "discard"]
    errors = [r for r in results if r.get("status") == "error"]

    print(f"=== Experiment Summary ===")
    print(f"Total experiments:  {len(results)}")
    print(f"  Kept:             {len(kept)}")
    print(f"  Discarded:        {len(discarded)}")
    print(f"  Errors:           {len(errors)}")
    print()

    if kept:
        scores = [r["score"] for r in kept]
        print(f"Score (kept only):")
        print(f"  Best:    {max(scores):.1f}")
        print(f"  Worst:   {min(scores):.1f}")
        print(f"  Mean:    {sum(scores)/len(scores):.1f}")
        print()

        # Dimension breakdown for best
        best = max(kept, key=lambda r: r["score"])
        print(f"Best experiment: {best.get('commit', '?')} — {best.get('description', '?')}")
        print(f"  Alignment: {best['alignment']:.1f}/25")
        print(f"  Structure: {best['structure']:.1f}/25")
        print(f"  Quality:   {best['quality']:.1f}/25")
        print(f"  Novelty:   {best['novelty']:.1f}/25")
        print(f"  Total:     {best['score']:.1f}/100")
        print()

        # Weakest dimension across all kept
        dims = ["alignment", "structure", "quality", "novelty"]
        avg_dims = {d: sum(r[d] for r in kept) / len(kept) for d in dims}
        weakest = min(avg_dims, key=avg_dims.get)
        print(f"Weakest dimension (avg): {weakest} ({avg_dims[weakest]:.1f}/25)")
        print(f"Strongest dimension (avg): {max(avg_dims, key=avg_dims.get)} ({max(avg_dims.values()):.1f}/25)")
        print()

    # Token efficiency
    all_scores = [r["score"] for r in results if r["score"] > 0]
    all_tokens = [r["tokens"] for r in results if r["tokens"] > 0]
    if all_tokens:
        print(f"Token usage:")
        print(f"  Total:   {sum(all_tokens):,.0f}")
        print(f"  Mean:    {sum(all_tokens)/len(all_tokens):,.0f} per experiment")
        if all_scores:
            print(f"  Efficiency: {max(all_scores) / (sum(all_tokens)/1000):.2f} pts per 1K tokens")

    # Progress over time
    if len(kept) >= 2:
        print(f"\nProgress:")
        print(f"  First kept score: {kept[0]['score']:.1f}")
        print(f"  Last kept score:  {kept[-1]['score']:.1f}")
        print(f"  Delta:            {kept[-1]['score'] - kept[0]['score']:+.1f}")


def print_table(results: list[dict]):
    """Print results as formatted table."""
    if not results:
        return

    print(f"\n{'Commit':<10} {'Score':>6} {'Align':>6} {'Struct':>6} {'Qual':>6} {'Novel':>6} {'Status':<8} Description")
    print("-" * 90)
    for r in results:
        print(f"{r.get('commit','?'):<10} {r['score']:>6.1f} {r['alignment']:>6.1f} {r['structure']:>6.1f} {r['quality']:>6.1f} {r['novelty']:>6.1f} {r.get('status','?'):<8} {r.get('description','')[:30]}")


def main():
    parser = argparse.ArgumentParser(description="Analyze OSIP experiment results")
    parser.add_argument("--table", action="store_true", help="Show full results table")
    args = parser.parse_args()

    results = load_results()
    print_summary(results)

    if args.table:
        print_table(results)


if __name__ == "__main__":
    main()
