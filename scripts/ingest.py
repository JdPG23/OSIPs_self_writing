"""
Ingest corpus into Supabase vector store.

Usage:
    python scripts/ingest.py
    python scripts/ingest.py --force        # re-ingest everything
    python scripts/ingest.py --test-query "debris removal"  # ingest + test
"""

import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="Ingest OSIP corpus into Supabase")
    parser.add_argument("--force", action="store_true", help="Force re-ingestion")
    parser.add_argument("--test-query", type=str, help="Run a test retrieval after ingestion")
    parser.add_argument("--stats", action="store_true", help="Show corpus statistics only")
    args = parser.parse_args()

    from prepare import get_corpus_summary
    print(f"Corpus: {get_corpus_summary()}")
    print()

    if args.stats:
        from rag import _load_corpus_documents
        docs = _load_corpus_documents()
        print(f"Total documents to index: {len(docs)}")

        # Count by type
        by_type = {}
        for doc in docs:
            t = doc.metadata.get("type", "unknown")
            by_type[t] = by_type.get(t, 0) + 1
        for t, count in sorted(by_type.items()):
            print(f"  {t}: {count}")
        return

    # --- Ingest ---
    from rag import ingest_corpus
    print("Starting ingestion into Supabase...")
    count = ingest_corpus(force=args.force)
    print(f"\nDone. {count} documents indexed.")

    # --- Test Query ---
    if args.test_query:
        print(f"\n=== Test Retrieval: '{args.test_query}' ===\n")
        from rag import retrieve_context, retrieve_similar_osips

        context = retrieve_context(args.test_query, top_k=5)
        print("--- Retrieved Context ---")
        print(context[:2000])

        print("\n--- Similar OSIPs ---")
        similar = retrieve_similar_osips(args.test_query, top_k=5)
        for osip in similar:
            score = f" (score: {osip['score']:.3f})" if osip.get('score') else ""
            print(f"  - [{osip['country']}] {osip['title']} — {osip['institution']}{score}")


if __name__ == "__main__":
    main()
