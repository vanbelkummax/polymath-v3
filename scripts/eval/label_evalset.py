#!/usr/bin/env python3
"""
Interactive labeling CLI for building evaluation sets.

Usage:
    python scripts/eval/label_evalset.py --query "spatial transcriptomics methods"
    python scripts/eval/label_evalset.py --query "H&E gene expression" --output data/evalsets/custom.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

# Add lib to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lib.search.hybrid_search import HybridSearcher


def label_query(
    query: str,
    searcher: HybridSearcher,
    n_show: int = 20,
) -> dict:
    """
    Interactive labeling for a single query.

    Returns evalset entry with relevant passage IDs.
    """
    print(f"\n{'='*70}")
    print(f"QUERY: {query}")
    print("="*70)

    # Run search
    response = searcher.search(query, n=n_show, rerank=True)

    if not response.results:
        print("No results found.")
        return None

    # Show results for labeling
    print(f"\nFound {len(response.results)} results. Mark relevant with 'y', irrelevant with 'n'.")
    print("Enter 'q' to finish, 's' to skip query.\n")

    relevant_ids = []

    for i, result in enumerate(response.results):
        print(f"\n[{i+1}/{len(response.results)}] Score: {result.score:.4f}")
        print(f"Title: {result.title}")
        if result.section:
            print(f"Section: {result.section}")
        print("-" * 50)
        print(result.passage_text[:500])
        print("-" * 50)
        print(f"Passage ID: {result.passage_id}")

        while True:
            response_input = input("\nRelevant? [y/n/q/s]: ").strip().lower()

            if response_input == "y":
                relevant_ids.append(str(result.passage_id))
                print("  -> Marked RELEVANT")
                break
            elif response_input == "n":
                print("  -> Marked NOT relevant")
                break
            elif response_input == "q":
                print("\nFinishing...")
                break
            elif response_input == "s":
                print("\nSkipping query...")
                return None
            else:
                print("  Invalid input. Enter y, n, q, or s.")

        if response_input == "q":
            break

    # Get tags
    tags_input = input("\nTags (comma-separated, or empty): ").strip()
    tags = [t.strip() for t in tags_input.split(",") if t.strip()]

    # Get notes
    notes = input("Notes (optional): ").strip()

    entry = {
        "query": query,
        "relevant_passage_ids": relevant_ids,
        "tags": tags,
        "notes": notes,
    }

    print(f"\n{len(relevant_ids)} passages marked relevant.")
    return entry


def main():
    parser = argparse.ArgumentParser(description="Interactive evalset labeling")
    parser.add_argument(
        "--query",
        type=str,
        help="Single query to label",
    )
    parser.add_argument(
        "--queries-file",
        type=Path,
        help="File with one query per line",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/evalsets/labeled.jsonl"),
        help="Output JSONL file",
    )
    parser.add_argument(
        "-n", "--n-show",
        type=int,
        default=15,
        help="Number of results to show per query",
    )

    args = parser.parse_args()

    if not args.query and not args.queries_file:
        parser.error("Either --query or --queries-file required")

    # Get queries
    queries = []
    if args.query:
        queries = [args.query]
    elif args.queries_file:
        with open(args.queries_file) as f:
            queries = [line.strip() for line in f if line.strip()]

    # Initialize searcher
    print("Initializing searcher...")
    searcher = HybridSearcher()

    # Label each query
    entries = []
    for query in queries:
        entry = label_query(query, searcher, n_show=args.n_show)
        if entry:
            entries.append(entry)

            # Append to output file immediately
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, "a") as f:
                json.dump(entry, f)
                f.write("\n")
            print(f"Saved to {args.output}")

    print(f"\n{'='*70}")
    print(f"Labeled {len(entries)} queries. Saved to {args.output}")


if __name__ == "__main__":
    main()
