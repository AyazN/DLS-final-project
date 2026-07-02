from __future__ import annotations

import argparse

from .contracts import Query
from .pipeline import EmptyRetriever, SearchPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aise", description="AI model card search CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    search_parser = subparsers.add_parser("search", help="Search relevant AI model cards")
    search_parser.add_argument("query", help="Natural language query")
    search_parser.add_argument("--top-k", type=int, default=10, help="Number of results")

    return parser


def run_search(query_text: str, top_k: int) -> int:
    pipeline = SearchPipeline(retriever=EmptyRetriever())
    results = pipeline.search(Query(text=query_text, top_k=top_k))

    if not results:
        print("No results yet. Connect a retriever implementation in src/aise/pipeline.py.")
        return 0

    for result in results:
        print(f"{result.rank}. {result.title} [{result.score:.4f}]")
        print(f"   model_id={result.model_id}")
        if result.snippet:
            print(f"   {result.snippet}")

    return 0


def main() -> int:
    args = build_parser().parse_args()

    if args.command == "search":
        return run_search(args.query, args.top_k)

    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
