#!/usr/bin/env python3
"""Run the full MARS pipeline for one or more benchmark queries.

Usage:
    python scripts/run_mars.py                        # all queries
    python scripts/run_mars.py --queries Query1          # specific query
    python scripts/run_mars.py --output-dir results    # custom output root
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.utils.ablation_utils import load_ablation_queries
from src.runner import initialize, run_query

def main():
    parser = argparse.ArgumentParser(description="Run the full MARS pipeline")
    parser.add_argument(
        "--queries", default=None,
        help="Comma-separated query names (default: all queries in config/queries.yaml)",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Root output directory (default: results)",
    )
    parser.add_argument(
        "--override", default=None,
        help="Path to a YAML override file deep-merged on top of config/config.yaml",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or "results"

    config = load_config(override_path=args.override)

    queries = load_ablation_queries()

    if args.queries:
        selected = {q.strip() for q in args.queries.split(",")}
        queries = [q for q in queries if q["name"] in selected]
        if not queries:
            print(f"ERROR: None of the specified queries found: {args.queries}")
            available = [q["name"] for q in load_ablation_queries()]
            print(f"  Available: {', '.join(available)}")
            sys.exit(1)

    print(f"MARS Full Pipeline")
    print(f"Queries: {', '.join(q['name'] for q in queries)}")
    _backend = "OpenAI API" if "openai.com" in config["llm"].get("base_url", "") else "local endpoint"
    print(f"Model:   {config['llm']['model_name']} ({_backend})")
    print(f"Output:  {output_dir}/")
    print()

    components = initialize(config)
    print()

    for i, query in enumerate(queries, 1):
        name = query["name"]
        query_dir = str(PROJECT_ROOT / output_dir / name)
        print(f"\n{'='*70}")
        print(f"[{i}/{len(queries)}] {name}")
        print(f"{'='*70}")
        run_query(components, query, query_dir)

    print(f"\nAll {len(queries)} queries complete.")


if __name__ == "__main__":
    main()
