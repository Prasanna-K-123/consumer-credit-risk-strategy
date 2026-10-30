"""Run with python -m us_card_market --output reference/us_market."""

import argparse
import json
import sys

from .pipeline import DEFAULT_MANIFEST, MarketDataError, build_market_pack


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate pinned public US bankcard aggregates and create a research risk pack.")
    parser.add_argument("--output", default="reference/us_market")
    parser.add_argument("--cache", default=None)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    args = parser.parse_args()
    try:
        report = build_market_pack(args.output, args.cache, args.manifest)
    except (MarketDataError, OSError, json.JSONDecodeError) as exc:
        print(f"Market pack failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"output": args.output, "source_release": report["source_release"], "coverage": report["coverage"], "quality": report["quality"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
