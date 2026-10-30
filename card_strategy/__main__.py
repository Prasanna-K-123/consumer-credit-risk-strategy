"""Execute the fixed retrospective strategy extension."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pipeline import run_strategy


def main() -> None:
    parser = argparse.ArgumentParser(description="Fixed shallow CART and policy-cost sensitivity on the known historical card cohort; acquire the pinned XLS separately.")
    parser.add_argument("--source", required=True, type=Path, help="Previously acquired, SHA-verified UCI XLS source file")
    parser.add_argument("--output", required=True, type=Path, help="Fresh output directory outside published references")
    parser.add_argument("--historical", type=Path, help="Immutable historical reference directory (required outside the source checkout)")
    args = parser.parse_args()
    result = run_strategy(args.source, args.output, args.historical)
    print(json.dumps({"output": result["output"], "verification": result["verification"]}, indent=2))


if __name__ == "__main__":
    main()
