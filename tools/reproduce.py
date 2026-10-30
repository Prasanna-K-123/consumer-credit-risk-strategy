"""Rebuild the evidence chain from official sources into a fresh working folder."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for name, value in (("OPENBLAS_NUM_THREADS", "1"), ("OMP_NUM_THREADS", "2"), ("MKL_NUM_THREADS", "1")):
    os.environ.setdefault(name, value)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_output(path: Path) -> Path:
    path = path.resolve()
    protected = tuple(ROOT / name for name in ("reference", "consumer_cards", "card_audit", "card_strategy", "us_card_market", "reports", "tools", "docs", "tests"))
    if any(path == item or item in path.parents or path in item.parents for item in protected):
        raise ValueError("Reproduction output must not overlap source or published evidence")
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError("Choose a fresh output directory; completed and partial runs are preserved")
    path.mkdir(parents=True, exist_ok=True)
    return path


def reproduce(output: Path, cache: Path, *, with_reports: bool = True) -> dict:
    # Import after the resource/thread setup so direct CLI and installed modules
    # use the same scientific settings.
    import numpy
    import pandas
    import scipy
    import sklearn
    from consumer_cards.data import acquire
    from consumer_cards.replay import run_replay, _compare, _compare_csv
    from card_audit.review import audit
    from card_strategy.pipeline import run_strategy
    from tools.audit_strategy import audit as audit_strategy
    from us_card_market.pipeline import build_market_pack

    output = prepare_output(output)
    cache = cache.resolve()
    started = time.perf_counter()
    source = cache / "uci_card_clients.xls"
    acquisition = acquire(source)
    card = output / "reference/card_clients/replay"
    replay = run_replay(source, card, ROOT / "reference/card_clients/historical")
    sql = output / "reference/card_clients/sql_review"
    review = audit(card / "holdout_predictions.csv", card / "partition_manifest.csv", card / "summary.json", sql,
                   queue_reference=card / "decision_queue.csv", queue_assignments=card / "queue_assignments.csv")
    if review.get("status") != "PASS" or not review.get("queue_assignment_reference_checked"):
        raise ValueError("Independent SQL and queue-membership review did not pass")
    strategy = output / "reference/card_strategy"
    strategy_result = run_strategy(source, strategy)
    strategy_review = audit_strategy(strategy, source)
    if strategy_review.get("status") != "PASS" or not strategy_review.get("source_labels_reconciled"):
        raise ValueError("Independent strategy review did not reconcile source outcomes")
    (strategy / "independent_review.json").write_text(json.dumps(strategy_review, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    strategy_reference = ROOT / "reference/card_strategy"
    strategy_fields = _compare(json.loads((strategy_reference / "summary.json").read_text()), strategy_result["summary"])
    strategy_cells = sum(_compare_csv(strategy_reference / name, strategy / name)
                         for name in ("predictions.csv", "policy_sensitivity.csv", "capacity.csv", "leaf_segments.csv"))
    for name in ("tree.txt", "protocol.json"):
        if (strategy_reference / name).read_bytes() != (strategy / name).read_bytes():
            raise ValueError("Published strategy definition changed: " + name)
    market = output / "reference/us_market"
    market_report = build_market_pack(market, cache / "us_card_market")
    comparable = ("quarterly.csv", "risk_indicators.csv", "origination_groups.csv", "quality_checks.csv", "report.json", "metric_dictionary.json", "DATA_TERMS.txt")
    for name in comparable:
        if (market / name).read_bytes() != (ROOT / "reference/us_market" / name).read_bytes():
            raise ValueError("Published US market evidence changed: " + name)
    # Retrieval timestamps are execution provenance, not an economic result.
    source_identities = json.loads((market / "sources.json").read_text())
    published_identities = json.loads((ROOT / "reference/us_market/sources.json").read_text())
    source_identities.pop("downloads", None)
    published_identities.pop("downloads", None)
    if source_identities != published_identities:
        raise ValueError("US market source identities changed")
    if with_reports:
        from tools.build_report import build
        from tools.build_strategy_report import build as build_strategy
        build(output, output / "reports")
        build_strategy(strategy, output / "reports")
    files = {str(path.relative_to(output)): digest(path) for path in sorted(output.rglob("*")) if path.is_file()}
    sources = {}
    for package in ("consumer_cards", "card_audit", "card_strategy", "us_card_market", "tools"):
        for path in sorted((ROOT / package).rglob("*")):
            if path.is_file() and path.suffix in {".py", ".sql", ".json"}:
                sources[str(path.relative_to(ROOT))] = digest(path)
    receipt = {
        "status": "PASS", "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "environment": {"python": platform.python_version(), "numpy": numpy.__version__, "pandas": pandas.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__},
        "acquisition": acquisition,
        "card_replay": replay["verification"],
        "independent_sql": {"status": review["status"], "controls": len(review["controls"]), "queue_assignments_compared": review["queue_assignments_compared"]},
        "strategy_extension": {"status": strategy_result["verification"]["status"], "independent_review": strategy_review,
                               "published_summary_fields_compared": strategy_fields, "published_csv_cells_compared": strategy_cells,
                               "published_float_absolute_tolerance": 1e-7, "exact_integer_and_identity_checks": True,
                               "scope": "Retrospective CART segmentation and five assumed error-cost ratios on the known historical cohort; no fresh test or monetary benefit claim."},
        "us_market": {"coverage": market_report["coverage"], "quality": market_report["quality"], "published_deterministic_files_compared": len(comparable)},
        "presentation_generated": with_reports,
        "implementation_sha256": sources, "output_sha256": files,
        "scope": "Executed source acquisition or hash-verified cache, historical fixed model refits, calibration, policy evaluation, bootstrap, independent SQL, retrospective tree/policy sensitivity and market-source reconstruction. No new unseen test or regulatory approval is implied.",
    }
    (output / "execution_receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "work/current")
    parser.add_argument("--cache", type=Path, default=ROOT / ".cache")
    parser.add_argument("--no-reports", action="store_true", help="Run the numerical evidence chain without matplotlib reporting")
    args = parser.parse_args()
    result = reproduce(args.output, args.cache, with_reports=not args.no_reports)
    print(json.dumps({"status": result["status"], "output": str(args.output), "elapsed_seconds": result["elapsed_seconds"], "card_replay": result["card_replay"], "independent_sql": result["independent_sql"], "us_market": result["us_market"]}, indent=2))


if __name__ == "__main__":
    main()
