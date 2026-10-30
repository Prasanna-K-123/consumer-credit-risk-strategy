"""Command-line entry point; source/refits are explicit and outputs are reviewable."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
import threadpoolctl

from . import __version__
from .constants import FEATURES, HISTORICAL, MODEL_NAMES, PROTOCOL_SHA256, SOURCE_SHA256
from .contracts import feature_frame, source_ids
from .data import acquire, load_protocol, load_source, sha256
from .models import fit_models
from .replay import dump, historical_integrity, run_replay, verify_replay, write_queue_outputs


def model_identity() -> str:
    names = ("constants.py", "contracts.py", "data.py", "evaluation.py", "woe.py", "models.py", "runtime.py")
    hashes = {name: sha256(Path(__file__).parent / name) for name in names}
    value = {"implementation_sha256": hashes, "source_sha256": SOURCE_SHA256,
             "protocol_sha256": PROTOCOL_SHA256, "environment": scientific_environment()}
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:16]


def scientific_environment() -> dict:
    return {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
            "scipy": scipy.__version__, "sklearn": sklearn.__version__, "threadpoolctl": threadpoolctl.__version__}


def score_clients(data: Path, input_file: Path, output: Path) -> dict:
    """Refit trusted original models; never deserialize a client-supplied model."""
    load_protocol()
    frame = pd.read_csv(input_file)
    if "source_id" not in frame.columns or frame.columns.duplicated().any():
        raise ValueError("Scoring CSV needs unique source_id plus the 20 exact feature columns")
    ids = source_ids(frame.source_id.to_numpy(), len(frame))
    features = feature_frame(frame.drop(columns="source_id"))
    output = Path(output)
    if output.exists() or output.with_suffix(output.suffix + ".json").exists():
        raise FileExistsError("Use a fresh scoring output path")
    fitted = fit_models(load_source(data))
    probabilities = fitted.predict(features)
    raw = fitted.raw_predictions(features)
    results = pd.DataFrame({"source_id": ids, **probabilities})
    for name in MODEL_NAMES:
        results["raw_" + name] = raw[name]
        results[name + "_review_flag"] = (probabilities[name] >= fitted.thresholds[name]).astype(int)
        results[name + "_threshold"] = fitted.thresholds[name]
    explanation = fitted.scorecard_contributions(features)
    results = pd.concat([results.reset_index(drop=True), explanation.reset_index(drop=True)], axis=1)
    results["model_version"] = model_identity()
    output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output, index=False, float_format="%.12g")
    receipt = {
        "model_version": model_identity(), "source_sha256": SOURCE_SHA256, "protocol_sha256": PROTOCOL_SHA256,
        "environment": scientific_environment(),
        "native_thread_limit": 2,
        "input_sha256": sha256(input_file), "output_sha256": sha256(output), "rows": len(results),
        "thresholds": fitted.thresholds,
        "execution": "Original fixed models refitted from checksum-verified source; no external model artifact loaded",
        "scope": "Historical Taiwan card-client models; applying them to supplied profiles is an illustration, not validated current or US lending advice",
        "review_flag": "Retained FN5/FP1 illustrative error-weight threshold; not an approve/reject instruction",
        "explanations": "Exact calibrated scorecard log-odds contributions unless raw probability clipping is flagged; not causal effects or legal adverse-action reasons",
    }
    dump(output.with_suffix(output.suffix + ".json"), receipt)
    return receipt


def example_clients(output: Path) -> dict:
    output = Path(output)
    if output.exists():
        raise FileExistsError("Use a fresh example output path")
    rows = []
    for source_id, age, limit, status, balance in [(900001, 28, 100000, 0, 15000), (900002, 45, 200000, 2, 90000), (900003, 62, 60000, -1, 0)]:
        row = {"source_id": source_id, "LIMIT_BAL": limit, "AGE": age}
        row.update({name: status for name in FEATURES if name.startswith("PAY_") and not name.startswith("PAY_AMT")})
        row.update({name: balance for name in FEATURES if name.startswith("BILL_AMT")})
        row.update({name: int(balance * 0.05) for name in FEATURES if name.startswith("PAY_AMT")})
        rows.append(row)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows)[["source_id", *FEATURES]].to_csv(output, index=False)
    return {"path": str(output), "rows": len(rows), "source": "Deliberately synthetic demonstration profiles; no actual people"}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Historical consumer-card models, full replay and explicit review-capacity scenarios")
    result.add_argument("--version", action="version", version=__version__)
    commands = result.add_subparsers(dest="command", required=True)
    acquire_command = commands.add_parser("acquire", help="Download and hash-check the official UCI XLS")
    acquire_command.add_argument("--data", type=Path, default=Path("runtime/uci_card_clients.xls"))
    replay_command = commands.add_parser("replay", help="Refit the original study and verify every historical scientific artifact")
    replay_command.add_argument("--data", type=Path, default=Path("runtime/uci_card_clients.xls"))
    replay_command.add_argument("--output", type=Path, default=Path("work/current"))
    replay_command.add_argument("--historical", type=Path, default=HISTORICAL, help="Immutable reference directory; supply checkout path when installed elsewhere")
    verify_command = commands.add_parser("verify", help="Compare an executed replay's artifacts; this command itself does not refit")
    verify_command.add_argument("--output", type=Path, default=Path("work/current"))
    verify_command.add_argument("--historical", type=Path, default=HISTORICAL)
    reference_command = commands.add_parser("check-reference", help="Check immutable original source and result hashes without source download")
    reference_command.add_argument("--historical", type=Path, default=HISTORICAL)
    score_command = commands.add_parser("score", help="Refit trusted models, then score a strict 20-feature client CSV")
    score_command.add_argument("--data", type=Path, default=Path("runtime/uci_card_clients.xls"))
    score_command.add_argument("--input", type=Path, required=True)
    score_command.add_argument("--output", type=Path, required=True)
    queue_command = commands.add_parser("queue", help="Rank supplied historical predictions at the fixed review capacities")
    queue_command.add_argument("--predictions", type=Path, required=True)
    queue_command.add_argument("--output", type=Path, required=True)
    example_command = commands.add_parser("example", help="Write three clearly synthetic example input profiles")
    example_command.add_argument("--output", type=Path, required=True)
    return result


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "acquire":
            result = acquire(args.data)
        elif args.command == "replay":
            result = run_replay(args.data, args.output, args.historical)
        elif args.command == "verify":
            result = verify_replay(args.output, args.historical)
        elif args.command == "check-reference":
            result = {"passed": True, "historical_sha256": historical_integrity(args.historical)}
        elif args.command == "score":
            result = score_clients(args.data, args.input, args.output)
        elif args.command == "queue":
            result = write_queue_outputs(args.predictions, args.output, require_fresh=True)
        else:
            result = example_clients(args.output)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except (ValueError, OSError, KeyError) as error:
        print(f"consumer_cards: {error}", file=sys.stderr)
        return 2
