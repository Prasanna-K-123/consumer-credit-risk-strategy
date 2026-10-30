"""Acquire, validate and reconcile a pinned release of public bankcard data."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
import sqlite3
import urllib.error
import urllib.request
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.resources import files
from pathlib import Path

from .schema import FOOTER, SCHEMAS, Metric

DEFAULT_MANIFEST = files("us_card_market").joinpath("data/source_manifest.json")
_NUMBER = r"-?(?:0|[1-9]\d*|[1-9]\d{0,2}(?:,\d{3})+)(?:\.\d+)?"
_PATTERNS = {
    "currency": re.compile(r"\$(" + _NUMBER + r")"),
    "number": re.compile(r"(" + _NUMBER + r")"),
    "percent": re.compile(r"(" + _NUMBER + r")\s*%"),
}


class MarketDataError(ValueError):
    """A source violates its declared provenance or data contract."""


def quarter_index(quarter: str) -> int:
    match = re.fullmatch(r"(\d{4})Q([1-4])", quarter)
    if not match:
        raise MarketDataError(f"Invalid quarter: {quarter!r}")
    return int(match[1]) * 4 + int(match[2]) - 1


def expected_quarters(first: str, last: str) -> tuple[str, ...]:
    lo, hi = quarter_index(first), quarter_index(last)
    if hi < lo:
        raise MarketDataError("Quarter range is reversed")
    return tuple(f"{i // 4}Q{i % 4 + 1}" for i in range(lo, hi + 1))


def parse_number(raw: str, metric: Metric, quarter: str, dataset: str) -> float | None:
    text = raw.strip()
    if text == "null":
        if (dataset, metric.key, quarter) == ("balances", "line_decrease_pct", "2012Q3"):
            return None
        raise MarketDataError(f"Unexpected null at {dataset}/{quarter}/{metric.key}")
    match = _PATTERNS[metric.style].fullmatch(text)
    if not match:
        raise MarketDataError(f"Malformed {metric.style} value {raw!r} at {dataset}/{quarter}/{metric.key}")
    value = float(match[1].replace(",", ""))
    if not math.isfinite(value):
        raise MarketDataError(f"Nonfinite value at {dataset}/{quarter}/{metric.key}")
    if metric.is_share and not 0 <= value <= 100:
        raise MarketDataError(f"Share outside [0,100] at {dataset}/{quarter}/{metric.key}")
    return value


def parse_source_csv(blob: bytes, dataset: str, *, quarters: tuple[str, ...] | None = None) -> list[dict]:
    """Parse the entire source, including its exact schema and footer contract.

    No row is silently discarded. A footer row must match its expected position
    and every other cell on that row must be empty.
    """
    if dataset not in SCHEMAS:
        raise MarketDataError(f"Unknown dataset: {dataset}")
    schema = SCHEMAS[dataset]
    try:
        reader = csv.reader(io.StringIO(blob.decode("utf-8-sig")), strict=True)
        header = next(reader)
        expected_header = ["YRQTR", *(metric.header for metric in schema)]
        if header != expected_header:
            raise MarketDataError(f"{dataset} schema changed; expected exact {len(expected_header)}-column header")
        records, seen, footer_position = [], set(), 0
        footer_started = False
        for line_number, row in enumerate(reader, 2):
            if len(row) != len(header):
                raise MarketDataError(f"{dataset} line {line_number}: expected {len(header)} fields, received {len(row)}")
            quarter = row[0]
            if re.fullmatch(r"\d{4}Q[1-4]", quarter) and not footer_started:
                if quarter in seen:
                    raise MarketDataError(f"Duplicate quarter {quarter} in {dataset}")
                if records and quarter_index(quarter) != quarter_index(records[-1]["quarter"]) + 1:
                    raise MarketDataError(f"Unordered or missing quarter before {quarter} in {dataset}")
                record = {"quarter": quarter, "quarter_index": quarter_index(quarter)}
                record.update({metric.key: parse_number(raw, metric, quarter, dataset) for metric, raw in zip(schema, row[1:])})
                records.append(record)
                seen.add(quarter)
            else:
                footer_started = True
                if footer_position >= len(FOOTER) or row[0] != FOOTER[footer_position] or any(row[1:]):
                    raise MarketDataError(f"Unexpected row/footer at {dataset} line {line_number}: {row[0]!r}")
                footer_position += 1
        if not records or footer_position != len(FOOTER):
            raise MarketDataError(f"{dataset} is empty or its footer is incomplete")
        observed = tuple(record["quarter"] for record in records)
        if quarters is not None and observed != tuple(quarters):
            raise MarketDataError(f"{dataset} does not match the pinned quarter coverage")
        return records
    except (UnicodeDecodeError, csv.Error, StopIteration) as exc:
        raise MarketDataError(f"Cannot parse {dataset} source CSV: {exc}") from exc


def _verify_blob(blob: bytes, spec: dict, dataset: str) -> None:
    actual_hash = hashlib.sha256(blob).hexdigest()
    if len(blob) != spec["bytes"] or actual_hash != spec["sha256"]:
        raise MarketDataError(f"{dataset} source bytes/hash differ from the pinned release; review the changed source before updating the manifest (received sha256={actual_hash})")


def acquire_source(spec: dict, dataset: str, cache_dir: Path) -> tuple[bytes, dict]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / f"{dataset}-{spec['sha256']}.csv"
    metadata_path = target.with_suffix(".metadata.json")
    if target.exists():
        blob = target.read_bytes()
        _verify_blob(blob, spec, dataset)
        metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {"retrieved_at_utc": None, "retrieval_note": "Verified existing cache; original retrieval time unavailable"}
        return blob, metadata
    try:
        request = urllib.request.Request(spec["url"], headers={"User-Agent": "ConsumerCreditRiskResearch/1.0"})
        with urllib.request.urlopen(request, timeout=30) as response:
            blob = response.read(1_000_001)
        _verify_blob(blob, spec, dataset)
    except (OSError, urllib.error.URLError) as exc:
        raise MarketDataError(f"Cannot acquire official {dataset} source: {exc}. No sample data was substituted.") from exc
    metadata = {"retrieved_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "source_url": spec["url"], "sha256": spec["sha256"], "bytes": len(blob)}
    temporary = target.with_suffix(".partial")
    temporary.write_bytes(blob)
    temporary.replace(target)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return blob, metadata


def _write_csv(path: Path, rows: list[dict], columns: list[str] | None = None) -> None:
    if not rows and columns is None:
        raise MarketDataError(f"Cannot write an untyped empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _make_database(path: Path, records: dict[str, list[dict]], manifest: dict) -> sqlite3.Connection:
    # A rebuild replaces only its generated database, never a downloaded source.
    if path.exists():
        path.unlink()
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys=ON")
    db.executescript("""
        CREATE TABLE source_release(dataset TEXT PRIMARY KEY, release TEXT NOT NULL,
          source_url TEXT NOT NULL, sha256 TEXT NOT NULL, expected_quarters INTEGER NOT NULL);
        CREATE TABLE metric_catalog(metric TEXT PRIMARY KEY, dataset TEXT NOT NULL,
          source_header TEXT NOT NULL, unit TEXT NOT NULL, population TEXT NOT NULL,
          aggregation TEXT NOT NULL, FOREIGN KEY(dataset) REFERENCES source_release(dataset));
        CREATE TABLE observations(dataset TEXT NOT NULL, quarter TEXT NOT NULL,
          quarter_index INTEGER NOT NULL, metric TEXT NOT NULL, value REAL,
          PRIMARY KEY(quarter,metric), FOREIGN KEY(metric) REFERENCES metric_catalog(metric),
          FOREIGN KEY(dataset) REFERENCES source_release(dataset));
    """)
    for dataset, schema in SCHEMAS.items():
        spec = manifest["sources"][dataset]
        db.execute("INSERT INTO source_release VALUES(?,?,?,?,?)", (dataset, manifest["release"], spec["url"], spec["sha256"], spec["quarters"]))
        for metric in schema:
            db.execute("INSERT INTO metric_catalog VALUES(?,?,?,?,?,?)", (metric.key, dataset, metric.header, metric.unit, metric.population, metric.aggregation))
        db.executemany("INSERT INTO observations VALUES(?,?,?,?,?)", ((dataset, row["quarter"], row["quarter_index"], metric.key, row[metric.key]) for row in records[dataset] for metric in schema))
    expressions = [f"MAX(CASE WHEN metric='{metric.key}' THEN value END) AS {metric.key}" for schema in SCHEMAS.values() for metric in schema]
    db.execute("CREATE VIEW quarterly_source AS SELECT quarter,quarter_index," + ",".join(expressions) + " FROM observations GROUP BY quarter,quarter_index")
    db.executescript((Path(__file__).parent / "sql" / "quarterly_measures.sql").read_text())
    db.commit()
    return db


def _check(checks: list[dict], name: str, failures: int, detail: str, severity: str = "error") -> None:
    checks.append({"check": name, "status": "pass" if failures == 0 else severity, "failures": failures, "detail": detail})


def _quality_checks(db: sqlite3.Connection, records: dict[str, list[dict]], manifest: dict) -> list[dict]:
    checks = []
    for dataset, schema in SCHEMAS.items():
        expected_rows = manifest["sources"][dataset]["quarters"]
        count = db.execute("SELECT COUNT(*),COUNT(DISTINCT quarter),COUNT(DISTINCT metric) FROM observations WHERE dataset=?", (dataset,)).fetchone()
        _check(checks, f"{dataset}_sql_population", int(tuple(count) != (expected_rows * len(schema), expected_rows, len(schema))), f"{count[0]} observations; {count[1]} distinct quarters; {count[2]} metrics")
        expected_sum = sum(row[schema[0].key] for row in records[dataset])
        actual_sum = db.execute("SELECT SUM(value) FROM observations WHERE metric=?", (schema[0].key,)).fetchone()[0]
        _check(checks, f"{dataset}_python_sql_checksum", int(not math.isclose(expected_sum, actual_sum, abs_tol=1e-7)), "Independent in-memory and SQL sums of the first source metric agree; summing stocks is an integrity checksum, not an economic measure")
    nulls = [tuple(row) for row in db.execute("SELECT dataset,quarter,metric FROM observations WHERE value IS NULL")]
    _check(checks, "explicit_missingness", int(nulls != [("balances", "2012Q3", "line_decrease_pct")]), "Exactly one source null: 2012Q3 credit-line-decrease share; preserved as NULL, never zero")
    for basis in ("accounts", "balances"):
        bad = db.execute(f"SELECT COUNT(*) FROM quarterly_source WHERE NOT(dpd90_{basis}_pct <= dpd60_{basis}_pct AND dpd60_{basis}_pct <= dpd30_{basis}_pct)").fetchone()[0]
        _check(checks, f"{basis}_cumulative_delinquency", bad, "Cumulative 90+ <= 60+ <= 30+ shares; stages are not added")
    for prefix, percentiles in (("active_balance", (50,75,90)), ("limit", (50,75,90)), ("active_utilization", (50,75,90)), ("score", (10,25,50)), ("new_limit", (50,75,90)), ("new_score", (10,25,50))):
        suffix = "_usd" if "balance" in prefix or "limit" in prefix else "_pct" if "utilization" in prefix else ""
        a,b,c = (f"{prefix}_p{p}{suffix}" for p in percentiles)
        bad = db.execute(f"SELECT COUNT(*) FROM quarterly_source WHERE NOT({a}<={b} AND {b}<={c})").fetchone()[0]
        _check(checks, f"{prefix}_percentile_order", bad, "Ordered source percentiles within each quarter and population")
    _check(checks, "payment_shares_partial_partition", db.execute("SELECT COUNT(*) FROM quarterly_source WHERE minimum_payment_pct+partial_payment_pct+full_payment_pct>100.02").fetchone()[0], "Reported payment categories need not total 100%; residual is not classified as nonpayment")
    _check(checks, "positive_ratio_denominators", db.execute("SELECT COUNT(*) FROM quarterly_source WHERE accounts_mn<=0 OR commitments_bn<=0 OR new_accounts_mn<=0 OR new_sub660_accounts_pct<=0 OR new_sub660_accounts_pct>=100").fetchone()[0], "Derived averages and complementary-group calculations have valid denominators")
    group_failures = db.execute("""
        SELECT COUNT(*) FROM quarterly_measures
        WHERE ABS(new_accounts_mn*new_sub660_accounts_pct/100.0
          + new_accounts_mn*(100.0-new_sub660_accounts_pct)/100.0-new_accounts_mn)>1e-9
        OR ABS(new_commitments_bn*new_sub660_commitments_pct/100.0
          + new_commitments_bn*(100.0-new_sub660_commitments_pct)/100.0-new_commitments_bn)>1e-9
        OR ABS(new_sub660_avg_limit_usd*new_sub660_accounts_pct/100.0
          +new_ge660_avg_limit_usd*(100.0-new_sub660_accounts_pct)/100.0-new_avg_initial_limit_usd)>1e-8
    """).fetchone()[0]
    _check(checks, "origination_group_reconciliation", group_failures, "Complementary original-score groups reconstruct total new accounts, commitments and account-weighted average initial limit")
    _check(checks, "sqlite_integrity", int(db.execute("PRAGMA integrity_check").fetchone()[0] != "ok"), "SQLite integrity_check")
    _check(checks, "sqlite_foreign_keys", len(db.execute("PRAGMA foreign_key_check").fetchall()), "Metric and dataset references reconcile")
    _check(checks, "metric_dataset_lineage", db.execute("SELECT COUNT(*) FROM observations o JOIN metric_catalog m ON o.metric=m.metric WHERE o.dataset<>m.dataset").fetchone()[0], "Each observation belongs to its metric's declared source dataset")
    _check(checks, "eligible_score_range", db.execute("SELECT COUNT(*) FROM observations WHERE metric IN ('score_p10','score_p25','score_p50','new_score_p10','new_score_p25','new_score_p50') AND (value<150 OR value>950)").fetchone()[0], "Published score-percentile populations use eligible credit scores from 150 to 950")
    # Independent reconstruction of one weighted subgroup calculation.
    failures = 0
    origins = {row["quarter"]: row for row in records["originations"]}
    for row in db.execute("SELECT quarter,new_sub660_avg_limit_usd FROM quarterly_measures"):
        original = origins[row["quarter"]]
        denominator = original["new_accounts_mn"] * original["new_sub660_accounts_pct"]
        if denominator <= 0 or row["new_sub660_avg_limit_usd"] is None:
            failures += 1
            continue
        expected = original["new_commitments_bn"] * 1000 * original["new_sub660_commitments_pct"] / denominator
        failures += int(not math.isclose(expected, row["new_sub660_avg_limit_usd"], rel_tol=1e-12))
    _check(checks, "sub660_average_limit_reconciliation", failures, "SQL implied average opening limit agrees with independent Python calculation from rounded source aggregates")
    return checks


def _summary(rows: list[dict], manifest: dict, checks: list[dict]) -> dict:
    latest = rows[-1]
    prior = rows[-5]
    indicators = (
        ("dpd90_balances_pct", "90+ delinquency by balance"),
        ("nco_annualized_pct", "Annualized net charge-off rate"),
        ("minimum_payment_pct", "Active accounts making minimum payment"),
        ("new_sub660_accounts_pct", "New accounts with original score below 660"),
        ("new_sub660_commitments_pct", "New commitments with original score below 660"),
    )
    changes = [{"metric": key, "label": label, "unit": "percent", "current": latest[key], "year_earlier": prior[key], "yoy_change_pp": round(latest[key]-prior[key], 6)} for key,label in indicators]
    return {
        "title": "US consumer bankcard market risk pack",
        "source_release": manifest["release"],
        "source_published_on": manifest["source_published_on"],
        "series_status": manifest["historical_series_status"],
        "coverage": {"first_quarter": rows[0]["quarter"], "last_quarter": latest["quarter"], "quarters": len(rows), "raw_source_metrics": sum(map(len, SCHEMAS.values()))},
        "quality": {"checks": len(checks), "errors": sum(c["status"] == "error" for c in checks), "source_nulls": 1},
        "latest": latest,
        "year_on_year_indicators": changes,
        "interpretation": [
            "Balance-based and account-based delinquency use different denominators; a higher balance-based rate indicates concentration in larger outstanding balances, not a measured causal effect.",
            "Compare the below-660 shares of new accounts and new commitments alongside the implied average initial limits. These are origination mix and line-size diagnostics, not subsequent default outcomes.",
            "Payment categories leave an explicitly unclassified residual. That residual combines several source exclusions and must not be called the nonpayer share.",
            "Use changes in these aggregates to frame questions for internal vintage, score-band and account-transition analyses; the public aggregates cannot answer those account-level questions.",
        ],
        "limitations": [
            "Published history can be restated for reporting-panel changes and corrections. No original-vintage, prospective backtest, causal-policy or model-performance claim is made.",
            "Coverage is large-bank consumer bankcards, not every US borrower or all card products. No Citi-specific performance can be inferred.",
            "Amounts are nominal and not seasonally adjusted. Year-on-year comparisons reduce simple seasonal mismatch but do not remove all seasonality or composition effects.",
            "Net charge-off rate is annualized using the three-month average balance; quarter-end balances cannot reproduce exact dollar charge-offs.",
            "Implied dollar averages use rounded published aggregates, are estimates and are not account medians. Percentages are stored on a 0-to-100 scale, not as fractions.",
        ],
    }


def build_market_pack(output_dir: str | Path, cache_dir: str | Path | None = None, source_manifest: str | Path = DEFAULT_MANIFEST) -> dict:
    """Create an auditable market pack or fail explicitly on changed/missing data."""
    output = Path(output_dir)
    cache = Path(cache_dir) if cache_dir is not None else Path.cwd() / ".cache" / "us_card_market"
    manifest = json.loads(Path(source_manifest).read_text())
    if set(manifest["sources"]) != set(SCHEMAS):
        raise MarketDataError("Manifest must identify exactly the balances and originations sources")
    records, downloads = {}, {}
    for dataset, spec in manifest["sources"].items():
        quarters = expected_quarters(spec["first_quarter"], spec["last_quarter"])
        if len(quarters) != spec["quarters"]:
            raise MarketDataError(f"{dataset} manifest quarter count is inconsistent")
        blob, metadata = acquire_source(spec, dataset, cache)
        records[dataset] = parse_source_csv(blob, dataset, quarters=quarters)
        downloads[dataset] = metadata
    if [row["quarter"] for row in records["balances"]] != [row["quarter"] for row in records["originations"]]:
        raise MarketDataError("Balance and origination quarter populations differ")
    output.mkdir(parents=True, exist_ok=True)
    db = _make_database(output / "market.sqlite", records, manifest)
    try:
        checks = _quality_checks(db, records, manifest)
        if any(check["status"] == "error" for check in checks):
            raise MarketDataError("Market reconciliations failed: " + ", ".join(c["check"] for c in checks if c["status"] == "error"))
        quarterly = [dict(row) for row in db.execute("SELECT * FROM quarterly_measures ORDER BY quarter_index")]
        report_rows = [dict(row) for row in db.execute((Path(__file__).parent / "sql" / "risk_report.sql").read_text())]
        group_rows = [dict(row) for row in db.execute((Path(__file__).parent / "sql" / "origination_groups.sql").read_text())]
        summary = _summary(quarterly, manifest, checks)
        _write_csv(output / "quarterly.csv", quarterly)
        _write_csv(output / "risk_indicators.csv", report_rows)
        _write_csv(output / "origination_groups.csv", group_rows)
        _write_csv(output / "quality_checks.csv", checks)
        catalog = [{"dataset": dataset, **asdict(metric)} for dataset,schema in SCHEMAS.items() for metric in schema]
        (output / "metric_dictionary.json").write_text(json.dumps(catalog, indent=2) + "\n")
        (output / "report.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
        (output / "sources.json").write_text(json.dumps({**manifest, "downloads": downloads}, indent=2) + "\n")
        (output / "DATA_TERMS.txt").write_text("Research outputs derived from Federal Reserve Bank of Philadelphia public aggregate data.\nSource data and source-derived series are subject to the source terms; they are not covered by the software license.\n" + manifest["terms_page"] + "\nNo Federal Reserve endorsement is implied.\n")
        return summary
    finally:
        db.close()
