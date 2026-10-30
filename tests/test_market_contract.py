"""Source contracts use explicitly synthetic fixtures; no network is needed."""

import csv
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from us_card_market.pipeline import (
    DEFAULT_MANIFEST,
    MarketDataError,
    _verify_blob,
    build_market_pack,
    expected_quarters,
    parse_source_csv,
)
from us_card_market.schema import FOOTER, SCHEMAS


def fixture_rows(dataset, quarters=("2026Q2",)):
    schema = SCHEMAS[dataset]
    rows = [["YRQTR", *(metric.header for metric in schema)]]
    for quarter in quarters:
        cells = [quarter]
        for metric in schema:
            if dataset == "balances" and metric.key == "line_decrease_pct" and quarter == "2012Q3":
                cells.append("null")
            elif metric.style == "currency":
                cells.append("$100.00")
            elif metric.style == "percent":
                cells.append("10.00 %")
            elif metric.unit == "credit score":
                cells.append("700")
            else:
                cells.append("100.00")
        rows.append(cells)
    rows.extend([[text] + [""] * len(schema) for text in FOOTER])
    return rows


def encoded(rows):
    stream = io.StringIO(newline="")
    csv.writer(stream, lineterminator="\n").writerows(rows)
    return stream.getvalue().encode("utf-8")


class MarketContractTests(unittest.TestCase):
    def test_invalid_ratio_population_fails_with_explanatory_error(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / "cache"
            cache.mkdir()
            manifest = json.loads(DEFAULT_MANIFEST.read_text())
            quarters = expected_quarters("2012Q3", "2026Q2")
            for dataset in SCHEMAS:
                rows = fixture_rows(dataset, quarters)
                if dataset == "originations":
                    column = next(i+1 for i,m in enumerate(SCHEMAS[dataset]) if m.key == "new_accounts_mn")
                    rows[1][column] = "0.00"
                blob = encoded(rows)
                digest = hashlib.sha256(blob).hexdigest()
                manifest["sources"][dataset].update({"sha256": digest, "bytes": len(blob)})
                (cache / f"{dataset}-{digest}.csv").write_bytes(blob)
            manifest_file = root / "manifest.json"
            manifest_file.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(MarketDataError, "positive_ratio_denominators"):
                build_market_pack(root / "pack", cache, manifest_file)

    def test_preserves_permitted_null_and_percent_scale(self):
        rows = parse_source_csv(encoded(fixture_rows("balances", ("2012Q3",))), "balances")
        self.assertIsNone(rows[0]["line_decrease_pct"])
        self.assertEqual(rows[0]["dpd90_balances_pct"], 10.0)

    def test_strict_currency_grouping(self):
        rows = fixture_rows("balances")
        rows[1][1] = "$1,23.00"
        with self.assertRaisesRegex(MarketDataError, "Malformed currency"):
            parse_source_csv(encoded(rows), "balances")

    def test_missing_percentage_suffix_is_not_silently_rescaled(self):
        rows = fixture_rows("balances")
        column = next(i+1 for i,m in enumerate(SCHEMAS["balances"]) if m.key == "dpd90_balances_pct")
        rows[1][column] = "0.10"
        with self.assertRaisesRegex(MarketDataError, "Malformed percent"):
            parse_source_csv(encoded(rows), "balances")

    def test_nan_rejected(self):
        rows = fixture_rows("originations")
        rows[1][2] = "NaN"
        with self.assertRaisesRegex(MarketDataError, "Malformed number"):
            parse_source_csv(encoded(rows), "originations")

    def test_unexpected_null_rejected(self):
        rows = fixture_rows("balances")
        rows[1][1] = "null"
        with self.assertRaisesRegex(MarketDataError, "Unexpected null"):
            parse_source_csv(encoded(rows), "balances")

    def test_empty_source_value_rejected(self):
        rows = fixture_rows("balances")
        rows[1][1] = ""
        with self.assertRaisesRegex(MarketDataError, "Malformed currency"):
            parse_source_csv(encoded(rows), "balances")

    def test_extra_schema_column_rejected(self):
        rows = fixture_rows("balances")
        rows[0].append("undocumented")
        with self.assertRaisesRegex(MarketDataError, "schema changed"):
            parse_source_csv(encoded(rows), "balances")

    def test_truncated_row_rejected(self):
        rows = fixture_rows("balances")
        rows[1].pop()
        with self.assertRaisesRegex(MarketDataError, "expected 36 fields"):
            parse_source_csv(encoded(rows), "balances")

    def test_footer_value_rejected(self):
        rows = fixture_rows("balances")
        rows[-1][2] = "unexpected payload"
        with self.assertRaisesRegex(MarketDataError, "Unexpected row/footer"):
            parse_source_csv(encoded(rows), "balances")

    def test_missing_footer_rejected(self):
        rows = fixture_rows("balances")[:-1]
        with self.assertRaisesRegex(MarketDataError, "footer is incomplete"):
            parse_source_csv(encoded(rows), "balances")

    def test_wrong_quarter_not_treated_as_footer(self):
        rows = fixture_rows("balances")
        rows[1][0] = "2026Q5"
        with self.assertRaisesRegex(MarketDataError, "Unexpected row/footer"):
            parse_source_csv(encoded(rows), "balances")

    def test_duplicate_quarter_rejected(self):
        rows = fixture_rows("balances", ("2026Q1", "2026Q1"))
        with self.assertRaisesRegex(MarketDataError, "Duplicate quarter"):
            parse_source_csv(encoded(rows), "balances")

    def test_internal_missing_quarter_rejected(self):
        rows = fixture_rows("balances", ("2025Q4", "2026Q2"))
        with self.assertRaisesRegex(MarketDataError, "missing quarter"):
            parse_source_csv(encoded(rows), "balances")

    def test_boundary_missing_quarter_rejected(self):
        rows = fixture_rows("balances", ("2026Q2",))
        with self.assertRaisesRegex(MarketDataError, "pinned quarter coverage"):
            parse_source_csv(encoded(rows), "balances", quarters=("2026Q1", "2026Q2"))

    def test_source_mutation_rejected(self):
        original = b"pinned source"
        spec = {"sha256": hashlib.sha256(original).hexdigest(), "bytes": len(original)}
        with self.assertRaisesRegex(MarketDataError, "hash differ"):
            _verify_blob(original + b"changed", spec, "balances")

    def test_synthetic_complete_pack_is_reconciled_and_repeatable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache, output = root / "cache", root / "pack"
            cache.mkdir()
            manifest = json.loads(DEFAULT_MANIFEST.read_text())
            quarters = expected_quarters("2012Q3", "2026Q2")
            for dataset in SCHEMAS:
                blob = encoded(fixture_rows(dataset, quarters))
                digest = hashlib.sha256(blob).hexdigest()
                manifest["sources"][dataset].update({"sha256": digest, "bytes": len(blob)})
                (cache / f"{dataset}-{digest}.csv").write_bytes(blob)
            manifest_file = root / "manifest.json"
            manifest_file.write_text(json.dumps(manifest))
            first = build_market_pack(output, cache, manifest_file)
            first_csv = (output / "quarterly.csv").read_bytes()
            second = build_market_pack(output, cache, manifest_file)
            self.assertEqual(first, second)
            self.assertEqual(first_csv, (output / "quarterly.csv").read_bytes())
            self.assertEqual(first["coverage"]["quarters"], 56)
            self.assertEqual(first["quality"]["errors"], 0)
            self.assertEqual(first["latest"]["new_sub660_avg_limit_usd"], 1000.0)
            self.assertEqual(first["latest"]["unclassified_payment_share_pct"], 70.0)
            self.assertTrue(all(row["yoy_change_pp"] == 0 for row in first["year_on_year_indicators"]))
            with (output / "risk_indicators.csv").open() as stream:
                reports = list(csv.DictReader(stream))
            self.assertEqual(reports[0]["nco_yoy_change_pp"], "")
            self.assertEqual(reports[-1]["nco_yoy_change_pp"], "0.0")


if __name__ == "__main__":
    unittest.main()
