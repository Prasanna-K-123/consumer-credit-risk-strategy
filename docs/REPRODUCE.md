# Reproduce the project

Use Python 3.12 and the exact packages in `requirements.txt`. The complete
checkout contains the historical reference data needed for replay verification.
Official source acquisition requires network access to UCI and the Philadelphia
Fed. A source denial or changed checksum stops the command; no sample dataset
is substituted.

Fitting, prediction and bootstrap calculations automatically limit native thread
pools to two threads and restore the caller's settings on return. The command-line
tools work without manual thread-environment setup; continuous verification also
sets explicit thread caps.

## Complete evidence chain

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python tools/reproduce.py --output work/current
```

The default local cache is `.cache/`. To establish a fresh source acquisition,
choose an unused cache as well as a fresh output:

```bash
python tools/reproduce.py --cache work/fresh-cache --output work/fresh-results
```

The output contains `reference/card_clients/replay`,
`reference/card_clients/sql_review`, `reference/card_strategy`, `reference/us_market`, `reports` and a final
`execution_receipt.json`. SQLite databases can be queried locally and are not
tracked in Git. The receipt records environment versions, current implementation
and output hashes, actual source acquisition status, verification counts and
elapsed time. Partial output has no successful final receipt.

Source/reference identities and counts are exact. Historical floating-point
replay uses absolute tolerance 1e-7 and zero relative tolerance across all
historical scientific fields and numeric CSV cells. The receipt states this
explicitly; numerical reproduction does not mean byte-identical model binaries.
US deterministic economic outputs are compared byte-for-byte. Retrieval times
remain provenance rather than economic results.

Strategy reproduction also compares every published summary field and all
prediction/scenario/leaf CSV cells, using exact identities/counts and absolute
floating-point tolerance 1e-7 with zero relative tolerance. Tree paths and the
recorded protocol must match byte-for-byte. Independent arithmetic and SQLite
checks apply tighter within-run accounting tolerances.

## Individual components

```bash
python -m consumer_cards acquire --data .cache/uci_card_clients.xls
python -m consumer_cards replay --data .cache/uci_card_clients.xls --output work/card-replay
python -m consumer_cards verify --output work/card-replay
python -O -m consumer_cards verify --output work/card-replay
python -m card_audit --predictions work/card-replay/holdout_predictions.csv --partition-manifest work/card-replay/partition_manifest.csv --summary work/card-replay/summary.json --queue-reference work/card-replay/decision_queue.csv --queue-assignments work/card-replay/queue_assignments.csv --output work/sql-review
python -m us_card_market --output work/us-market
python -m card_strategy --source .cache/uci_card_clients.xls --output work/card-strategy
python tools/audit_strategy.py --evidence work/card-strategy --source .cache/uci_card_clients.xls --output work/card-strategy/independent_review.json
```

`verify` checks executed artifacts; `replay` actually refits the historical
models and bootstrap. These are different operations. `check-reference` verifies
immutable history without requiring a download or refit.

## Batch scoring example

```bash
python -m consumer_cards example --output work/example-profiles.csv
python -m consumer_cards score --data .cache/uci_card_clients.xls --input work/example-profiles.csv --output work/example-scores.csv
```

The three profiles are deliberately synthetic examples. Scoring refits trusted
historical models from the verified source rather than loading a supplied
pickle. It emits probabilities, model/version identity, review flags,
scorecard-contribution columns and a sidecar receipt. These outputs illustrate
the interface; they are not validated advice for actual customers.

## Installed package

```bash
python -m pip install .
python -m consumer_cards --help
python -m us_card_market --help
python -m card_audit --help
python -m card_strategy --help
```

Acquisition, scoring, market reporting and independent SQL queries include their
runtime package resources. Historical replay/verification from outside the
checkout needs `--historical /path/to/checkout/reference/card_clients/historical`.
The strategy extension accepts the same explicit `--historical` path for use
from an installed wheel outside the checkout.
The full reproduction and report builders run from the source checkout.

## Reporting and continuous verification

```bash
python tools/build_report.py
python tools/build_report.py --check
python tools/build_strategy_report.py --check
```

Reports are generated from committed executed evidence. `--check` rebuilds to
a temporary directory and rejects stale text or figures. Scientific outputs
are not refitted by the report builder.

The GitHub workflow uses standard `ubuntu-24.04`, read-only repository
permissions, pinned actions and no stored artifact/cache uploads. Existing
account spending controls remain independent of this repository. Local runs
provide the same source-to-report path if hosted execution is unavailable.
